# main.py

import gymnasium as gym
import tianshou as ts
from deep_q_env import NetworkInfluenceEnv
from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
import networkx as nx
import random
import torch
import torch.nn as nn
import numpy as np
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import os
from dataclasses import dataclass
from typing import Dict
import matplotlib.pyplot as plt

# Set up TensorBoard logger
log_path = os.path.join('logs', 'double_dqn')
writer = SummaryWriter(log_path)
writer.add_text("Experiment Info", "Double DQN training with custom environment")
logger = TensorboardLogger(writer)

# Define the neural network
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)  # Output is a scalar Q-value

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value

# Define the custom policy with Double DQN
class CustomDoubleDQNPolicy(BasePolicy):
    def __init__(self, model, target_model, optim, action_dim, k=5, gamma=0.99, epsilon=1.0, tau=0.005):
        super().__init__(action_space=gym.spaces.MultiBinary(action_dim))
        self.model = model
        self.target_model = target_model
        self.optim = optim
        self.k = k
        self.action_dim = action_dim
        self._gamma = gamma
        self.epsilon = epsilon  # Epsilon for epsilon-greedy exploration
        self.tau = tau  # Soft update parameter

    def forward(self, batch, state=None):
        obs = batch.obs  # Shape: [batch_size, state_dim]

        # Convert obs to PyTorch tensor
        device = next(self.model.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.float().to(device)

        batch_size = obs.shape[0]

        act = torch.zeros(batch_size, self.action_dim, device=device)

        for i in range(batch_size):
            state_i = obs[i].clone()  # Current state for the ith sample
            selected_node_indices = []

            for _ in range(self.k):
                # Identify available nodes (not active and not already selected)
                available_indices = (state_i == 0).nonzero(as_tuple=False).flatten().tolist()
                if not available_indices:
                    break  # No more available nodes to select

                # Generate actions for available nodes
                actions_list = []
                for idx in available_indices:
                    action = torch.zeros(self.action_dim, device=device)
                    action[idx] = 1
                    actions_list.append(action)
                actions_tensor = torch.stack(actions_list)  # Shape: [num_available, action_dim]
                states_tensor = state_i.unsqueeze(0).repeat(len(available_indices), 1)  # Shape: [num_available, state_dim]

                # Compute Q-values
                with torch.no_grad():
                    q_values = self.model(states_tensor, actions_tensor).squeeze()  # Shape: [num_available]

                # Apply epsilon-greedy
                if random.random() < self.epsilon:
                    # Exploration: Randomly select an available action
                    selected_idx = random.choice(range(len(available_indices)))
                else:
                    # Exploitation: Select the action with highest Q-value
                    selected_idx = torch.argmax(q_values).item()

                selected_node = available_indices[selected_idx]
                selected_node_indices.append(selected_node)

                # Update state to reflect that the selected node is now active
                state_i[selected_node] = 1

            # Set the selected actions in the action tensor
            act[i, selected_node_indices] = 1

        return Batch(act=act)

    def learn(self, batch, **kwargs):
        # Convert batch data to tensors
        device = next(self.model.parameters()).device

        states = batch.obs
        actions = batch.act
        rewards = batch.rew
        next_states = batch.obs_next
        dones = batch.done

        # Convert to tensors if necessary
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).view(-1)

        self.optim.zero_grad()
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        loss.backward()
        self.optim.step()

        # Soft update of target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return TrainStepResult(loss=loss.item())

    def compute_loss(self, states, actions, rewards, next_states, dones):
        gamma = self._gamma

        # Flatten actions for input
        actions = actions.view(-1, self.action_dim)
        states = states.view(-1, self.action_dim)

        q_values = self.model(states, actions).squeeze()

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            next_q_values = []
            for i in range(next_states.shape[0]):
                next_state = next_states[i]
                available_indices = (next_state == 0).nonzero(as_tuple=False).flatten().tolist()
                if not available_indices:
                    max_q_value = 0.0
                else:
                    # Main network selects the action
                    actions_list = []
                    for idx in available_indices:
                        action = torch.zeros(self.action_dim, device=next_state.device)
                        action[idx] = 1
                        actions_list.append(action)
                    actions_tensor = torch.stack(actions_list)
                    states_tensor = next_state.unsqueeze(0).repeat(len(available_indices), 1)
                    q_vals_main = self.model(states_tensor, actions_tensor).squeeze()
                    best_action_index = torch.argmax(q_vals_main).item()
                    best_action = actions_tensor[best_action_index].unsqueeze(0)
                    best_state = states_tensor[best_action_index].unsqueeze(0)
                    # Target network evaluates the Q-value
                    max_q_value = self.target_model(best_state, best_action).item()
                next_q_values.append(max_q_value)
            next_q_values = torch.tensor(next_q_values, device=states.device)

            target_q_values = rewards + gamma * (1 - dones) * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)
        return loss


@dataclass
class TrainStepResult:
    loss: float

    def get_loss_stats_dict(self) -> Dict[str, float]:
        return {'loss': self.loss}

def train_double_dqn_agent(config, num_actions, num_epochs=3):
    # Set up environment
    def get_env():
        return NetworkInfluenceEnv(config)
    
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(1)])

    def stop_fn(mean_rewards):
        return mean_rewards >= 500000  # Define a suitable threshold for your problem

    def train_fn(epoch, env_step):
        epsilon = max(0.1, 1 - env_step / 100_000)  # Linear decay
        policy.epsilon = epsilon

    def test_fn(epoch, env_step):
        pass

    # Instantiate the model and policy
    state_dim = config['num_nodes']
    action_dim = config['num_nodes']

    model = QNet(state_dim, action_dim)
    target_model = QNet(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())  # Initialize target model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy = CustomDoubleDQNPolicy(
        model,
        target_model,
        optimizer,
        action_dim=action_dim,
        k=num_actions,
        gamma=0.99,
        tau=0.005  # Soft update parameter
    )

    # Set up collectors
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(total_size=20000 * train_envs.env_num, buffer_num=train_envs.env_num)
    )
    test_collector = Collector(policy, test_envs)

    # Start training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=num_epochs,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        update_per_step=0.1,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        logger=logger
    ).run()

    return model, policy

def select_action_double_dqn(graph, model, num_actions):
    """
    Hill-climbing action selection using a Double DQN model.
    
    Args:
        graph: The graph representing the environment.
        model: The trained DQN model.
        num_actions: The number of actions to select (k).

    Returns:
        A list of node objects representing the selected actions.
    """
    num_nodes = len(graph.nodes())
    state = np.array([int(graph.nodes[i]['obj'].isActive()) for i in graph.nodes()], dtype=np.float32)
    device = torch.device("cpu")  # Ensure tensor is on CPU
    selected_node_indices = []

    for _ in range(num_actions):
        # Prepare the state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Identify available nodes
        available_indices = [i for i in range(num_nodes) if state[i] == 0]

        if not available_indices:
            break  # No more nodes to select

        # Generate possible actions
        actions_list = []
        for idx in available_indices:
            action = torch.zeros(num_nodes, device=device)
            action[idx] = 1
            actions_list.append(action)
        actions_tensor = torch.stack(actions_list)
        states_tensor = state_tensor.repeat(len(available_indices), 1)

        # Compute Q-values
        with torch.no_grad():
            q_values = model(states_tensor, actions_tensor).squeeze()

        # Select the action with the highest Q-value
        selected_idx = torch.argmax(q_values).item()
        selected_node = available_indices[selected_idx]
        selected_node_indices.append(selected_node)

        # Update the state to reflect the selected action
        state[selected_node] = 1

    seeded_nodes = [graph.nodes[node_index]['obj'] for node_index in selected_node_indices]

    return seeded_nodes
