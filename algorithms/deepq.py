import gymnasium as gym
from algorithms.deepq_env import NetworkInfluenceEnv
import random
import torch
import torch.nn as nn
import numpy as np
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import os
from dataclasses import dataclass
from typing import Dict
import time
from typing import Dict, Iterator, Tuple


log_path = os.path.join('logs', 'dqn')
writer = SummaryWriter(log_path)
writer.add_text("Experiment Info", "DQN training with custom environment")
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

    def forward(self, state, action, state_shape=None, action_shape=None):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        q_value = self.fc5(x)
        return q_value

# Define the custom policy
class CustomQPolicy(BasePolicy):
    def __init__(self, model, optim, action_dim, k=5, gamma=0.95, epsilon=1.0):
        super().__init__(action_space=gym.spaces.MultiBinary(action_dim))
        self.model = model
        self.optim = optim
        self.k = k
        self.action_dim = action_dim
        self._gamma = gamma
        self.epsilon = epsilon  # Epsilon for epsilon-greedy exploration

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
                available_indices = (state_i == 0).nonzero(as_tuple=False).squeeze().tolist()
                if not available_indices:
                    break  # No more available nodes to select
                if type(available_indices) == int:
                    available_indices = [available_indices]

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

        return TrainStepResult(loss=loss.item())


    def compute_loss(self, states, actions, rewards, next_states, dones):
        gamma = self._gamma

        # Flatten actions for input
        actions = actions.view(-1, self.action_dim)
        states = states.view(-1, self.action_dim)

        q_values = self.model(states, actions).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(next_states.shape[0]):
                next_state = next_states[i]
                available_indices = (next_state == 0).nonzero(as_tuple=False).squeeze().tolist()
                if not available_indices:
                    max_q_value = 0.0
                else:
                    if type(available_indices) == int:
                        available_indices = [available_indices]

                    actions_list = []
                    for idx in available_indices:
                        action = torch.zeros(self.action_dim, device=next_state.device)
                        action[idx] = 1
                        actions_list.append(action)
                    actions_tensor = torch.stack(actions_list)
                    states_tensor = next_state.unsqueeze(0).repeat(len(available_indices), 1)
                    q_vals = self.model(states_tensor, actions_tensor).squeeze()
                    max_q_value = q_vals.max().item()
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

    def keys(self) -> Tuple[str, ...]:
        return ("loss",)

    def __getitem__(self, key: str) -> float:
        if key == "loss":
            return self.loss
        raise KeyError(f"TrainStepResult does not contain key: {key}")

    def __setitem__(self, key: str, value: float) -> None:
        if key == "loss":
            self.loss = value
        else:
            raise KeyError(f"TrainStepResult does not contain key: {key}")
    
    def items(self) -> Iterator[Tuple[str, float]]:
        for k in self.keys():
            yield k, self[k]
    


def train_dqn_agent(config, num_actions, num_epochs=3, step_per_epoch=1000):
    start_time = time.perf_counter()

    # Set up environment
    def get_env():
        return NetworkInfluenceEnv(config)
    
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(1)])

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        epsilon = max(0.1, 1 - env_step / 50000)  # Linear decay
        policy.epsilon = epsilon
    def test_fn(epoch, env_step):
        pass

    # Instantiate the model and policy
    state_dim = config['num_nodes']
    action_dim = config['num_nodes']

    model = QNet(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    policy = CustomQPolicy(model, optimizer, action_dim=action_dim, k=num_actions, gamma=0.99)

    # Set up collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=20000 * train_envs.env_num, buffer_num=train_envs.env_num))
    test_collector = Collector(policy, test_envs)

    # Start training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=50,
        episode_per_test=0,
        batch_size=64,
        update_per_step=0.1,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        logger=logger
    ).run()

    end_time = time.perf_counter()
    
    time_taken = end_time - start_time
    train_dqn_agent.time = time_taken

    return model, policy

train_dqn_agent.time = 0
def get_train_dqn_agent_time():
    return train_dqn_agent.time

def select_action_dqn(graph, model, num_actions):
    """
    Hill-climbing action selection using a DQN model.
    
    Args:
        graph: The graph representing the environment.
        model: The trained DQN model.
        num_actions: The number of actions to select (k).

    Returns:
        A list of node objects representing the selected actions.
    """
    start_time = time.perf_counter()
    select_action_dqn.times_called += 1
    num_nodes = len(graph.nodes())
    state = np.array([int(graph.nodes[i]['obj'].isActive()) for i in graph.nodes()], dtype=np.float32)
    device = next(model.parameters()).device
    selected_node_indices = []

    for _ in range(num_actions):
        # Prepare the state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Generate all possible actions (one-hot encoded)
        actions = torch.eye(num_nodes, device=device)
        states = state_tensor.repeat(num_nodes, 1)
        actions_tensor = actions

        # Compute Q-values
        with torch.no_grad():
            q_values = model(states, actions_tensor).squeeze()

        # Mask already active nodes and previously selected actions
        active_nodes_indices = [i for i, node in enumerate(graph.nodes()) if graph.nodes[node]['obj'].isActive()]
        already_selected_indices = [node for node in selected_node_indices]
        mask_indices = active_nodes_indices + already_selected_indices

        q_values_np = q_values.cpu().numpy()
        q_values_np[mask_indices] = -np.inf  # Assign negative infinity to already active or selected nodes

        # Select the top action
        top_action_index = np.argmax(q_values_np)
        selected_node_indices.append(top_action_index)

        # Update the state to assume the selected action is now active
        state[top_action_index] = 1
    
    seeded_nodes = [graph.nodes[node_index]['obj'] for node_index in selected_node_indices]
    end_time = time.perf_counter()
    elapsed = end_time - start_time
        
    select_action_dqn.total_time += elapsed
    return seeded_nodes
select_action_dqn.total_time = 0.0
select_action_dqn.times_called = 0

@staticmethod
def get_dqn_total_time():
    return select_action_dqn.total_time
@staticmethod
def get_dqn_times_called():
    return select_action_dqn.times_called