# main.py

import gymnasium as gym
import tianshou as ts
from deep_q_env import NetworkInfluenceEnv
from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
import networkx as nx

import torch
import torch.nn as nn
import numpy as np
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net

# Define constants and configuration
PASSIVE_ACTIVATION_CHANCE = 0.1
PASSIVE_DEACTIVATION_CHANCE = 0.2
ACTIVE_ACTIVATION_CHANCE = 0.95
ACTIVE_DEACTIVATION_CHANCE = 0.05
CASCADE_PROB = 0.05

num_nodes = 50
num_actions = 3  # Number of actions to select per timestep

G = ns.init_random_graph(
    num_nodes,
    num_nodes * 1.5,
    PASSIVE_ACTIVATION_CHANCE,
    PASSIVE_DEACTIVATION_CHANCE,
    ACTIVE_ACTIVATION_CHANCE,
    ACTIVE_DEACTIVATION_CHANCE
)

config = {
    "graph": G,
    "num_nodes": num_nodes,
    "cascade_prob": CASCADE_PROB,
    "stop_percent": 0.8
}

# Environment setup
def get_env():
    return NetworkInfluenceEnv(config)

train_envs = DummyVectorEnv([get_env for _ in range(10)])
test_envs = DummyVectorEnv([get_env for _ in range(1)])

# Define the neural network
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output is a scalar Q-value

    def forward(self, state, action, state_shape=None, action_shape=None):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Define the custom policy
class CustomQPolicy(BasePolicy):
    def __init__(self, model, optim, action_dim, k=5, gamma=0.99):
        super().__init__(action_space=gym.spaces.MultiBinary(action_dim))

        self.model = model
        self.optim = optim
        self.k = k
        self.action_dim = action_dim
        self._gamma = gamma

    def forward(self, batch, state=None):
        obs = batch.obs
        batch_size = obs.shape[0]
        device = obs.device

        # Generate all possible actions (one-hot encoded)
        actions = torch.eye(self.action_dim, device=device)
        actions = actions.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, self.action_dim)
        states = obs.unsqueeze(1).repeat(1, self.action_dim, 1).view(-1, obs.shape[-1])

        # Compute Q-values for all state-action pairs
        q_values = self.model(states, actions).view(batch_size, self.action_dim)

        # Select top-k actions with highest Q-values
        topk_q_values, topk_actions = q_values.topk(self.k, dim=1)

        # Create action masks
        act = torch.zeros(batch_size, self.action_dim, device=device)
        for i in range(batch_size):
            act[i, topk_actions[i]] = 1

        return Batch(act=act), state

    def learn(self, batch, **kwargs):
        self.optim.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    def compute_loss(self, batch):
        gamma = self._gamma

        states = batch.obs
        actions = batch.act
        rewards = batch.rew
        next_states = batch.obs_next
        dones = batch.done

        # Flatten actions for input
        actions = actions.view(-1, self.action_dim)
        states = states.view(-1, states.shape[-1])

        q_values = self.model(states, actions).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            # Generate all possible actions for next states
            next_actions = torch.eye(self.action_dim, device=states.device)
            next_actions = next_actions.unsqueeze(0).repeat(len(next_states), 1, 1).view(-1, self.action_dim)
            next_states_rep = next_states.unsqueeze(1).repeat(1, self.action_dim, 1).view(-1, states.shape[-1])

            q_values_next = self.model(next_states_rep, next_actions).view(len(next_states), self.action_dim)
            max_q_values_next, _ = q_values_next.max(dim=1)

            target_q_values = rewards + gamma * (1 - dones) * max_q_values_next

        loss = nn.functional.mse_loss(q_values, target_q_values)
        return loss


def train_dqn_agent(config, num_actions, num_epochs=50):
    # Set up environment
    def get_env():
        return NetworkInfluenceEnv(config)
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(1)])

    # Instantiate the model and policy
    state_dim = config['num_nodes']
    action_dim = config['num_nodes']

    model = QNet(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy = CustomQPolicy(model, optimizer, action_dim=action_dim, k=num_actions, gamma=0.99)

    # Set up collectors
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=20000 * train_envs.env_num, buffer_num=train_envs.env_num))
    test_collector = Collector(policy, test_envs)

    # Training function
    def stop_fn(mean_rewards):
        return mean_rewards >= 50

    # Start training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=10,
        episode_per_test=10,
        batch_size=64,
        update_per_step=0.1,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        logger=None
    )

    return model, policy


# Instantiate the model and policy
state_dim = num_nodes
action_dim = num_nodes  # Number of possible actions (nodes)

model = QNet(state_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
policy = CustomQPolicy(model, optimizer, action_dim=action_dim, k=num_actions, gamma=0.99)

# Set up collectors
train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=20000 * train_envs.env_num, buffer_num=train_envs.env_num))
test_collector = Collector(policy, test_envs)

# Training function
def stop_fn(mean_rewards):
    return mean_rewards >= 50  # Define a suitable threshold for your problem

def train_fn(epoch, env_step):
    pass  # You can implement epsilon decay or other training strategies here

def test_fn(epoch, env_step):
    pass

# Start training
result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=50,
    step_per_epoch=1000,
    step_per_collect=10,
    episode_per_test=10,
    batch_size=64,
    update_per_step=0.1,
    train_fn=train_fn,
    test_fn=test_fn,
    stop_fn=stop_fn,
    logger=None
)

# Save the trained model
#torch.save(model.state_dict(), 'dqn_model.pth')

# ------------------ Simulation with Trained Model ------------------

# Load the trained model (optional, if continuing from above)
#model.load_state_dict(torch.load('dqn_model.pth'))
model.eval()

# Initialize environment for simulation


# env = NetworkInfluenceEnv(config)
# state, _ = env.reset()
# done = False

# # Visualization setup
# pos = nx.spring_layout(G)
# nv.render(env.graph, pos)

# while not done:
#     # Convert state to tensor
#     state_tensor = torch.FloatTensor(state).unsqueeze(0)

#     # Generate all possible actions
#     actions = torch.eye(action_dim)
#     states = state_tensor.repeat(action_dim, 1)
#     actions_tensor = actions

#     # Compute Q-values
#     with torch.no_grad():
#         q_values = model(states, actions_tensor).squeeze()

#     # Select top-k actions
#     topk_q_values, topk_actions = torch.topk(q_values, num_actions)
#     selected_actions = torch.zeros(action_dim)
#     selected_actions[topk_actions] = 1

#     # Apply selected actions in the environment
#     action = selected_actions.numpy().astype(int)
#     state, reward, done, _, info = env.step(action)

#     # Render the updated graph
#     nv.render(env.graph, pos)

#     print(f"Selected actions: {topk_actions.numpy()}, Reward: {reward}")
#     input()

# print("Simulation completed.")
