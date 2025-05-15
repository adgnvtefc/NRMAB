import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns
import numpy as np
from algorithms.graph_env import GraphEnv
from torch_geometric.data import Batch, Data

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
#KILL EDGES???

#MODIFY TO TAKE IN STATE AND SET OF ACTIONS, AND PREDICT Q VALUE OF STATE-ACTION
#SEE PHOTO

class GNNQNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_nodes):
        """
        in_channels: size of node feature vector (6 in our case)
        hidden_channels: hidden layer size
        num_nodes: number of nodes (discrete actions)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc_global = nn.Linear(hidden_channels, hidden_channels)
        self.fc_q = nn.Linear(hidden_channels, num_nodes)

    def forward(self, obs, state=None, info={}):
        x = torch.as_tensor(obs.x, dtype=torch.float)
        edge_index = torch.as_tensor(obs.edge_index, dtype=torch.long)

        # Remove extra batch dimension if present
        if x.dim() == 3:  # Shape [batch_size, num_nodes, features]
            x = x.squeeze(0)  # Now shape [num_nodes, features]
        if edge_index.dim() == 3:  # Shape [batch_size, 2, num_edges]
            edge_index = edge_index.squeeze(0)  # Now shape [2, num_edges]
        batch = None

        # If there's a 'batch' attribute, it indicates multiple graphs.
        if hasattr(obs, "batch"):
            batch = obs.batch
        else:
            # Single graph -> create a batch tensor with all zeros
            batch = torch.zeros(x.shape[0], dtype=torch.long)

        # Debug prints
        print(x.shape)  # Should be [num_nodes, features], e.g., [202, 6]
        print(edge_index.shape)  # Should be [2, num_edges], e.g., [2, 1384]
        print(batch.shape)  # Should be [num_nodes], e.g., [202]

        # Pass through GCN layers
        x = self.conv1(x, edge_index)  # [num_nodes, hidden_channels]
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        global_embedding = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        global_embedding = F.relu(self.fc_global(global_embedding))
        q = self.fc_q(global_embedding)  # [num_graphs, num_nodes]

        # If there's only one graph in the batch, remove the batch dimension
        if q.size(0) == 1:
            q = q.unsqueeze(0)
        return q, state

def pyg_collate_fn(batch):
    data_list = []
    for obs in batch:
        # Ensure x has shape [num_nodes, 6]
        x = torch.tensor(obs["x"], dtype=torch.float)
        if x.dim() == 3:  # Remove extra batch dimension if present
            x = x.squeeze(0)
        # Ensure edge_index has shape [2, num_edges]
        edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
        if edge_index.dim() == 3:  # Remove extra batch dimension if present
            edge_index = edge_index.squeeze(0)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    
    # Batch the Data objects
    batch = Batch.from_data_list(data_list)
    return batch


def train_gnn_agent(config, num_epochs=3):
    def get_env():
        return GraphEnv(config)

    # Create vectorized train/test envs
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(1)])

    # Instantiate model & policy
    in_channels = 6
    hidden_channels = 16
    num_nodes = config['num_nodes']
    model = GNNQNetwork(in_channels, hidden_channels, num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # We pass a sample_env to obtain its action_space
    sample_env = get_env()
    policy = DQNPolicy(
        model=model,
        optim=optimizer,
        discount_factor=config['gamma'],
        estimation_step=1,
        target_update_freq=100,
        action_space=sample_env.action_space
    )

    # Create replay buffer & collectors
    buffer = VectorReplayBuffer(total_size=20000, buffer_num=train_envs.env_num)
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # Assign the custom collate function so each batch of observations
    # is turned into a PyG Batch object
    train_collector.collate_fn = pyg_collate_fn
    test_collector.collate_fn = pyg_collate_fn

    def train_fn(epoch, env_step):
        # Example: linear epsilon decay
        epsilon = max(0.1, 1 - env_step / 10000)
        policy._epsilon = epsilon

    # Run the OffpolicyTrainer
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
        test_fn=lambda epoch, env_step: None,
        stop_fn=lambda mean_rewards: False,
        logger=None
    ).run()

    return model, policy
