import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN
import numpy as np
from collections import deque
import random
from algorithms.graph_env import GraphEnv, convert_nx_to_pyg

class GraphQ:
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.8, lr=0.00005, epsilon=1.0, epsilon_decay=0.99999, epsilon_min=0.2):
        """
        Initialize the GraphQ model.

        Args:
            input_dim (int): Dimension of node features.
            hidden_dim (int): Dimension of hidden layers in the GNN.
            output_dim (int): Dimension of the output (Q-values).
            gamma (float): Discount factor for Q-learning.
            lr (float): Learning rate for the optimizer.
            epsilon (float): Initial exploration rate for epsilon-greedy strategy.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_min (float): Minimum value for epsilon.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr

        # Initialize GNN model
        self.gnn = GCN(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=10000)

    def forward(self, data):
        """
        Forward pass through the GNN.

        Args:
            data (torch_geometric.data.Data): PyG Data object containing node features and edge indices.

        Returns:
            torch.Tensor: Predicted Q-values for all nodes.
        """
        return self.gnn(torch.tensor(data["x"]), torch.tensor(data["edge_index"]))

    def select_action(self, q_values, epsilon=None):
        """
        Select an action using an epsilon-greedy strategy.

        Args:
            q_values (torch.Tensor): Predicted Q-values for all nodes.
            epsilon (float): Exploration rate. If None, use self.epsilon.

        Returns:
            int: Index of the selected node.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Explore: select a random node
            return random.randint(0, len(q_values) - 1)
        else:
            # Exploit: select the node with the highest Q-value
            return torch.argmax(q_values).item()

    def train_step(self, data, action, reward, next_data, done):
        """
        Perform a single training step.

        Args:
            data (torch_geometric.data.Data): Current graph state.
            action (int): Index of the selected node.
            reward (float): Reward received for the action.
            next_data (torch_geometric.data.Data): Next graph state.
            done (bool): Whether the episode is terminated.
        """
        # Predict Q-values for the current state
        q_values = self.forward(data)

        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.forward(next_data)
            target_q_value = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        # Compute loss and update the model
        loss = self.criterion(q_values[action], target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, env, num_episodes, batch_size=32):
        """
        Train the model using the provided environment.

        Args:
            env (gym.Env): The graph environment.
            num_episodes (int): Number of episodes to train.
            batch_size (int): Batch size for experience replay.
        """
        for episode in range(num_episodes):
            data, _ = env.reset()
            done = False
            total_reward = 0

            while not done:

                # Predict Q-values for all nodes
                q_values = self.forward(data)

                # Select action using epsilon-greedy strategy
                action = self.select_action(q_values)

                # Take action in the environment
                next_data, reward, done, _, _ = env.step(action)
                total_reward += reward

                # Store experience in replay buffer
                self.replay_buffer.append((data, action, reward, next_data, done))

                # Sample a batch from the replay buffer
                if len(self.replay_buffer) >= batch_size:
                    batch = random.sample(self.replay_buffer, batch_size)
                    for b in batch:
                        self.train_step(*b)

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

    def predict(self, graph, k=1):
        """
        Predict the top k nodes to activate.

        Args:
            graph (networkx.Graph): Input graph.
            k (int): Number of nodes to select.

        Returns:
            list: Indices of the top k nodes.
        """
        # Convert graph to PyG Data
        data = convert_nx_to_pyg(graph)

        # Predict Q-values for all nodes
        with torch.no_grad():
            q_values = self.forward(data)

        # Select top k nodes
        top_k_values, top_k_indices = torch.topk(q_values.squeeze(), k)
        return top_k_indices.tolist()


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for Q-value prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2):
        super().__init__()
        
        # 1) First GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.dropout1 = nn.Dropout(p=0.2)
        
        # 2) Stack of hidden GCN layers
        self.hidden_layers = nn.ModuleList()
        self.bn_hidden = nn.ModuleList()
        # self.dropouts_hidden = nn.ModuleList()
        
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_hidden.append(nn.BatchNorm1d(hidden_dim))
            # self.dropouts_hidden.append(nn.Dropout(p=0.2))
        
        # 3) Final GCN layer to map to output dimension
        self.conv_out = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        # --- First layer ---
        x = self.conv1(x, edge_index)             # shape: [num_nodes, hidden_dim]
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        
        # --- Hidden layers ---
        for conv, bn in zip(self.hidden_layers, self.bn_hidden):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # x = drop(x)
        
        # --- Final layer for Q-values ---
        q_values = self.conv_out(x, edge_index)   # shape: [num_nodes, output_dim]
        
        return q_values