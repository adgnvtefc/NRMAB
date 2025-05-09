import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from collections import deque
import random
from algorithms.graph_env import GraphEnv, convert_nx_to_pyg

class GraphQ:
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            gamma=0.99,
            lr=1e-3,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.1,
            target_update_freq=50,
            replay_buffer_size=20000,
            batch_size=64,
            update_per_step=1.0
    ):
        # Define device: use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.batch_size = batch_size
        self.update_per_step = update_per_step

        # Initialize networks and move them to the proper device
        self.gnn = DeeperGCN(input_dim, hidden_dim, output_dim, num_hidden_layers=3).to(self.device)
        self.target_gnn = DeeperGCN(input_dim, hidden_dim, output_dim, num_hidden_layers=3).to(self.device)
        self.update_target_network()

        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.target_update_freq = target_update_freq
        self.steps_done = 0

    def update_target_network(self):
        self.target_gnn.load_state_dict(self.gnn.state_dict())

    def forward(self, data):
        # Create tensors on the correct device
        x = torch.tensor(data["x"], dtype=torch.float, device=self.device)
        edge_index = torch.tensor(data["edge_index"], dtype=torch.long, device=self.device)
        return self.gnn(x, edge_index)

    def forward_target(self, data):
        x = torch.tensor(data["x"], dtype=torch.float, device=self.device)
        edge_index = torch.tensor(data["edge_index"], dtype=torch.long, device=self.device)
        return self.target_gnn(x, edge_index)

    def select_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return torch.argmax(q_values).item()

    def store_experience(self, data, action, reward, next_data, done):
        self.replay_buffer.append((data, action, reward, next_data, done))

    def sample_batch(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def compute_loss_on_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        all_q_values, all_target_q_values = [], []
        for i in range(self.batch_size):
            q_values = self.forward(states[i])
            q_value_action = q_values[actions[i]]

            with torch.no_grad():
                next_q_values = self.forward_target(next_states[i])
                max_next_q = torch.max(next_q_values).item()
                target_q = rewards[i] + (1 - dones[i]) * self.gamma * max_next_q

            all_q_values.append(q_value_action.view(1))
            all_target_q_values.append(torch.tensor([target_q], dtype=torch.float32, device=self.device))

        all_q_values = torch.cat(all_q_values, dim=0)
        all_target_q_values = torch.cat(all_target_q_values, dim=0)
        return self.criterion(all_q_values, all_target_q_values)

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        loss = self.compute_loss_on_batch(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes=300):
        for episode in range(num_episodes):
            data, _ = env.reset()
            done, total_reward = False, 0

            while not done:
                q_values = self.forward(data).detach()
                action = self.select_action(q_values)

                next_data, reward, done, _, _ = env.step(action)
                total_reward += reward

                self.store_experience(data, action, reward, next_data, done)
                data = next_data

                for _ in range(int(self.update_per_step)):
                    self.optimize()

                self.steps_done += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

    def predict(self, graph, k=1):
        # Convert a NetworkX graph to a PyG Data object
        data = convert_nx_to_pyg(graph)
        # Forward pass and then move the result to CPU for numpy conversion
        q_values = self.forward(data).detach().cpu().numpy().squeeze()

        # Set q-value to -inf for nodes that are already active
        for node_idx in graph.nodes():
            if graph.nodes[node_idx]['obj'].isActive():
                q_values[node_idx] = -np.inf

        return np.argsort(q_values)[-k:][::-1].tolist()


class DeeperGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input -> hidden
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers: hidden -> hidden
        for _ in range(num_hidden_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer: hidden -> output
        self.out_conv = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return self.out_conv(x, edge_index).view(-1)
