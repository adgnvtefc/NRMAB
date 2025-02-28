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
        """
        Improved GraphQ initialization to align with DQN approach.
          - bigger hidden_dim
          - gamma=0.99 to account for longer horizons
          - higher lr=1e-3 for faster learning
          - slower epsilon_decay=0.995
          - target_update_freq=50 steps (instead of every episode)
          - larger replay buffer
          - batch_size=64
          - update_per_step=1.0 meaning 1 gradient update per environment step
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.batch_size = batch_size
        self.update_per_step = update_per_step

        # GNN with multiple hidden layers
        self.gnn = DeeperGCN(input_dim, hidden_dim, output_dim, num_hidden_layers=3)
        self.target_gnn = DeeperGCN(input_dim, hidden_dim, output_dim, num_hidden_layers=3)
        self.update_target_network()  # initialize target

        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # We'll update the target net every N steps
        self.target_update_freq = target_update_freq
        self.steps_done = 0  # track how many steps have been taken overall

    def update_target_network(self):
        """
        Copy main network weights into the target network.
        """
        self.target_gnn.load_state_dict(self.gnn.state_dict())

    def forward(self, data):
        """
        Forward pass through the main GNN.
        Args:
            data (torch_geometric.data.Data): node features (x) and edges (edge_index).
        """
        x = torch.tensor(data["x"], dtype=torch.float)
        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        return self.gnn(x, edge_index)

    def forward_target(self, data):
        """
        Forward pass through the target GNN.
        """
        x = torch.tensor(data["x"], dtype=torch.float)
        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        return self.target_gnn(x, edge_index)

    def select_action(self, q_values):
        """
        Epsilon-greedy selection of a single node.
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return torch.argmax(q_values).item()

    def store_experience(self, data, action, reward, next_data, done):
        self.replay_buffer.append((data, action, reward, next_data, done))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        return batch

    def compute_loss_on_batch(self, batch):
        """
        Compute MSE loss on a batch of transitions.
        """
        # separate the batch into arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to Tensors
        # We'll handle them one by one in a vectorized manner
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        all_q_values = []
        all_target_q_values = []
        for i in range(self.batch_size):
            # process i-th transition
            data_i = states[i]
            action_i = actions[i]
            reward_i = rewards[i]
            next_data_i = next_states[i]
            done_i = dones[i]

            q_values = self.forward(data_i)
            q_value_action = q_values[action_i]  # Q(s,a)

            with torch.no_grad():
                next_q_values = self.forward_target(next_data_i)
                max_next_q = torch.max(next_q_values).item()
                target_q = reward_i + (1 - done_i) * self.gamma * max_next_q

            all_q_values.append(q_value_action.view(1))
            all_target_q_values.append(torch.tensor([target_q], dtype=torch.float32))

        all_q_values = torch.cat(all_q_values, dim=0)
        all_target_q_values = torch.cat(all_target_q_values, dim=0)
        loss = self.criterion(all_q_values, all_target_q_values)
        return loss

    def optimize(self):
        """
        One optimizer step on a sampled batch from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data

        batch = self.sample_batch()
        loss = self.compute_loss_on_batch(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes=300):
        """
        Train the GNN agent in an environment.
        We'll do multiple updates per step, as in DQN,
        where after each environment step, we do 'update_per_step' gradient updates.
        """
        for episode in range(num_episodes):
            data, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Get Q-values
                q_values = self.forward(data).detach()
                # Pick an action
                action = self.select_action(q_values)

                # Step environment
                next_data, reward, done, _, _ = env.step(action)
                total_reward += reward

                self.store_experience(data, action, reward, next_data, done)
                data = next_data

                # Training steps
                for _ in range(int(self.update_per_step)):
                    self.optimize()

                # Update target net occasionally
                self.steps_done += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()

            # End of episode, epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            print(f"Episode {episode+1}/{num_episodes}, Reward={total_reward:.2f}, Epsilon={self.epsilon:.3f}")

    def predict(self, graph, k=1):
        """
        Predict top-k nodes to activate in a sequential manner,
        recomputing Q-values after each node is 'activated'.
        This helps avoid picking multiple redundant nodes.
        """
        import copy
        from algorithms.graph_env import convert_nx_to_pyg

        selected_nodes = []
        temp_graph = copy.deepcopy(graph)

        for _ in range(k):
            data = convert_nx_to_pyg(temp_graph, last_action_indices=None)
            with torch.no_grad():
                q_values = self.forward(data)

            # mask out already-active or already-selected nodes
            q_array = q_values.detach().cpu().numpy().squeeze()
            for node_idx in temp_graph.nodes():
                if temp_graph.nodes[node_idx]["obj"].isActive() or (node_idx in selected_nodes):
                    q_array[node_idx] = -float('inf')

            best_node = int(np.argmax(q_array))
            selected_nodes.append(best_node)
            # 'Activate' in temp graph
            temp_graph.nodes[best_node]["obj"].activate()

        return selected_nodes


class DeeperGCN(torch.nn.Module):
    """
    A deeper GCN architecture for better representational capacity.
    We'll use multiple GCNConv layers, each followed by BatchNorm and ReLU.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=3):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # first layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # hidden layers
        for _ in range(num_hidden_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # output layer
        self.out_conv = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for i in range(self.num_hidden_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        # final output layer
        x = self.out_conv(x, edge_index)
        return x.view(-1)  # shape [num_nodes]
