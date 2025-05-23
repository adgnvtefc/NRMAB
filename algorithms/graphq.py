import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import random
from algorithms.graphq_env import GraphEnv, convert_nx_to_pyg
import matplotlib.pyplot as plt
import time

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, experience, priority=None):
        max_prio = max(self.priorities, default=1.0) if priority is None else priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

class GraphQ(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            gamma=0.99,
            lr=0.0001,
            epsilon=1.0,
            epsilon_decay=0.97,
            epsilon_min=0.1,
            target_update_freq=20,
            replay_buffer_size=20000,
            batch_size=64,
            update_per_step=1.0,
            device=None,
            per_alpha=0.6,
            per_beta_start=0.4,
            per_beta_increment=1e-3,
            double_dqn=True
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.batch_size = batch_size
        self.update_per_step = update_per_step
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        # Double DQN setting
        self.double_dqn = double_dqn

        self.train_time = 0

        # Networks
        self.gnn = DeeperGCN(input_dim, hidden_dim, output_dim, num_hidden_layers=3).to(self.device)
        self.target_gnn = DeeperGCN(input_dim, hidden_dim, output_dim, num_hidden_layers=3).to(self.device)
        self.update_target_network()

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='none')

        # Prioritized replay
        self.per_buffer = PrioritizedReplayBuffer(replay_buffer_size, alpha=per_alpha)
        self.per_beta = per_beta_start
        self.per_beta_increment = per_beta_increment

        # Track rewards
        self.episode_rewards = []

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_gnn.load_state_dict(self.gnn.state_dict())

    def forward(self, data):
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
        """Store a transition in the replay buffer with no initial priority (defaults to max)."""
        self.per_buffer.add((data, action, reward, next_data, done))

    def compute_td_error(self, sample):
        data, action, reward, next_data, done = sample
        q_vals = self.forward(data)
        q_val = q_vals[action]
        with torch.no_grad():
            # Compute target Q-value using Double DQN or standard DQN
            if self.double_dqn:
                next_q = self.forward(next_data)
                next_action = torch.argmax(next_q).unsqueeze(0)
                target_q_vals = self.forward_target(next_data)
                target_val = target_q_vals.gather(0, next_action).squeeze(0)
            else:
                target_q_vals = self.forward_target(next_data)
                target_val = torch.max(target_q_vals)
            target = reward + (1 - done) * self.gamma * target_val
        td_error = (target - q_val).abs().item()
        return td_error

    def optimize(self):
        if len(self.per_buffer.buffer) < self.batch_size:
            return
        # Sample a batch with priority
        batch, indices, is_weights = self.per_buffer.sample(self.batch_size, beta=self.per_beta)
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        states, actions, rewards, next_states, dones = zip(*batch)
        is_weights = is_weights.to(self.device)

        # Compute Q-values for current states
        q_vals = [self.forward(s) for s in states]
        # Compute next-state Q estimates
        if self.double_dqn:
            # Double DQN: use policy net for action selection, target net for evaluation
            with torch.no_grad():
                next_policy_q = [self.forward(s) for s in next_states]
                next_target_q = [self.forward_target(s) for s in next_states]
        else:
            with torch.no_grad():
                next_target_q = [self.forward_target(s) for s in next_states]
            next_policy_q = [None] * len(next_states)

        loss_vals = []
        td_errors = []
        for i in range(self.batch_size):
            q_val = q_vals[i][actions[i]]
            # Compute TD target for each sample
            if self.double_dqn:
                next_action_idx = torch.argmax(next_policy_q[i]).item()
                target_val = next_target_q[i][next_action_idx]
            else:
                target_val = torch.max(next_target_q[i])
            target = rewards[i] + (1 - dones[i]) * self.gamma * target_val
            td_error = (target - q_val).abs()
            td_errors.append(td_error.detach().cpu().item())
            loss_vals.append(td_error.pow(2))

        loss_tensor = torch.stack(loss_vals)
        loss = (is_weights * loss_tensor).mean()

        # Gradient step with clipping
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities in replay buffer
        new_prios = [e + 1e-6 for e in td_errors]
        self.per_buffer.update_priorities(indices, new_prios)

    def train(self, env, num_episodes=300, plot_rewards=False, save_path='training_rewards.png'):
        start_time = time.perf_counter()

        self.episode_rewards = []
        for episode in range(1, num_episodes + 1):
            data, _ = env.reset()
            done, total_reward = False, 0

            episode_step = 0
            maximum_steps = 50
            while (not done) and (episode_step < maximum_steps):
                q_vals = self.forward(data).detach()
                action = self.select_action(q_vals)

                next_data, reward, done, _, _ = env.step(action)
                total_reward += reward

                self.store_experience(data, action, reward, next_data, done)
                data = next_data

                for _ in range(int(self.update_per_step)):
                    self.optimize()

                self.steps_done += 1
                episode_step += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()

            # Decay exploration rate
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {self.epsilon:.4f}")
            self.episode_rewards.append(total_reward)

        end_time = time.perf_counter()
        
        time_taken = end_time - start_time
        self.train_time = time_taken

        # Optionally plot training rewards curve
        if plot_rewards:
            plt.figure()
            plt.plot(range(1, num_episodes + 1), self.episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Rewards')
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()


    def get_train_gnn_agent_time(self):
        return self.train_time

    def predict(self, graph, k=1):
        """Return the indices of the top-k nodes (arms) to activate based on current Q-values."""
        data = convert_nx_to_pyg(graph)
        q_vals = self.forward(data).detach().cpu().numpy().squeeze()
        # Exclude already active nodes from selection
        for idx in graph.nodes():
            if graph.nodes[idx]['obj'].isActive():
                q_vals[idx] = -np.inf
        top_k = np.argsort(q_vals)[-k:][::-1]
        return top_k.tolist()

class DeeperGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.out_conv = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        return self.out_conv(x, edge_index).view(-1)
