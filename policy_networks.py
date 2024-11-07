# policy_networks.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, num_nodes):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_nodes, 128)
        self.fc2 = nn.Linear(128, num_nodes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, num_nodes):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(num_nodes, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PolicyNetworkAgent:
    def __init__(self, num_nodes, num_actions, lr=1e-3):
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.policy_net = PolicyNetwork(num_nodes)
        self.value_net = ValueNetwork(num_nodes)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr
        )

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy_net(state_tensor)
        logits_np = logits.detach().numpy().flatten()
        topk_indices = np.argpartition(-logits_np, self.num_actions)[:self.num_actions]
        return topk_indices, logits_np

    def train(self, env, num_episodes=500, gamma=0.9):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropies = []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits = self.policy_net(state_tensor)
                value = self.value_net(state_tensor)

                # Select top k nodes
                logits_np = logits.detach().numpy().flatten()
                topk_indices = np.argpartition(-logits_np, self.num_actions)[:self.num_actions]

                # Create action vector
                action = np.zeros(self.num_nodes, dtype=np.float32)
                action[topk_indices] = 1.0

                # Calculate log probabilities and entropy
                probs = F.softmax(logits, dim=-1)
                log_probs_all = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs_all[0, topk_indices]
                log_prob = selected_log_probs.sum()
                entropy = -(probs * log_probs_all).sum()

                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)

                # Step environment
                next_state, reward, done, _, info = env.step(logits.detach().numpy())
                rewards.append(torch.tensor([reward], dtype=torch.float32))
                masks.append(torch.tensor([1 - done], dtype=torch.float32))

                state = next_state

            # Compute returns and advantages
            returns = []
            advantages = []
            R = torch.zeros(1, 1)
            for step in reversed(range(len(rewards))):
                R = rewards[step] + gamma * R * masks[step]
                advantage = R - values[step]
                returns.insert(0, R)
                advantages.insert(0, advantage)

            # Normalize advantages
            advantages = torch.cat(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Convert lists to tensors
            log_probs = torch.stack(log_probs)
            returns = torch.stack(returns).detach()
            values = torch.stack(values)
            entropies = torch.stack(entropies)

            # Compute losses
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropies.mean()
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Optionally, print progress
            total_reward = sum([r.item() for r in rewards])
            print(f"Episode {episode}, Total Reward: {total_reward}")

    def save_model(self, model_path_policy, model_path_value):
        torch.save(self.policy_net.state_dict(), model_path_policy)
        torch.save(self.value_net.state_dict(), model_path_value)

    def load_model(self, model_path_policy, model_path_value):
        self.policy_net.load_state_dict(torch.load(model_path_policy))
        self.value_net.load_state_dict(torch.load(model_path_value))
