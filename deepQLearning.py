import networkx as nx
from simpleNode import SimpleNode as Node
import itertools
import copy
from networkSim import NetworkSim as ns
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os  # Added to check for model file existence

from networkvis import NetworkVis as nv

# Neural network to approximate Q-values
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output a single Q-value


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Replay Buffer for storing experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Neural Network based Q-learning agent
class NeuralQLearner:
    def __init__(self, graph, num_actions=1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_capacity=10000, hidden_size=128, model_path='deep_q_model.pth'):
        self.graph = graph
        self.num_nodes = len(graph.nodes())
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.state_size = self.num_nodes
        self.action_size = self.num_nodes

        self.q_network = QNetwork(self.state_size, self.action_size, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

        self.model_path = model_path

    def get_state_representation(self, graph):
        return np.array([1 if graph.nodes[node]['obj'].isActive() else 0 for node in graph.nodes()])

    def get_action_representation(self, action_indices):
        action_vector = np.zeros(self.num_nodes)
        # Ensure action_indices is a flat array
        action_indices = np.array(action_indices).flatten()
        action_vector[action_indices] = 1
        return action_vector

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            # Random action
            action_indices = random.sample(range(self.num_nodes), self.num_actions)
        else:
            # Evaluate all possible actions (or sample a subset if too many)
            possible_actions = list(itertools.combinations(range(self.num_nodes), self.num_actions))
            max_q_value = float('-inf')
            best_action_indices = None
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            for action_indices in possible_actions:
                action_vector = self.get_action_representation(action_indices)
                action_tensor = torch.FloatTensor(action_vector).unsqueeze(0)
                q_value = self.q_network(state_tensor, action_tensor)
                if q_value.item() > max_q_value:
                    max_q_value = q_value.item()
                    best_action_indices = action_indices
            action_indices = best_action_indices
        return action_indices

    def train(self, num_episodes=1000, batch_size=32):
        for episode in range(num_episodes):
            state = self.get_state_representation(self.graph)
            done = False
            episode_loss = 0
            step = 0

            while not done:
                action_indices = self.epsilon_greedy_action(state)
                action_vector = self.get_action_representation(action_indices)

                # Apply action and get next state, reward
                next_graph = copy.deepcopy(self.graph)

                exempt_nodes = [next_graph.nodes[node_index]['obj'] for node_index in action_indices]
                ns.passive_state_transition_without_neighbors(next_graph, exempt_nodes=exempt_nodes)
                ns.active_state_transition_graph_indices(next_graph, action_indices)
                ns.independent_cascade_allNodes(next_graph, 0.1)
                ns.rearm_nodes(next_graph)

                next_state = self.get_state_representation(next_graph)
                reward = ns.reward_function(next_graph, set(action_indices)) ** 2  # square it
                done = self.termination_condition(next_graph)

                # Add experience to replay buffer
                self.buffer.add((state, action_indices, reward, next_state, done))

                # Sample from replay buffer and train
                if self.buffer.size() > batch_size:
                    experiences = self.buffer.sample(batch_size)
                    loss = self.train_q_network(experiences)
                    episode_loss += loss.item()
                    step += 1

                state = next_state

            print(f"Episode {episode+1}/{num_episodes}, Loss: {episode_loss/step if step > 0 else 0}")

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def train_q_network(self, experiences):
        states, action_indices_list, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Reconstruct action vectors
        actions = []
        for action_indices in action_indices_list:
            action_vector = self.get_action_representation(action_indices)
            actions.append(action_vector)
        actions = torch.FloatTensor(np.array(actions))

        # Current Q-values
        q_values = self.q_network(states, actions)

        # Compute target Q-values
        with torch.no_grad():
            # Sample possible next actions (or use policy to select next action)
            next_actions = []
            for next_state in next_states:
                possible_actions = list(itertools.combinations(range(self.num_nodes), self.num_actions))
                max_q_value = float('-inf')
                best_action_vector = None
                for action_indices in possible_actions:
                    action_vector = self.get_action_representation(action_indices)
                    action_tensor = torch.FloatTensor(action_vector).unsqueeze(0)
                    q_value = self.q_network(next_state.unsqueeze(0), action_tensor)
                    if q_value.item() > max_q_value:
                        max_q_value = q_value.item()
                        best_action_vector = action_vector
                if best_action_vector is not None:
                    next_actions.append(best_action_vector)
                else:
                    # Handle the case where no action was selected
                    next_actions.append(np.zeros(self.num_nodes))
            next_actions = torch.FloatTensor(np.array(next_actions))
            max_next_q_values = self.q_network(next_states, next_actions)

            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def termination_condition(self, graph):
        # Terminate if 80% or more of the nodes are active
        active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())
        return active_nodes >= 0.8 * self.num_nodes
    def save_model(self):
        torch.save(self.q_network.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.q_network.load_state_dict(torch.load(self.model_path))
            self.q_network.eval()  # Set the model to evaluation mode
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No saved model found at {self.model_path}")
            return False

# Example usage -- this trains and saves a model for the graph with the specified specifications
#doesn't exactly work all that well in the first place
#but is extra bad because each saved NN is tailored to a graph and we don't save that graph
if __name__ == "__main__":
    # Initialize the graph
    G = ns.init_random_graph(10, 20, 0.1, 0.3, 0.95, 0.05)


    # For Deep Q-Learning
    deep_q_agent = NeuralQLearner(G, num_actions=2, model_path='deep_q_model.pth')
    # Attempt to load the model
    model_loaded = deep_q_agent.load_model()
    if not model_loaded:
        print("training model")
        # If model is not loaded, train it and save the model
        deep_q_agent.train(num_episodes=100)
        deep_q_agent.save_model()
    else:
        print("Using the loaded Deep Q-Learning model.")


    # Initialize graph position for visualization
    pos = nx.spring_layout(G)
    node_obj_to_id = {data['obj']: node_id for node_id, data in G.nodes(data=True)}
    timestep = 0
    while True:
        plt.clf()  # Clear the previous plot
        best_action = None

        # Use the trained Q-network to decide the best action
        with torch.no_grad():
            state = deep_q_agent.get_state_representation(G)
            possible_actions = list(itertools.combinations(range(deep_q_agent.num_nodes), deep_q_agent.num_actions))
            max_q_value = float('-inf')
            best_action_indices = None
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            for action_indices in possible_actions:
                action_vector = deep_q_agent.get_action_representation(action_indices)
                action_tensor = torch.FloatTensor(action_vector).unsqueeze(0)
                q_value = deep_q_agent.q_network(state_tensor, action_tensor)
                if q_value.item() > max_q_value:
                    max_q_value = q_value.item()
                    best_action_indices = action_indices
            best_action = best_action_indices

        print("Best action: " + str(best_action))
        nv.do_things(G, pos, best_action, node_obj_to_id, timestep)
        timestep += 1
