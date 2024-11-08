# main.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
from networkSim import NetworkSim as ns
from network_env import NetworkEnv
from policy_networks import PolicyNetwork, ValueNetwork
from networkvis import NetworkVis  # Import NetworkVis

def train_policy(graph, k=2, num_episodes=500, gamma=0.9, lr=1e-3):
    env = NetworkEnv(graph, k=k)
    num_nodes = env.num_nodes
    policy_net = PolicyNetwork(num_nodes)
    value_net = ValueNetwork(num_nodes)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
    
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
            logits = policy_net(state_tensor)
            value = value_net(state_tensor)
            
            # Select top k nodes
            logits_np = logits.detach().numpy().flatten()
            topk_indices = np.argpartition(-logits_np, k)[:k]
            
            # Create action vector
            action = np.zeros(num_nodes, dtype=np.float32)
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Optionally, print progress
        total_reward = sum([r.item() for r in rewards])
        print(f"Episode {episode}, Total Reward: {total_reward}")
    
    # Save the trained models (but not for now)
    #torch.save(policy_net.state_dict(), 'ppo_policy_net.pth')
    #torch.save(value_net.state_dict(), 'ppo_value_net.pth')

def run_simulation(graph, k=2):
    num_nodes = len(graph.nodes())
    policy_net = PolicyNetwork(num_nodes)
    policy_net.load_state_dict(torch.load('ppo_policy_net.pth'))
    policy_net.eval()
    
    env = NetworkEnv(graph, k=k)
    state = env.reset()
    timestep = 0
    
    # Initialize graph position for visualization
    pos = nx.spring_layout(graph)
    # Create mapping from node objects to node IDs
    
    while True:
        plt.clf()  # Clear the previous plot
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = policy_net(state_tensor)
        logits_np = logits.detach().numpy().flatten()
        topk_indices = np.argpartition(-logits_np, k)[:k]
        best_action = topk_indices
        
        # Print information
        print(f"Timestep {timestep}, Best action: {best_action}")
        
        node_obj_to_id = {data['obj']: node_id for node_id, data in graph.nodes(data=True)}

        # Visualize the graph using NetworkVis
        NetworkVis.do_things(
            env.graph,     # Use the environment's graph
            pos,           # Positions of nodes
            best_action,   # Seeded nodes (indices of nodes to activate)
            node_obj_to_id,
            timestep
        )
        
        # Step the environment
        next_state, reward, done, _, info = env.step(logits_np)
        
        # Update state and timestep
        state = next_state
        timestep += 1

if __name__ == "__main__":
    # Initialize the graph
    num_nodes = 200
    num_edges = 300
    k = 2  # Number of nodes to activate
    G = ns.init_random_graph(
        num_nodes, 
        num_edges, 
        activation_chance=0.1, 
        deactivation_chance=0.3, 
        active_activation_chance=0.95, 
        active_deactivation_chance=0.05
    )
    
    # Training
    train_policy(G, k=k, num_episodes=500)
    
    # Simulation
    run_simulation(G, k=k)
