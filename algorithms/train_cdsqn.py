import torch
import numpy as np
import copy
from algorithms.submodular_dqn import SubmodularDQN
from algorithms.cdsqn_env import CDSQNEnv
from networkSim import NetworkSim
import matplotlib.pyplot as plt
import os

def main():
    # 1. Setup Environment
    print("Setting up environment...")
    num_nodes = 20
    num_edges = 40
    num_actions = 3
    cascade_prob = 0.1
    
    # Generate graph
    initial_graph = NetworkSim.init_random_graph(num_nodes, num_edges, 1.0, 5.0)
    
    config = {
        'graph': copy.deepcopy(initial_graph),
        'num_nodes': num_nodes,
        'cascade_prob': cascade_prob,
        'stop_percent': 0.9,
        'gamma': 0.99,
        'num_actions': num_actions,
        'batch_size': 16 # Small batch for demo
    }
    
    env = CDSQNEnv(config)
    
    # 2. Setup Agent
    print("Initializing CDSQN Agent...")
    agent = SubmodularDQN(config)
    
    # 3. Training Loop
    num_episodes = 20 # Short run for demo
    rewards_history = []
    
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # We need to access the graph for availability checks in greedy selection
        # CDSQNEnv updates self.graph in place
        
        while not done:
            # Action Selection
            # Pass env.graph to the agent
            action = agent.select_action(env.graph, training=True)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Push to memory
            # Note: obs is a Dict {x, edge_index}
            # The ReplayBuffer expects separate items? 
            # SubmodularDQN.ReplayBuffer pushes (state_data, action, reward, next_state_data, done)
            # state_data needs to be convertable to PyG Batch later.
            # CDSQNEnv returns x and edge_index numpy arrays.
            # We need to construct Data object or list of Dicts.
            # SubmodularDQN.optimize expects `PyGBatch.from_data_list(state_data_list)`
            # So state_data should be a PyG Data object.
            
            # Reconstruction of PyG Data from obs dict
            from torch_geometric.data import Data
            
            current_data = Data(
                x=torch.tensor(obs['x'], dtype=torch.float), 
                edge_index=torch.tensor(obs['edge_index'], dtype=torch.long)
            )
            
            next_data_pyg = Data(
                x=torch.tensor(next_obs['x'], dtype=torch.float),
                edge_index=torch.tensor(next_obs['edge_index'], dtype=torch.long)
            )
            
            agent.memory.push(current_data, action, reward, next_data_pyg, done)
            agent.optimize()
            
            obs = next_obs
            episode_reward += reward
            
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.4f}, Epsilon = {agent.epsilon:.4f}")
        
    # Plot
    plt.figure()
    plt.plot(range(1, num_episodes + 1), rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CDSQN Training Rewards')
    plt.savefig('results/cdsqn_rewards.png')
    print("Training finished. Results saved to results/cdsqn_rewards.png")

if __name__ == "__main__":
    main()
