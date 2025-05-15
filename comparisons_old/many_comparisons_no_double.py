# many_comparisons.py

import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from networkSim import NetworkSim as ns
from algorithms.hillClimb import HillClimb
from algorithms.deepq import train_dqn_agent, select_action_dqn
from algorithms.whittle import WhittleIndexPolicy  # Import Whittle policy

# Parameters
NUM_SIMULATIONS = 100  # Number of simulations per graph size
NUM_TIMESTEPS = 15     # Number of timesteps per simulation
UPDATE_INTERVAL = 10   # Interval to print progress
CASCADE_PROB = 0.1
GAMMA = 0.99
NUM_ACTIONS = 10

def determine_stop_percent(num_nodes):
    """
    Determines the stop_percent based on the number of nodes.
    """
    if num_nodes > 2000:
        return 0.05
    if num_nodes > 450:
        return 0.15
    if num_nodes == 50:
        return 0.5
    elif num_nodes == 100:
        return 0.5
    elif num_nodes == 150:
        return 0.3
    elif num_nodes == 200:
        return 0.25
    else:
        return 0.25  # Default

def run_single_simulation(initial_graph, num_actions, model_normal, whittle_policy):
    """
    Runs a single simulation applying Hill Climb, DQN Normal, and Whittle policies.
    Returns three arrays of rewards per timestep: rewards_hc, rewards_dqn, rewards_whittle.
    """
    graph_hc = copy.deepcopy(initial_graph)
    graph_dqn_normal = copy.deepcopy(initial_graph)
    graph_whittle = copy.deepcopy(initial_graph)

    rewards_hc = []
    rewards_dqn = []
    rewards_whittle = []

    for t in range(NUM_TIMESTEPS):
        # Hill Climb
        hc_nodes = HillClimb.hill_climb(graph_hc, num=num_actions)
        ns.passive_state_transition_without_neighbors(graph_hc, exempt_nodes=hc_nodes)
        ns.active_state_transition(hc_nodes)
        ns.independent_cascade_allNodes(graph_hc, CASCADE_PROB)
        ns.rearm_nodes(graph_hc)
        reward_hc = ns.reward_function(graph_hc, set([node for node in graph_hc.nodes() if graph_hc.nodes[node]['obj'].isActive()]))
        rewards_hc.append(reward_hc)

        # DQN Normal
        dqn_nodes = select_action_dqn(graph_dqn_normal, model_normal, num_actions)
        ns.passive_state_transition_without_neighbors(graph_dqn_normal, exempt_nodes=dqn_nodes)
        ns.active_state_transition(dqn_nodes)
        ns.independent_cascade_allNodes(graph_dqn_normal, CASCADE_PROB)
        ns.rearm_nodes(graph_dqn_normal)
        reward_dqn = ns.reward_function(graph_dqn_normal, set([node for node in graph_dqn_normal.nodes() if graph_dqn_normal.nodes[node]['obj'].isActive()]))
        rewards_dqn.append(reward_dqn)

        # Whittle
        current_states_whittle = {node: int(graph_whittle.nodes[node]['obj'].isActive()) for node in graph_whittle.nodes()}
        whittle_indices = whittle_policy.compute_whittle_indices(current_states_whittle)
        seeded_nodes_whittle_ids = whittle_policy.select_top_k(whittle_indices, num_actions)
        seeded_nodes_whittle = [graph_whittle.nodes[node_id]['obj'] for node_id in seeded_nodes_whittle_ids]

        ns.passive_state_transition_without_neighbors(graph_whittle, exempt_nodes=seeded_nodes_whittle)
        ns.active_state_transition(seeded_nodes_whittle)
        ns.independent_cascade_allNodes(graph_whittle, CASCADE_PROB)
        ns.rearm_nodes(graph_whittle)
        reward_whittle = ns.reward_function(graph_whittle, set([node for node in graph_whittle.nodes() if graph_whittle.nodes[node]['obj'].isActive()]))
        rewards_whittle.append(reward_whittle)

    return np.array(rewards_hc), np.array(rewards_dqn), np.array(rewards_whittle)

def run_simulations_for_graph_size(graph_size, num_actions, num_simulations, gamma):
    """
    For a given graph size, trains agents and runs multiple simulations.
    Collects rewards for each algorithm and computes mean/cumulative rewards.
    """
    num_nodes, num_edges = graph_size
    print(f"\n=== Running simulations for Graph with {num_nodes} nodes and {num_edges} edges ===\n")

    # Initialize the graph
    initial_graph = ns.init_random_graph(num_nodes, num_edges, 1, 2)
    stop_percent = determine_stop_percent(num_nodes)

    # Train DQN normal agent
    config_normal = {
        "graph": copy.deepcopy(initial_graph),
        "num_nodes": num_nodes,
        "cascade_prob": CASCADE_PROB,
        "stop_percent": stop_percent,
        "reward_function": "normal"
    }
    print("Training DQN agent with normal reward function...")
    model_normal, policy_normal = train_dqn_agent(config_normal, num_actions)

    # Prepare Whittle policy
    transitions_whittle = {}
    node_values = {}
    for node_id in initial_graph.nodes():
        node_obj = initial_graph.nodes[node_id]['obj']
        transition_matrix = np.zeros((2, 2, 2))

        # From Passive state (s=0)
        transition_matrix[0, 0, 1] = node_obj.passive_activation_passive
        transition_matrix[0, 0, 0] = 1 - node_obj.passive_activation_passive
        transition_matrix[0, 1, 1] = node_obj.passive_activation_active
        transition_matrix[0, 1, 0] = 1 - node_obj.passive_activation_active

        # From Active state (s=1)
        transition_matrix[1, 0, 1] = node_obj.active_activation_passive
        transition_matrix[1, 0, 0] = 1 - node_obj.active_activation_passive
        transition_matrix[1, 1, 1] = node_obj.active_activation_active
        transition_matrix[1, 1, 0] = 1 - node_obj.active_activation_active

        transitions_whittle[node_id] = transition_matrix
        node_values[node_id] = node_obj.getValue()

    whittle_policy = WhittleIndexPolicy(
        transitions=transitions_whittle,
        node_values=node_values,
        discount=gamma,
        subsidy_break=0.0,
        eps=1e-4
    )

    # Run simulations and record rewards
    all_rewards_hc = []
    all_rewards_dqn = []
    all_rewards_whittle = []

    for sim in range(1, num_simulations + 1):
        if sim % UPDATE_INTERVAL == 0 or sim == 1:
            print(f"Starting simulation {sim}/{num_simulations}...")
        r_hc, r_dqn, r_whittle = run_single_simulation(initial_graph, num_actions, model_normal, whittle_policy)
        all_rewards_hc.append(r_hc)
        all_rewards_dqn.append(r_dqn)
        all_rewards_whittle.append(r_whittle)

    # Convert to arrays
    all_rewards_hc = np.array(all_rewards_hc)        # shape [num_simulations, num_timesteps]
    all_rewards_dqn = np.array(all_rewards_dqn)
    all_rewards_whittle = np.array(all_rewards_whittle)

    # Compute mean rewards per timestep
    mean_rewards_hc = all_rewards_hc.mean(axis=0)
    mean_rewards_dqn = all_rewards_dqn.mean(axis=0)
    mean_rewards_whittle = all_rewards_whittle.mean(axis=0)

    # Compute cumulative rewards
    cum_rewards_hc = np.cumsum(mean_rewards_hc)
    cum_rewards_dqn = np.cumsum(mean_rewards_dqn)
    cum_rewards_whittle = np.cumsum(mean_rewards_whittle)

    # Save results to CSV
    plots = {
        'Timestep': range(1, NUM_TIMESTEPS + 1),
        'HillClimb_mean_reward': mean_rewards_hc,
        'DQN_mean_reward': mean_rewards_dqn,
        'Whittle_mean_reward': mean_rewards_whittle,
        'HillClimb_cum_reward': cum_rewards_hc,
        'DQN_cum_reward': cum_rewards_dqn,
        'Whittle_cum_reward': cum_rewards_whittle
    }
    df = pd.DataFrame(plots)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    filename = f'many_comparisons_{num_nodes}nodes_{num_edges}edges.csv'
    df.to_csv(os.path.join('plots', filename), index=False)

    # Plot mean rewards
    plt.figure(figsize=(12,6))
    plt.plot(range(1, NUM_TIMESTEPS+1), mean_rewards_hc, label='Hill Climb Mean Reward')
    plt.plot(range(1, NUM_TIMESTEPS+1), mean_rewards_dqn, label='DQN Mean Reward')
    plt.plot(range(1, NUM_TIMESTEPS+1), mean_rewards_whittle, label='Whittle Mean Reward')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward per Timestep')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('plots', f'mean_reward_comparison_{num_nodes}nodes_{num_edges}edges.png'))
    plt.close()

    # Plot cumulative rewards
    plt.figure(figsize=(12,6))
    plt.plot(range(1, NUM_TIMESTEPS+1), cum_rewards_hc, label='Hill Climb Cumulative Reward')
    plt.plot(range(1, NUM_TIMESTEPS+1), cum_rewards_dqn, label='DQN Cumulative Reward')
    plt.plot(range(1, NUM_TIMESTEPS+1), cum_rewards_whittle, label='Whittle Cumulative Reward')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward per Timestep')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('plots', f'cum_reward_comparison_{num_nodes}nodes_{num_edges}edges.png'))
    plt.close()

    print("Finished. Results saved to 'plots' directory.")

def main():
    # Define the graph sizes to test
    graph_sizes = [
        (200, 1000),
        (900, 1800),
        (2500, 5000)
    ]

    for graph_size in graph_sizes:
        run_simulations_for_graph_size(
            graph_size,
            NUM_ACTIONS,
            NUM_SIMULATIONS,
            GAMMA
        )

if __name__ == "__main__":
    main()
