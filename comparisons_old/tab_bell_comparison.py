# tab_bell_comparison.py

import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from networkSim import NetworkSim as ns
from algorithms.deepq import train_dqn_agent, select_action_dqn
from algorithms.tabularbellman import TabularBellman  # Assume tabular Q-learning-based Bellman is implemented

# Parameters
NUM_NODES = 15
NUM_EDGES = 45
NUM_SIMULATIONS = 100
NUM_TIMESTEPS = 15
NUM_ACTIONS = 2  # number of nodes to activate each timestep
CASCADE_PROB = 0.1
GAMMA = 0.9  # discount factor for DQN
STOP_PERCENT = 0.5


def build_node_obj_to_id_mapping(graph):
    """
    Build a dictionary mapping node objects to their node IDs.
    """
    node_obj_to_id = {}
    for node_id in graph.nodes():
        node_obj = graph.nodes[node_id]['obj']
        node_obj_to_id[node_obj] = node_id
    return node_obj_to_id


def run_single_simulation(graph, num_actions, model_normal, tab_bell):
    """
    Run a single simulation comparing Tabular Bellman, DQN Normal, and No-Selection policies.
    Returns three lists of rewards per timestep: rewards_tab_bell, rewards_dqn, rewards_no.
    """
    graph_tab = copy.deepcopy(graph)
    graph_dqn = copy.deepcopy(graph)
    graph_no = copy.deepcopy(graph)

    node_obj_to_id_tab = build_node_obj_to_id_mapping(graph_tab)
    node_obj_to_id_dqn = build_node_obj_to_id_mapping(graph_dqn)
    node_obj_to_id_no = build_node_obj_to_id_mapping(graph_no)

    rewards_tab_bell = []
    rewards_dqn = []
    rewards_no = []

    for t in range(NUM_TIMESTEPS):
        # --- Tabular Bellman ---
        tab_nodes, _ = tab_bell.get_best_action_nodes(graph_tab)
        exempt_nodes_tab = set(tab_nodes)
        ns.passive_state_transition_without_neighbors(graph_tab, exempt_nodes=exempt_nodes_tab)
        ns.active_state_transition(tab_nodes)
        ns.independent_cascade_allNodes(graph_tab, CASCADE_PROB)
        ns.rearm_nodes(graph_tab)

        tab_node_ids = {node_obj_to_id_tab[nobj] for nobj in tab_nodes}
        reward_tab = ns.reward_function(graph_tab, tab_node_ids)
        rewards_tab_bell.append(reward_tab)

        # --- DQN Normal ---
        dqn_nodes = select_action_dqn(graph_dqn, model_normal, num_actions)
        exempt_nodes_dqn = set(dqn_nodes)
        ns.passive_state_transition_without_neighbors(graph_dqn, exempt_nodes=exempt_nodes_dqn)
        ns.active_state_transition(dqn_nodes)
        ns.independent_cascade_allNodes(graph_dqn, CASCADE_PROB)
        ns.rearm_nodes(graph_dqn)

        dqn_node_ids = {node_obj_to_id_dqn[nobj] for nobj in dqn_nodes}
        reward_dqn = ns.reward_function(graph_dqn, dqn_node_ids)
        rewards_dqn.append(reward_dqn)

        # --- No-Selection Scenario ---
        # No nodes are chosen each timestep
        no_nodes = []  # empty selection
        exempt_nodes_no = set(no_nodes)
        ns.passive_state_transition_without_neighbors(graph_no, exempt_nodes=exempt_nodes_no)
        ns.active_state_transition(no_nodes)  # no nodes activated
        ns.independent_cascade_allNodes(graph_no, CASCADE_PROB)
        ns.rearm_nodes(graph_no)

        no_node_ids = {node_obj_to_id_no[nobj] for nobj in no_nodes}  # empty set
        reward_no = ns.reward_function(graph_no, no_node_ids)
        rewards_no.append(reward_no)

    return rewards_tab_bell, rewards_dqn, rewards_no


def main():
    # Initialize the graph
    G = ns.init_random_graph(NUM_NODES, NUM_EDGES, 1, 10)

    # Train DQN agent
    config_normal = {
        "graph": copy.deepcopy(G),
        "num_nodes": NUM_NODES,
        "cascade_prob": CASCADE_PROB,
        "stop_percent": STOP_PERCENT,
        "reward_function": "normal"
    }
    print("Training DQN agent...")
    model_normal, policy_normal = train_dqn_agent(config_normal, NUM_ACTIONS)

    # Initialize Tabular Bellman policy
    print("Training Tabular Bellman Q-table...")
    tab_bell = TabularBellman(G, num_actions=NUM_ACTIONS, gamma=GAMMA, alpha=0.8)
    tab_bell.update_q_table(num_episodes=2000, steps_per_episode=500, epsilon=0.1)

    # Run simulations for Tab_Bell, DQN, and NoSelection
    all_rewards_tab = []
    all_rewards_dqn = []
    all_rewards_no = []

    print("Running simulations...")
    for sim in range(1, NUM_SIMULATIONS + 1):
        r_tab, r_dqn, r_no = run_single_simulation(G, NUM_ACTIONS, model_normal, tab_bell)
        all_rewards_tab.append(r_tab)
        all_rewards_dqn.append(r_dqn)
        all_rewards_no.append(r_no)
        if sim % 10 == 0:
            print(f"Completed simulation {sim}/{NUM_SIMULATIONS}")

    # Convert to arrays
    all_rewards_tab = np.array(all_rewards_tab)  # [NUM_SIMULATIONS, NUM_TIMESTEPS]
    all_rewards_dqn = np.array(all_rewards_dqn)
    all_rewards_no = np.array(all_rewards_no)

    # Compute mean rewards for Tab_Bell, DQN, and NoSelection
    mean_rewards_tab = all_rewards_tab.mean(axis=0)
    mean_rewards_dqn = all_rewards_dqn.mean(axis=0)
    mean_rewards_no = all_rewards_no.mean(axis=0)

    # Compute cumulative rewards
    cum_rewards_tab = np.cumsum(mean_rewards_tab)
    cum_rewards_dqn = np.cumsum(mean_rewards_dqn)
    cum_rewards_no = np.cumsum(mean_rewards_no)

    # Save results to CSV
    results = {
        'Timestep': range(1, NUM_TIMESTEPS + 1),
        'NoSelection_mean_reward': mean_rewards_no,
        'Tab_Bell_mean_reward': mean_rewards_tab,
        'DQN_mean_reward': mean_rewards_dqn,
        'NoSelection_cum_reward': cum_rewards_no,
        'Tab_Bell_cum_reward': cum_rewards_tab,
        'DQN_cum_reward': cum_rewards_dqn
    }
    df = pd.DataFrame(results)
    if not os.path.exists('results'):
        os.makedirs('results')
    df.to_csv('results/tab_bell_dqn_noselection_comparison.csv', index=False)

    # Plot results: Mean Reward Comparison
    plt.figure(figsize=(12,6))
    plt.plot(df['Timestep'], df['NoSelection_mean_reward'], label='NoSelection Mean Reward', marker='x')
    plt.plot(df['Timestep'], df['Tab_Bell_mean_reward'], label='Tab Bell Mean Reward', marker='o')
    plt.plot(df['Timestep'], df['DQN_mean_reward'], label='DQN Mean Reward', marker='s')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward per Timestep (NoSelection, Tab_Bell, DQN)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/mean_reward_all_comparison.png')
    plt.close()

    # Plot results: Cumulative Reward Comparison
    plt.figure(figsize=(12,6))
    plt.plot(df['Timestep'], df['NoSelection_cum_reward'], label='NoSelection Cumulative Reward', marker='x')
    plt.plot(df['Timestep'], df['Tab_Bell_cum_reward'], label='Tab Bell Cumulative Reward', marker='o')
    plt.plot(df['Timestep'], df['DQN_cum_reward'], label='DQN Cumulative Reward', marker='s')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward per Timestep (NoSelection, Tab_Bell, DQN)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/cum_reward_all_comparison.png')
    plt.close()

    print("Finished. Results and plots saved to 'results' directory.")


if __name__ == "__main__":
    main()
