import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import torch
import random
import os  # Added to check for model file existence

from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
from tabularbellman import TabularBellman as tb
from deepQLearning import NeuralQLearner
from simpleNode import SimpleNode as Node
from hillClimb import HillClimb  # Import the HillClimb class

# Activation chance in passive action
PASSIVE_ACTIVATION_CHANCE = 0.1
# Deactivation chance in passive action
PASSIVE_DEACTIVATION_CHANCE = 0.2

ACTIVE_ACTIVATION_CHANCE = 0.95
ACTIVE_DEACTIVATION_CHANCE = 0.05

def main():
    # Initialize the graph
    G = ns.init_random_graph(10, 20, PASSIVE_ACTIVATION_CHANCE, PASSIVE_DEACTIVATION_CHANCE,
                             ACTIVE_ACTIVATION_CHANCE, ACTIVE_DEACTIVATION_CHANCE)

    # Copy the graph for each algorithm to ensure they start from the same initial state
    graph_hill_climb = copy.deepcopy(G)
    graph_hill_climb_bellman = copy.deepcopy(G)  # New graph for hill_climb_with_bellman
    graph_tabular_bellman = copy.deepcopy(G)
    graph_deep_q_learning = copy.deepcopy(G)
    graph_random_selection = copy.deepcopy(G)
    graph_no_selection = copy.deepcopy(G)

    # Initialize agents
    # For Tabular Bellman
    tabular_agent = tb(graph_tabular_bellman, num_actions=2)
    tabular_agent.update_q_table(num_iterations=3, num_samples=3)

    # For Deep Q-Learning
    deep_q_agent = NeuralQLearner(graph_deep_q_learning, num_actions=2, model_path='deep_q_model.pth')
    # Attempt to load the model
    model_loaded = deep_q_agent.load_model()
    if not model_loaded:
        # If model is not loaded, train it and save the model
        deep_q_agent.train(num_episodes=100)
        deep_q_agent.save_model()
    else:
        print("Using the loaded Deep Q-Learning model.")

    # Initialize data collection structures
    algorithms = ['Hill Climb', 'Hill Climb Bellman', 'Tabular Bellman', 'Deep Q-Learning', 'Random Selection', 'No Selection']
    data_collection = {algo: {
        'timestep': [],
        'active_nodes': [],
        'nodes_activated_this_timestep': [],
        'cascade_activated_nodes': [],
        'cumulative_reward': [],
        'total_active_nodes': [],
    } for algo in algorithms}

    # Set the number of timesteps for the simulation
    num_timesteps = 50
    update_interval = 5  # Update the table every 5 timesteps

    # Initialize cumulative variables
    cumulative_rewards = {algo: 0 for algo in algorithms}
    cumulative_active_nodes = {algo: 0 for algo in algorithms}
    cumulative_nodes_activated = {algo: 0 for algo in algorithms}
    cumulative_cascade_activated_nodes = {algo: 0 for algo in algorithms}

    for timestep in range(1, num_timesteps + 1):
        # ----------------- HILL CLIMB -----------------
        # Select action using Hill Climb
        seeded_nodes_hc = HillClimb.hill_climb(graph_hill_climb, num=2)
        # Apply actions and state transitions
        transition_nodes_hc = ns.passive_state_transition_without_neighbors(graph_hill_climb, exempt_nodes=seeded_nodes_hc)
        changed_nodes_hc = ns.active_state_transition([node for node in seeded_nodes_hc])
        newly_activated_hc = ns.independent_cascade_allNodes(graph_hill_climb, 0.1)
        ns.rearm_nodes(graph_hill_climb)
        # Collect data
        collect_data('Hill Climb', graph_hill_climb, seeded_nodes_hc, changed_nodes_hc, newly_activated_hc,
                     cumulative_rewards, cumulative_active_nodes, cumulative_nodes_activated,
                     cumulative_cascade_activated_nodes, data_collection, timestep)

        # ----------------- HILL CLIMB WITH BELLMAN -----------------
        # Select action using Hill Climb with Bellman
        seeded_nodes_hc_bellman = HillClimb.hill_climb_with_bellman(
            graph_hill_climb_bellman, num=2, gamma=0.7, horizon=1, num_samples=1)
        # Apply actions and state transitions
        transition_nodes_hc_bellman = ns.passive_state_transition_without_neighbors(graph_hill_climb_bellman, exempt_nodes=seeded_nodes_hc_bellman)
        changed_nodes_hc_bellman = ns.active_state_transition([node for node in seeded_nodes_hc_bellman])
        newly_activated_hc_bellman = ns.independent_cascade_allNodes(graph_hill_climb_bellman, 0.1)
        ns.rearm_nodes(graph_hill_climb_bellman)
        # Collect data
        collect_data('Hill Climb Bellman', graph_hill_climb_bellman, seeded_nodes_hc_bellman, changed_nodes_hc_bellman, newly_activated_hc_bellman,
                     cumulative_rewards, cumulative_active_nodes, cumulative_nodes_activated,
                     cumulative_cascade_activated_nodes, data_collection, timestep)

        # ----------------- TABULAR BELLMAN -----------------
        # Select action using Tabular Bellman
        best_action_tb, utility_tb = tabular_agent.get_best_action_nodes(graph_tabular_bellman)
        seeded_nodes_tb = best_action_tb
        # Apply actions and state transitions
        transition_nodes_tb = ns.passive_state_transition_without_neighbors(graph_tabular_bellman, exempt_nodes=seeded_nodes_tb)
        changed_nodes_tb = ns.active_state_transition([node for node in seeded_nodes_tb])
        newly_activated_tb = ns.independent_cascade_allNodes(graph_tabular_bellman, 0.1)
        ns.rearm_nodes(graph_tabular_bellman)
        # Collect data
        collect_data('Tabular Bellman', graph_tabular_bellman, seeded_nodes_tb, changed_nodes_tb, newly_activated_tb,
                     cumulative_rewards, cumulative_active_nodes, cumulative_nodes_activated,
                     cumulative_cascade_activated_nodes, data_collection, timestep)

        # ----------------- DEEP Q-LEARNING -----------------
        # Select action using Deep Q-Learning
        state_dql = deep_q_agent.get_state_representation(graph_deep_q_learning)
        possible_actions = list(itertools.combinations(range(deep_q_agent.num_nodes), deep_q_agent.num_actions))
        max_q_value = float('-inf')
        best_action_indices = None
        state_tensor = torch.FloatTensor(state_dql).unsqueeze(0)
        with torch.no_grad():
            for action_indices in possible_actions:
                action_vector = deep_q_agent.get_action_representation(action_indices)
                action_tensor = torch.FloatTensor(action_vector).unsqueeze(0)
                q_value = deep_q_agent.q_network(state_tensor, action_tensor)
                if q_value.item() > max_q_value:
                    max_q_value = q_value.item()
                    best_action_indices = action_indices
        if best_action_indices is None:
            # If no action was found (which shouldn't happen), select random nodes
            best_action_indices = random.sample(range(deep_q_agent.num_nodes), deep_q_agent.num_actions)
        # Apply actions and state transitions
        exempt_nodes = [graph_deep_q_learning.nodes[node_index]['obj'] for node_index in best_action_indices]
        transition_nodes_dql = ns.passive_state_transition_without_neighbors(graph_deep_q_learning, exempt_nodes=exempt_nodes)
        changed_nodes_dql = ns.active_state_transition_graph_indices(graph_deep_q_learning, best_action_indices)
        newly_activated_dql = ns.independent_cascade_allNodes(graph_deep_q_learning, 0.1)
        ns.rearm_nodes(graph_deep_q_learning)
        # Collect data
        collect_data('Deep Q-Learning', graph_deep_q_learning, [graph_deep_q_learning.nodes[node_index]['obj'] for node_index in best_action_indices],
                     changed_nodes_dql, newly_activated_dql, cumulative_rewards, cumulative_active_nodes,
                     cumulative_nodes_activated, cumulative_cascade_activated_nodes, data_collection, timestep)

        # ----------------- RANDOM SELECTION -----------------
        # Select action by randomly choosing nodes
        num_actions_random = 2  # Adjust as needed
        random_nodes_indices = random.sample(range(len(graph_random_selection.nodes())), num_actions_random)
        seeded_nodes_random = [graph_random_selection.nodes[node_index]['obj'] for node_index in random_nodes_indices]
        # Apply actions and state transitions
        exempt_nodes = seeded_nodes_random
        transition_nodes_random = ns.passive_state_transition_without_neighbors(graph_random_selection, exempt_nodes=exempt_nodes)
        changed_nodes_random = ns.active_state_transition(seeded_nodes_random)
        newly_activated_random = ns.independent_cascade_allNodes(graph_random_selection, 0.1)
        ns.rearm_nodes(graph_random_selection)
        # Collect data
        collect_data('Random Selection', graph_random_selection, seeded_nodes_random, changed_nodes_random,
                     newly_activated_random, cumulative_rewards, cumulative_active_nodes, cumulative_nodes_activated,
                     cumulative_cascade_activated_nodes, data_collection, timestep)

        # ----------------- NO SELECTION -----------------
        # No action is taken
        seeded_nodes_none = []
        # Apply actions and state transitions
        transition_nodes_none = ns.passive_state_transition_without_neighbors(graph_no_selection, exempt_nodes=[])
        # Since no nodes are activated, changed_nodes_none is empty
        changed_nodes_none = []
        newly_activated_none = ns.independent_cascade_allNodes(graph_no_selection, 0.1)
        ns.rearm_nodes(graph_no_selection)
        # Collect data
        collect_data('No Selection', graph_no_selection, seeded_nodes_none, changed_nodes_none, newly_activated_none,
                     cumulative_rewards, cumulative_active_nodes, cumulative_nodes_activated,
                     cumulative_cascade_activated_nodes, data_collection, timestep)

        # ----------------- PRINT RESULTS -----------------
        # Every update_interval timesteps, print the data
        if timestep % update_interval == 0 or timestep == num_timesteps:
            print_results(data_collection, cumulative_active_nodes, cumulative_nodes_activated,
                          cumulative_cascade_activated_nodes, cumulative_rewards, timestep)

    # Optionally, plot the results at the end
    plot_results(data_collection)

def collect_data(algorithm, graph, seeded_nodes, changed_nodes, newly_activated_nodes, cumulative_rewards,
                 cumulative_active_nodes, cumulative_nodes_activated, cumulative_cascade_activated_nodes,
                 data_collection, timestep):
    # Calculate metrics
    active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())
    nodes_activated_this_timestep = len(changed_nodes) + len(newly_activated_nodes)
    cascade_activated_nodes = len(newly_activated_nodes)

    # Create node_obj_to_id mapping for the current graph
    node_obj_to_id_current_graph = {data['obj']: node_id for node_id, data in graph.nodes(data=True)}

    # Map seeded_nodes to their corresponding IDs
    seeded_node_ids = [node_obj_to_id_current_graph[node] for node in seeded_nodes]

    reward = ns.reward_function(graph, set(seeded_node_ids))
    cumulative_rewards[algorithm] += reward
    cumulative_active_nodes[algorithm] += active_nodes
    cumulative_nodes_activated[algorithm] += nodes_activated_this_timestep
    cumulative_cascade_activated_nodes[algorithm] += cascade_activated_nodes

    # Store data
    data_collection[algorithm]['timestep'].append(timestep)
    data_collection[algorithm]['active_nodes'].append(active_nodes)
    data_collection[algorithm]['nodes_activated_this_timestep'].append(nodes_activated_this_timestep)
    data_collection[algorithm]['cascade_activated_nodes'].append(cascade_activated_nodes)
    data_collection[algorithm]['cumulative_reward'].append(cumulative_rewards[algorithm])
    data_collection[algorithm]['total_active_nodes'].append(cumulative_active_nodes[algorithm])

def print_results(data_collection, cumulative_active_nodes, cumulative_nodes_activated,
                  cumulative_cascade_activated_nodes, cumulative_rewards, timestep):
    print(f"\n=== Simulation Results at Timestep {timestep} ===")
    header = "{:<20} {:<10} {:<15} {:<20} {:<25} {:<25} {:<25}".format(
        'Algorithm', 'Timestep', 'Active Nodes', 'Cumulative Reward',
        'Avg Active Nodes', 'Avg Nodes Activated', 'Avg Cascade Activated'
    )
    print(header)
    print("=" * len(header))
    for algo in data_collection.keys():
        avg_active_nodes = cumulative_active_nodes[algo] / timestep
        avg_nodes_activated = cumulative_nodes_activated[algo] / timestep
        avg_cascade_activated = cumulative_cascade_activated_nodes[algo] / timestep
        print("{:<20} {:<10} {:<15} {:<20} {:<25} {:<25} {:<25}".format(
            algo,
            data_collection[algo]['timestep'][-1],
            data_collection[algo]['active_nodes'][-1],
            round(cumulative_rewards[algo], 2),
            round(avg_active_nodes, 2),
            round(avg_nodes_activated, 2),
            round(avg_cascade_activated, 2)
        ))
    print("=" * len(header) + "\n")

def plot_results(data_collection):
    import matplotlib.pyplot as plt

    timesteps = data_collection['Hill Climb']['timestep']
    plt.figure(figsize=(15, 8))

    # Plot Active Nodes over Time
    plt.subplot(2, 2, 1)
    for algo in data_collection:
        plt.plot(timesteps, data_collection[algo]['active_nodes'], label=algo)
    plt.xlabel('Timestep')
    plt.ylabel('Number of Active Nodes')
    plt.title('Active Nodes over Time')
    plt.legend()

    # Plot Cumulative Reward over Time
    plt.subplot(2, 2, 2)
    for algo in data_collection:
        plt.plot(timesteps, data_collection[algo]['cumulative_reward'], label=algo)
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Time')
    plt.legend()

    # Plot Average Nodes Activated per Timestep
    plt.subplot(2, 2, 3)
    for algo in data_collection:
        avg_nodes_activated = [sum(data_collection[algo]['nodes_activated_this_timestep'][:i+1]) / (i+1) for i in range(len(timesteps))]
        plt.plot(timesteps, avg_nodes_activated, label=algo)
    plt.xlabel('Timestep')
    plt.ylabel('Avg Nodes Activated per Timestep')
    plt.title('Average Nodes Activated per Timestep')
    plt.legend()

    # Plot Average Cascade Activated Nodes per Timestep
    plt.subplot(2, 2, 4)
    for algo in data_collection:
        avg_cascade_activated = [sum(data_collection[algo]['cascade_activated_nodes'][:i+1]) / (i+1) for i in range(len(timesteps))]
        plt.plot(timesteps, avg_cascade_activated, label=algo)
    plt.xlabel('Timestep')
    plt.ylabel('Avg Cascade Activated Nodes')
    plt.title('Average Cascade Activated Nodes per Timestep')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
