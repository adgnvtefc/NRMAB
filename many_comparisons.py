# many_comparisons.py

import numpy as np
import copy
import random
import multiprocessing as mp

from networkSim import NetworkSim as ns
from hillClimb import HillClimb
from deepq import train_dqn_agent, select_action_dqn
from doubleq import train_double_dqn_agent, select_action_double_dqn

import matplotlib.pyplot as plt
from collections import defaultdict

import csv
import os
import pandas as pd
import seaborn as sns

# Activation chances
PASSIVE_ACTIVATION_CHANCE = 0.1
PASSIVE_DEACTIVATION_CHANCE = 0.3 #currently fails to train if you set this to 0.4 (never hits 80% activation threshold)
ACTIVE_ACTIVATION_CHANCE = 0.8
ACTIVE_DEACTIVATION_CHANCE = 0.1
CASCADE_PROB = 0.1

# Simulation parameters
NUM_SIMULATIONS = 100  # Number of simulations per graph size
NUM_TIMESTEPS = 15     # Number of timesteps per simulation
UPDATE_INTERVAL = 10   # Interval to print progress

def collect_data_single_simulation(algorithm, graph, seeded_nodes, simulation_data, timestep, gamma):
    """
    Collects and updates metrics for a single algorithm in a single simulation timestep.
    """
    # Calculate metrics
    total_nodes = len(graph.nodes())
    active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())

    # Update cumulative active nodes (sum over time)
    simulation_data[algorithm]['cumulative_active_nodes'] += active_nodes

    # Update cumulative seeds used
    simulation_data[algorithm]['cumulative_seeds_used'] += len(seeded_nodes)

    # Calculate percentage of network activated (current timestep)
    percent_activated = (active_nodes / total_nodes) * 100

    # Activation Efficiency
    if simulation_data[algorithm]['cumulative_seeds_used'] > 0:
        activation_efficiency = active_nodes / simulation_data[algorithm]['cumulative_seeds_used']
    else:
        activation_efficiency = 0

    # Discounted activation
    simulation_data[algorithm]['discounted_activation_prev'] = (
        simulation_data[algorithm]['discounted_activation_prev'] * gamma + active_nodes
    )

    # Record active nodes at current timestep
    simulation_data[algorithm]['active_nodes_over_time'].append(active_nodes)

def run_single_simulation(initial_graph, num_actions, model_normal, model_double_dqn, gamma):
    """
    Runs a single simulation applying all algorithms to separate copies of the initial graph.
    Returns the active nodes over time for each algorithm.
    """
    # Initialize fresh copies of the graph for each algorithm
    graph_hc = copy.deepcopy(initial_graph)
    graph_dqn_normal = copy.deepcopy(initial_graph)
    graph_double_dqn = copy.deepcopy(initial_graph)
    graph_random_selection = copy.deepcopy(initial_graph)
    graph_no_selection = copy.deepcopy(initial_graph)

    # Initialize data structures for this simulation
    algorithms = ['Hill Climb', 'DQN Normal', 'Double DQN', 'Random Selection', 'No Selection']
    simulation_data = {algo: {
        'cumulative_seeds_used': 0,
        'cumulative_active_nodes': 0,
        'discounted_activation_prev': 0,
        'active_nodes_over_time': [],
    } for algo in algorithms}

    for timestep in range(1, NUM_TIMESTEPS + 1):
        # ----------------- HILL CLIMB -----------------
        seeded_nodes_hc = HillClimb.hill_climb(graph_hc, num=num_actions)
        ns.passive_state_transition_without_neighbors(graph_hc, exempt_nodes=seeded_nodes_hc)
        ns.active_state_transition(seeded_nodes_hc)
        ns.independent_cascade_allNodes(graph_hc, CASCADE_PROB)
        ns.rearm_nodes(graph_hc)
        collect_data_single_simulation(
            'Hill Climb',
            graph_hc,
            seeded_nodes_hc,
            simulation_data,
            timestep,
            gamma
        )

        # ----------------- DQN NORMAL -----------------
        seeded_nodes_dqn_normal = select_action_dqn(graph_dqn_normal, model_normal, num_actions)
        ns.passive_state_transition_without_neighbors(graph_dqn_normal, exempt_nodes=seeded_nodes_dqn_normal)
        ns.active_state_transition(seeded_nodes_dqn_normal)
        ns.independent_cascade_allNodes(graph_dqn_normal, CASCADE_PROB)
        ns.rearm_nodes(graph_dqn_normal)
        collect_data_single_simulation(
            'DQN Normal',
            graph_dqn_normal,
            seeded_nodes_dqn_normal,
            simulation_data,
            timestep,
            gamma
        )

        # ----------------- DOUBLE DQN -----------------
        seeded_nodes_double_dqn = select_action_double_dqn(graph_double_dqn, model_double_dqn, num_actions)
        ns.passive_state_transition_without_neighbors(graph_double_dqn, exempt_nodes=seeded_nodes_double_dqn)
        ns.active_state_transition(seeded_nodes_double_dqn)
        ns.independent_cascade_allNodes(graph_double_dqn, CASCADE_PROB)
        ns.rearm_nodes(graph_double_dqn)
        collect_data_single_simulation(
            'Double DQN',
            graph_double_dqn,
            seeded_nodes_double_dqn,
            simulation_data,
            timestep,
            gamma
        )

        # ----------------- RANDOM SELECTION -----------------
        random_nodes_indices = random.sample(range(len(graph_random_selection.nodes())), num_actions)
        seeded_nodes_random = [graph_random_selection.nodes[node_index]['obj'] for node_index in random_nodes_indices]
        ns.passive_state_transition_without_neighbors(graph_random_selection, exempt_nodes=seeded_nodes_random)
        ns.active_state_transition(seeded_nodes_random)
        ns.independent_cascade_allNodes(graph_random_selection, CASCADE_PROB)
        ns.rearm_nodes(graph_random_selection)
        collect_data_single_simulation(
            'Random Selection',
            graph_random_selection,
            seeded_nodes_random,
            simulation_data,
            timestep,
            gamma
        )

        # ----------------- NO SELECTION -----------------
        seeded_nodes_none = []
        ns.passive_state_transition_without_neighbors(graph_no_selection, exempt_nodes=[])
        ns.independent_cascade_allNodes(graph_no_selection, CASCADE_PROB)
        ns.rearm_nodes(graph_no_selection)
        collect_data_single_simulation(
            'No Selection',
            graph_no_selection,
            seeded_nodes_none,
            simulation_data,
            timestep,
            gamma
        )

    # After simulation, extract active nodes over time for each algorithm
    final_active_nodes = {algo: simulation_data[algo]['active_nodes_over_time'] for algo in algorithms}

    return final_active_nodes

def run_simulations_for_graph_size(graph_size, num_actions, num_simulations, gamma):
    """
    Trains agents for a given graph size and runs multiple simulations.
    Returns aggregated outperformance percentages, individual simulation results, and performance metrics.
    """
    num_nodes, num_edges = graph_size
    print(f"\n=== Running simulations for Graph with {num_nodes} nodes and {num_edges} edges ===\n")

    # Initialize the initial graph
    initial_graph = ns.init_random_graph(
        num_nodes,
        num_edges,
        PASSIVE_ACTIVATION_CHANCE,
        PASSIVE_DEACTIVATION_CHANCE,
        ACTIVE_ACTIVATION_CHANCE,
        ACTIVE_DEACTIVATION_CHANCE
    )

    # Determine stop_percent based on number of nodes
    stop_percent = determine_stop_percent(num_nodes)

    # ----------------- DQN AGENT WITH NORMAL REWARD FUNCTION -----------------
    config_normal = {
        "graph": copy.deepcopy(initial_graph),
        "num_nodes": num_nodes,
        "cascade_prob": CASCADE_PROB,
        "stop_percent": stop_percent,
        "reward_function": "normal"
    }
    print("Training DQN agent with normal reward function...")
    model_normal, policy_normal = train_dqn_agent(config_normal, num_actions)

    # ----------------- DOUBLE DQN AGENT -----------------
    config_double_dqn = {
        "graph": copy.deepcopy(initial_graph),
        "num_nodes": num_nodes,
        "cascade_prob": CASCADE_PROB,
        "stop_percent": stop_percent,
        "reward_function": "normal"
    }
    print("Training Double DQN agent...")
    model_double_dqn, policy_double_dqn = train_double_dqn_agent(config_double_dqn, num_actions)

    # Initialize outperformance counts for this graph size
    # Keys: 'DQN Normal vs Hill Climb', 'Double DQN vs Hill Climb', etc.
    outperformance_counts = defaultdict(int)

    # Initialize a list to store individual simulation results for this graph size
    individual_simulation_results = []

    # Initialize storage for active nodes over all simulations for this graph size
    # Structure: {algo: [ [sim1_t1, sim1_t2, ...], [sim2_t1, sim2_t2, ...], ... ]}
    all_active_nodes = {algo: [] for algo in ['Hill Climb', 'DQN Normal', 'Double DQN', 'Random Selection', 'No Selection']}

    # Run simulations
    for sim in range(1, num_simulations + 1):
        if sim % UPDATE_INTERVAL == 0 or sim == 1:
            print(f"Starting simulation {sim}/{num_simulations}...")
        final_active_nodes = run_single_simulation(initial_graph, num_actions, model_normal, model_double_dqn, gamma)

        # Store active nodes for aggregate metrics
        for algo in all_active_nodes:
            all_active_nodes[algo].append(final_active_nodes[algo])

        # Compare active nodes at each timestep across all simulations
        # Iterate over each timestep
        for timestep in range(NUM_TIMESTEPS):
            # Retrieve active node counts for each algorithm at this timestep
            try:
                hc_active = final_active_nodes['Hill Climb'][timestep]
                dqn_normal_active = final_active_nodes['DQN Normal'][timestep]
                double_dqn_active = final_active_nodes['Double DQN'][timestep]
            except IndexError:
                # Handle cases where a simulation might have fewer timesteps
                continue

            # Compare DQN Normal vs Hill Climb
            if dqn_normal_active > hc_active:
                outperformance_counts['DQN Normal vs Hill Climb'] += 1

            # Compare Double DQN vs Hill Climb
            if double_dqn_active > hc_active:
                outperformance_counts['Double DQN vs Hill Climb'] += 1

            # If you wish to compare against other baselines like Random Selection or No Selection, include them here
            # Example:
            # random_selection_active = final_active_nodes['Random Selection'][timestep]
            # if dqn_normal_active > random_selection_active:
            #     outperformance_counts['DQN Normal vs Random Selection'] += 1
            # if double_dqn_active > random_selection_active:
            #     outperformance_counts['Double DQN vs Random Selection'] += 1

        # Optionally, you can also record final activation percentages for additional analyses
        final_percent = calculate_final_percent_activated(final_active_nodes, num_nodes, NUM_TIMESTEPS)
        simulation_result = {'Simulation': sim}
        for algo, percent in final_percent.items():
            simulation_result[algo] = percent
        individual_simulation_results.append(simulation_result)

    # Calculate outperformance percentages for this graph size
    total_comparisons = num_simulations * NUM_TIMESTEPS
    outperformance_percentages = {key: (count / total_comparisons) * 100 for key, count in outperformance_counts.items()}

    # Aggregate performance metrics for this graph size
    performance_metrics = analyze_performance(all_active_nodes)

    print(f"\n=== Completed {num_simulations} simulations for Graph Size {graph_size} ===")
    for key, percentage in outperformance_percentages.items():
        print(f"{key}: {percentage:.2f}%")

    # Store results in the overall data structures
    return outperformance_percentages, individual_simulation_results, performance_metrics

def determine_stop_percent(num_nodes):
    """
    Determines the stop_percent based on the number of nodes.
    """
    if num_nodes == 50:
        return 0.8
    elif num_nodes == 100:
        return 0.5
    elif num_nodes == 150:
        return 0.3
    elif num_nodes == 200:
        return 0.25
    else:
        return 0.5  # Default value if not specified

def calculate_final_percent_activated(final_active_nodes, num_nodes, num_timesteps):
    """
    Calculates the final percent activated for each algorithm.
    """
    final_percent_activated = {}
    for algo, active_nodes_over_time in final_active_nodes.items():
        # Sum active nodes across all timesteps and normalize
        total_active = sum(active_nodes_over_time)
        max_possible = num_nodes * num_timesteps
        percent = (total_active / max_possible) * 100
        final_percent_activated[algo] = percent
    return final_percent_activated

def analyze_performance(all_active_nodes, thresholds=[25, 50, 75, 90]):
    """
    Analyzes performance metrics across all simulations for a given graph size.
    
    Args:
        all_active_nodes (dict): Dictionary mapping algorithms to lists of active node counts over simulations.
        thresholds (list): Activation thresholds in percentage.
    
    Returns:
        dict: Performance metrics for each algorithm.
    """
    performance_metrics = {}
    for algo, simulations in all_active_nodes.items():
        # Convert list of active node lists to numpy array for easier manipulation
        # Shape: [num_simulations, num_timesteps]
        active_nodes_matrix = np.array(simulations)  # Each simulation has a list of active nodes per timestep
        
        # Mean and Standard Deviation per timestep
        mean_active = active_nodes_matrix.mean(axis=0)  # Shape: [num_timesteps]
        std_active = active_nodes_matrix.std(axis=0)
        
        # AUAC per simulation, then average
        auac_per_simulation = np.trapz(active_nodes_matrix, axis=1)  # Shape: [num_simulations]
        mean_auac = auac_per_simulation.mean()
        
        # Time to reach thresholds
        time_to_threshold = {}
        for threshold in thresholds:
            # Define target per simulation
            target = (threshold / 100.0) * active_nodes_matrix.max(axis=1)
            times = []
            for sim in range(active_nodes_matrix.shape[0]):
                reached = np.where(active_nodes_matrix[sim] >= target[sim])[0]
                if len(reached) > 0:
                    timestep_reached = reached[0] + 1  # +1 for 1-based timestep
                else:
                    timestep_reached = NUM_TIMESTEPS + 1  # Assign a value beyond the last timestep
                times.append(timestep_reached)
            mean_time = np.nanmean(times)
            time_to_threshold[threshold] = mean_time
        
        # Probability of reaching thresholds
        prob_reach_threshold = {}
        for threshold in thresholds:
            count = np.sum(active_nodes_matrix[:, -1] >= (threshold / 100.0 * active_nodes_matrix.max(axis=1)))
            prob = (count / active_nodes_matrix.shape[0]) * 100
            prob_reach_threshold[threshold] = prob
        
        # Coefficient of Variation (CV) for AUAC
        cv = auac_per_simulation.std() / auac_per_simulation.mean() if auac_per_simulation.mean() != 0 else np.nan
        
        # Store metrics
        performance_metrics[algo] = {
            'mean_active_per_timestep': mean_active,
            'std_active_per_timestep': std_active,
            'auac': mean_auac,
            'time_to_threshold': time_to_threshold,
            'prob_reach_threshold': prob_reach_threshold,
            'coefficient_of_variation': cv
        }
    
    return performance_metrics

def save_individual_results(all_individual_results, output_dir='simulation_results'):
    """
    Saves individual simulation results to CSV files, one per graph size.
    
    Args:
        all_individual_results (dict): Dictionary mapping graph sizes to lists of simulation results.
        output_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for graph_size, results in all_individual_results.items():
        num_nodes, num_edges = graph_size
        filename = f'graph_{num_nodes}nodes_{num_edges}edges_simulation_results.csv'
        filepath = os.path.join(output_dir, filename)
        
        # Determine the fieldnames based on the keys in the first result
        fieldnames = results[0].keys() if results else []
        
        with open(filepath, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"Saved individual simulation results for graph size {graph_size} to {filepath}")

def plot_aggregated_results(overall_outperformance, performance_metrics, graph_sizes):
    """
    Plots the aggregated outperformance percentages and other performance metrics for each graph size.
    
    Args:
        overall_outperformance (dict): {graph_size: {comparison: percentage, ...}, ...}
        performance_metrics (dict): {graph_size: {algorithm: metrics, ...}, ...}
        graph_sizes (list): List of graph size tuples.
    """
    # Plot Outperformance Percentages
    for graph_size in graph_sizes:
        comparisons = overall_outperformance[graph_size].keys()
        values = overall_outperformance[graph_size].values()
        labels = list(overall_outperformance[graph_size].keys())
        values = list(overall_outperformance[graph_size].values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=values, palette="viridis")
        plt.title(f'Outperformance Percentages for Graph Size {graph_size[0]} Nodes x {graph_size[1]} Edges')
        plt.xlabel('Algorithm Comparisons')
        plt.ylabel('Outperformance Percentage (%)')
        plt.ylim(0, 100)
        for index, value in enumerate(values):
            plt.text(index, value + 1, f"{value:.2f}%", ha='center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Plot AUAC for each algorithm across graph sizes
    plt.figure(figsize=(12, 8))
    for algo in ['Hill Climb', 'DQN Normal', 'Double DQN', 'Random Selection', 'No Selection']:
        auac_values = [performance_metrics[size][algo]['auac'] for size in graph_sizes]
        labels = [f"{size[0]}x{size[1]}" for size in graph_sizes]
        sns.lineplot(x=labels, y=auac_values, marker='o', label=algo)
    plt.title('AUAC Comparison Across Graph Sizes')
    plt.xlabel('Graph Size (Nodes x Edges)')
    plt.ylabel('Average AUAC')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot Time to Reach 50% Activation
    plt.figure(figsize=(12, 8))
    for algo in ['Hill Climb', 'DQN Normal', 'Double DQN', 'Random Selection', 'No Selection']:
        times = [performance_metrics[size][algo]['time_to_threshold'].get(50, np.nan) for size in graph_sizes]
        labels = [f"{size[0]}x{size[1]}" for size in graph_sizes]
        sns.lineplot(x=labels, y=times, marker='o', label=algo)
    plt.title('Average Time to Reach 50% Activation Across Graph Sizes')
    plt.xlabel('Graph Size (Nodes x Edges)')
    plt.ylabel('Average Timestep to Reach 50% Activation')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ----------------- ACTIVATION PERCENTAGES GRAPH -----------------
    # Plot Mean Activation Percentage Over Time for Each Algorithm Across Graph Sizes
    for graph_size in graph_sizes:
        plt.figure(figsize=(12, 8))
        for algo in ['Hill Climb', 'DQN Normal', 'Double DQN', 'Random Selection', 'No Selection']:
            mean_active = performance_metrics[graph_size][algo]['mean_active_per_timestep']
            mean_percent = (mean_active / graph_size[0]) * 100  # Assuming graph_size[0] is num_nodes
            plt.plot(range(1, NUM_TIMESTEPS + 1), mean_percent, label=algo)
        plt.title(f'Mean Activation Percentage Over Time for Graph Size {graph_size[0]} Nodes x {graph_size[1]} Edges')
        plt.xlabel('Timestep')
        plt.ylabel('Mean Activation Percentage (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_individual_results(all_individual_results):
    """
    Plots individual simulation results for each graph size and algorithm.
    """
    for graph_size, results in all_individual_results.items():
        num_nodes, num_edges = graph_size
        df = pd.DataFrame(results)
        
        # Melt the DataFrame for seaborn
        df_melted = df.melt(id_vars=['Simulation'], var_name='Algorithm', value_name='Percent Activated')
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Algorithm', y='Percent Activated', data=df_melted)
        plt.title(f'Activation Percentages for Graph Size {num_nodes} Nodes x {num_edges} Edges')
        plt.xlabel('Algorithm')
        plt.ylabel('Percent Activated')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Define the different graph sizes to test
    graph_sizes = [
        (50, 50),
        (100, 100),
        (150, 150),
        (200, 200)
    ]

    num_actions = 10  # Number of nodes to activate per timestep

    # Initialize data collection structures for all graph sizes
    overall_outperformance = {}  # {graph_size: {comparison: percentage, ...}, ...}
    all_individual_results = {}  # {graph_size: [simulation_results, ...], ...}
    all_performance_metrics = {}  # {graph_size: {algorithm: metrics, ...}, ...}

    # Simulation parameters
    simulation_params = {
        'passive_activation_chance': PASSIVE_ACTIVATION_CHANCE,
        'passive_deactivation_chance': PASSIVE_DEACTIVATION_CHANCE,
        'active_activation_chance': ACTIVE_ACTIVATION_CHANCE,
        'active_deactivation_chance': ACTIVE_DEACTIVATION_CHANCE,
        'cascade_prob': CASCADE_PROB,
        'k': num_actions,
        'discount_factor': 0.99  # Discount factor gamma set to 0.99
    }

    gamma = simulation_params['discount_factor']

    for graph_size in graph_sizes:
        outperformance_percentages, individual_results, performance_metrics = run_simulations_for_graph_size(
            graph_size,
            num_actions,
            NUM_SIMULATIONS,
            gamma
        )
        overall_outperformance[graph_size] = outperformance_percentages
        all_individual_results[graph_size] = individual_results
        all_performance_metrics[graph_size] = performance_metrics

    # Save individual simulation results to CSV files
    save_individual_results(all_individual_results)

    # Plot aggregated outperformance percentages and other metrics
    plot_aggregated_results(overall_outperformance, all_performance_metrics, graph_sizes)

    # Plot individual simulation results
    plot_individual_results(all_individual_results)

if __name__ == "__main__":
    main()
