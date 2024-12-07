# many_comparisons.py

import numpy as np
import copy
import random
import multiprocessing as mp

from networkSim import NetworkSim as ns
from hillClimb import HillClimb
from deepq import train_dqn_agent, select_action_dqn
from whittle import WhittleIndexPolicy  # Import Whittle policy

import matplotlib.pyplot as plt
from collections import defaultdict

import csv
import os
import pandas as pd
import seaborn as sns

# Activation chances
CASCADE_PROB = 0.1

# Simulation parameters
NUM_SIMULATIONS = 100  # Number of simulations per graph size
NUM_TIMESTEPS = 15     # Number of timesteps per simulation
UPDATE_INTERVAL = 10   # Interval to print progress

def collect_data_single_simulation(algorithm, graph, seeded_nodes, simulation_data, timestep, gamma):
    """
    Collects and updates metrics for a single algorithm in a single simulation timestep.
    Includes both active node counts and sum of node values.
    """
    # Calculate metrics
    total_nodes = len(graph.nodes())
    active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())
    
    # Calculate sum of node values for active nodes
    active_node_values = sum(graph.nodes[node]['obj'].getValue() for node in graph.nodes() if graph.nodes[node]['obj'].isActive())
    
    # Update cumulative active nodes (sum over time)
    simulation_data[algorithm]['cumulative_active_nodes'] += active_nodes
    
    # Update cumulative node values (sum over time)
    simulation_data[algorithm]['cumulative_node_values'] += active_node_values

    # Update cumulative seeds used
    simulation_data[algorithm]['cumulative_seeds_used'] += len(seeded_nodes)

    # Discounted activation (not currently printed, but stored)
    simulation_data[algorithm]['discounted_activation_prev'] = (
        simulation_data[algorithm]['discounted_activation_prev'] * gamma + active_nodes
    )

    # Record active nodes and node values at current timestep
    simulation_data[algorithm]['active_nodes_over_time'].append(active_nodes)
    simulation_data[algorithm]['active_node_values_over_time'].append(active_node_values)

def run_single_simulation(initial_graph, num_actions, model_normal, whittle_policy, gamma):
    """
    Runs a single simulation applying all algorithms to separate copies of the initial graph.
    Returns the active nodes and node values over time for each algorithm.
    """
    # Initialize fresh copies of the graph for each algorithm
    graph_hc = copy.deepcopy(initial_graph)
    graph_dqn_normal = copy.deepcopy(initial_graph)
    graph_whittle = copy.deepcopy(initial_graph)

    # Add Whittle to the algorithms
    algorithms = ['Hill Climb', 'DQN Normal', 'Whittle']

    simulation_data = {algo: {
        'cumulative_seeds_used': 0,
        'cumulative_active_nodes': 0,
        'cumulative_node_values': 0,
        'discounted_activation_prev': 0,
        'active_nodes_over_time': [],
        'active_node_values_over_time': []
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

        # ----------------- WHITTLE -----------------
        # Compute Whittle indices for current states
        current_states_whittle = {node: int(graph_whittle.nodes[node]['obj'].isActive()) for node in graph_whittle.nodes()}
        whittle_indices = whittle_policy.compute_whittle_indices(current_states_whittle)
        seeded_nodes_whittle_ids = whittle_policy.select_top_k(whittle_indices, num_actions)
        seeded_nodes_whittle = [graph_whittle.nodes[node_id]['obj'] for node_id in seeded_nodes_whittle_ids]

        ns.passive_state_transition_without_neighbors(graph_whittle, exempt_nodes=seeded_nodes_whittle)
        ns.active_state_transition(seeded_nodes_whittle)
        ns.independent_cascade_allNodes(graph_whittle, CASCADE_PROB)
        ns.rearm_nodes(graph_whittle)
        collect_data_single_simulation(
            'Whittle',
            graph_whittle,
            seeded_nodes_whittle,
            simulation_data,
            timestep,
            gamma
        )

    # After simulation, extract active nodes and node values over time for each algorithm
    final_active_nodes = {algo: simulation_data[algo]['active_nodes_over_time'] for algo in algorithms}
    final_node_values = {algo: simulation_data[algo]['active_node_values_over_time'] for algo in algorithms}

    return final_active_nodes, final_node_values

def run_simulations_for_graph_size(graph_size, num_actions, num_simulations, gamma):
    """
    Trains agents for a given graph size and runs multiple simulations.
    Returns aggregated outperformance percentages, individual simulation results, and performance metrics.
    Focus on DQN vs Hill Climb and DQN vs Whittle comparisons.
    """
    num_nodes, num_edges = graph_size
    print(f"\n=== Running simulations for Graph with {num_nodes} nodes and {num_edges} edges ===\n")

    # Initialize the initial graph
    initial_graph = ns.init_random_graph(
        num_nodes,
        num_edges,
        1,
        10
    )

    # Determine stop_percent based on number of nodes
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

    # Prepare Whittle policy data
    # Extract transitions and node values from initial_graph
    transitions_whittle = {}
    node_values = {}
    for node_id in initial_graph.nodes():
        node_obj = initial_graph.nodes[node_id]['obj']
        # Construct the transition matrix
        # States: 0 - Passive, 1 - Active
        # Actions: 0 - Passive action, 1 - Active action
        transition_matrix = np.zeros((2, 2, 2))  # (s, a, next_s)

        # From Passive state (s=0)
        # Action Passive (a=0)
        transition_matrix[0, 0, 1] = node_obj.passive_activation_passive
        transition_matrix[0, 0, 0] = 1 - node_obj.passive_activation_passive

        # Action Active (a=1)
        transition_matrix[0, 1, 1] = node_obj.passive_activation_active
        transition_matrix[0, 1, 0] = 1 - node_obj.passive_activation_active

        # From Active state (s=1)
        # Action Passive (a=0)
        transition_matrix[1, 0, 1] = node_obj.active_activation_passive
        transition_matrix[1, 0, 0] = 1 - node_obj.active_activation_passive

        # Action Active (a=1)
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

    # Outperformance counters
    outperformance_counts_nodes = defaultdict(int)
    outperformance_counts_values = defaultdict(int)

    # Initialize results storage
    individual_simulation_results = []

    # Algorithms we track: 'Hill Climb', 'DQN Normal', 'Whittle'
    all_active_nodes = {algo: [] for algo in ['Hill Climb', 'DQN Normal', 'Whittle']}
    all_node_values = {algo: [] for algo in ['Hill Climb', 'DQN Normal', 'Whittle']}

    for sim in range(1, num_simulations + 1):
        if sim % UPDATE_INTERVAL == 0 or sim == 1:
            print(f"Starting simulation {sim}/{num_simulations}...")
        final_active_nodes, final_node_values = run_single_simulation(initial_graph, num_actions, model_normal, whittle_policy, gamma)

        for algo in all_active_nodes:
            all_active_nodes[algo].append(final_active_nodes[algo])
            all_node_values[algo].append(final_node_values[algo])

        # Perform comparisons: DQN Normal vs Hill Climb and DQN Normal vs Whittle
        for timestep in range(NUM_TIMESTEPS):
            try:
                hc_active = final_active_nodes['Hill Climb'][timestep]
                hc_value = final_node_values['Hill Climb'][timestep]

                dqn_active = final_active_nodes['DQN Normal'][timestep]
                dqn_value = final_node_values['DQN Normal'][timestep]

                wh_active = final_active_nodes['Whittle'][timestep]
                wh_value = final_node_values['Whittle'][timestep]
            except IndexError:
                continue

            # DQN vs Hill Climb (Active nodes)
            if dqn_active > hc_active:
                outperformance_counts_nodes['DQN Normal vs Hill Climb'] += 1

            # DQN vs Hill Climb (Node values)
            if dqn_value > hc_value:
                outperformance_counts_values['DQN Normal vs Hill Climb'] += 1

            # DQN vs Whittle (Active nodes)
            if dqn_active > wh_active:
                outperformance_counts_nodes['DQN Normal vs Whittle'] += 1

            # DQN vs Whittle (Node values)
            if dqn_value > wh_value:
                outperformance_counts_values['DQN Normal vs Whittle'] += 1

        # Record final percent and values for each simulation
        final_percent = calculate_final_percent_activated(final_active_nodes, num_nodes, NUM_TIMESTEPS)
        final_val = calculate_final_node_values(final_node_values, num_timesteps=NUM_TIMESTEPS)
        simulation_result = {'Simulation': sim}
        for algo in ['Hill Climb', 'DQN Normal', 'Whittle']:
            simulation_result[f'{algo}_Percent'] = final_percent.get(algo, 0)
            simulation_result[f'{algo}_Value'] = final_val.get(algo, 0)
        individual_simulation_results.append(simulation_result)

    total_comparisons = num_simulations * NUM_TIMESTEPS
    outperformance_percentages_nodes = {key: (count / total_comparisons) * 100 for key, count in outperformance_counts_nodes.items()}
    outperformance_percentages_values = {key: (count / total_comparisons) * 100 for key, count in outperformance_counts_values.items()}

    # Performance metrics
    performance_metrics_nodes = analyze_performance(all_active_nodes)
    performance_metrics_values = analyze_performance_node_values(all_node_values)

    print(f"\n=== Completed {num_simulations} simulations for Graph Size {graph_size} ===")
    print("Outperformance Percentages (Active Nodes):")
    for key, percentage in outperformance_percentages_nodes.items():
        print(f"{key}: {percentage:.2f}%")
    
    print("\nOutperformance Percentages (Node Values):")
    for key, percentage in outperformance_percentages_values.items():
        print(f"{key}: {percentage:.2f}%")

    return {
        'outperformance_nodes': outperformance_percentages_nodes,
        'outperformance_values': outperformance_percentages_values
    }, individual_simulation_results, {
        'nodes': performance_metrics_nodes,
        'values': performance_metrics_values
    }

def determine_stop_percent(num_nodes):
    """
    Determines the stop_percent based on the number of nodes.
    """
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

def calculate_final_percent_activated(final_active_nodes, num_nodes, num_timesteps):
    """
    Calculates the final percent activated for each algorithm.
    """
    final_percent_activated = {}
    for algo, active_nodes_over_time in final_active_nodes.items():
        total_active = sum(active_nodes_over_time)
        max_possible = num_nodes * num_timesteps
        percent = (total_active / max_possible) * 100
        final_percent_activated[algo] = percent
    return final_percent_activated

def calculate_final_node_values(final_node_values, num_timesteps):
    """
    Calculates the final sum of node values activated for each algorithm.
    """
    final_node_sum = {}
    for algo, node_values_over_time in final_node_values.items():
        total_value = sum(node_values_over_time)
        final_node_sum[algo] = total_value
    return final_node_sum

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
        active_nodes_matrix = np.array(simulations)  # Shape: [num_simulations, num_timesteps]
        
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

def analyze_performance_node_values(all_node_values, thresholds=[25, 50, 75, 90]):
    """
    Analyzes performance metrics for node values across all simulations for a given graph size.
    
    Args:
        all_node_values (dict): Dictionary mapping algorithms to lists of node value sums over simulations.
        thresholds (list): Thresholds for node value sums as percentages.
    
    Returns:
        dict: Performance metrics for each algorithm.
    """
    performance_metrics = {}
    for algo, simulations in all_node_values.items():
        node_values_array = np.array(simulations)  # Shape: [num_simulations, num_timesteps]
        
        # Mean and Standard Deviation per timestep
        mean_values = node_values_array.mean(axis=0)  # Shape: [num_timesteps]
        std_values = node_values_array.std(axis=0)
        
        # AUAC per simulation, then average
        auac_per_simulation = np.trapz(node_values_array, axis=1)  # Shape: [num_simulations]
        mean_auac = auac_per_simulation.mean()
        
        # Time to reach thresholds
        time_to_threshold = {}
        for threshold in thresholds:
            # Define target per simulation
            target = (threshold / 100.0) * node_values_array.max(axis=1)
            times = []
            for sim in range(node_values_array.shape[0]):
                reached = np.where(node_values_array[sim] >= target[sim])[0]
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
            count = np.sum(node_values_array[:, -1] >= (threshold / 100.0 * node_values_array.max(axis=1)))
            prob = (count / node_values_array.shape[0]) * 100
            prob_reach_threshold[threshold] = prob
        
        # Coefficient of Variation (CV) for AUAC
        cv = auac_per_simulation.std() / auac_per_simulation.mean() if auac_per_simulation.mean() != 0 else np.nan
        
        # Store metrics
        performance_metrics[algo] = {
            'mean_value_per_timestep': mean_values,
            'std_value_per_timestep': std_values,
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

def create_directory(directory):
    """
    Creates a directory if it does not exist.
    
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_aggregated_results(overall_outperformance, performance_metrics, graph_sizes, output_dir='plots'):
    """
    Plots aggregated results and saves them to files.

    Args:
        overall_outperformance (dict): Aggregated outperformance data.
        performance_metrics (dict): Performance metrics data.
        graph_sizes (list): List of graph sizes.
        output_dir (str): Directory to save the plots.
    """
    # Ensure the output directory exists
    create_directory(output_dir)

    for graph_size in graph_sizes:
        num_nodes, num_edges = graph_size
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Aggregated Results for Graph Size {num_nodes} Nodes x {num_edges} Edges', fontsize=16)

        # Outperformance Percentages (Active Nodes)
        labels = list(overall_outperformance[graph_size]['outperformance_nodes'].keys())
        values = list(overall_outperformance[graph_size]['outperformance_nodes'].values())
        sns.barplot(x=labels, y=values, palette="viridis", ax=axes[0, 0])
        axes[0, 0].set_title('Outperformance Percentages (Active Nodes)')
        axes[0, 0].set_xlabel('Algorithm Comparisons')
        axes[0, 0].set_ylabel('Outperformance Percentage (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Outperformance Percentages (Node Values)
        labels_values = list(overall_outperformance[graph_size]['outperformance_values'].keys())
        values_values = list(overall_outperformance[graph_size]['outperformance_values'].values())
        sns.barplot(x=labels_values, y=values_values, palette="magma", ax=axes[0, 1])
        axes[0, 1].set_title('Outperformance Percentages (Node Values)')
        axes[0, 1].set_xlabel('Algorithm Comparisons')
        axes[0, 1].set_ylabel('Outperformance Percentage (%)')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Mean Activation Percentage Over Time
        for algo in ['Hill Climb', 'DQN Normal', 'Whittle']:
            mean_active = performance_metrics[graph_size]['nodes'][algo]['mean_active_per_timestep']
            mean_percent = (mean_active / num_nodes) * 100
            axes[1, 0].plot(range(1, NUM_TIMESTEPS + 1), mean_percent, label=algo)
        axes[1, 0].set_title('Mean Activation Percentage Over Time')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Mean Activation Percentage (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Mean Node Values Over Time
        for algo in ['Hill Climb', 'DQN Normal', 'Whittle']:
            mean_values = performance_metrics[graph_size]['values'][algo]['mean_value_per_timestep']
            axes[1, 1].plot(range(1, NUM_TIMESTEPS + 1), mean_values, label=algo)
        axes[1, 1].set_title('Mean Node Values Over Time')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Mean Node Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        filename = f'aggregated_results_{num_nodes}nodes_{num_edges}edges.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved aggregated plots for graph size {graph_size} to {filepath}")

def plot_individual_results(all_individual_results, output_dir='plots'):
    """
    Plots individual simulation results and saves them to files.

    Args:
        all_individual_results (dict): Individual simulation results.
        output_dir (str): Directory to save the plots.
    """
    create_directory(output_dir)

    for graph_size, results in all_individual_results.items():
        num_nodes, num_edges = graph_size
        df = pd.DataFrame(results)
        
        # Melt the DataFrame for seaborn (Percentage)
        df_percent = df.melt(id_vars=['Simulation'], 
                             value_vars=[f'{algo}_Percent' for algo in ['Hill Climb', 'DQN Normal', 'Whittle']],
                             var_name='Algorithm', value_name='Percent Activated')
        df_percent['Algorithm'] = df_percent['Algorithm'].str.replace('_Percent', '')
        
        # Melt the DataFrame for seaborn (Node Values)
        df_values = df.melt(id_vars=['Simulation'], 
                            value_vars=[f'{algo}_Value' for algo in ['Hill Climb', 'DQN Normal', 'Whittle']],
                            var_name='Algorithm', value_name='Node Values')
        df_values['Algorithm'] = df_values['Algorithm'].str.replace('_Value', '')

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Individual Simulation Results for Graph Size {num_nodes} Nodes x {num_edges} Edges', fontsize=16)

        # Plot Activation Percentages
        sns.boxplot(x='Algorithm', y='Percent Activated', data=df_percent, ax=axes[0])
        axes[0].set_title('Activation Percentages')
        axes[0].set_xlabel('Algorithm')
        axes[0].set_ylabel('Percent Activated')
        axes[0].tick_params(axis='x', rotation=45)

        # Plot Node Values
        sns.boxplot(x='Algorithm', y='Node Values', data=df_values, ax=axes[1])
        axes[1].set_title('Node Values Activated')
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Sum of Node Values Activated')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        filename = f'individual_results_{num_nodes}nodes_{num_edges}edges.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Saved individual simulation plots for graph size {graph_size} to {filepath}")

def main():
    # Define the different graph sizes to test
    graph_sizes = [
        # (50, 50),
        # (100, 100),
        # (150, 150),
        (200, 200)
    ]

    num_actions = 10  # Number of nodes to activate per timestep

    # Initialize data collection structures for all graph sizes
    overall_outperformance = {}  # {graph_size: {'outperformance_nodes': {...}, 'outperformance_values': {...}}, ...}
    all_individual_results = {}  # {graph_size: [simulation_results, ...], ...}
    all_performance_metrics = {}  # {graph_size: {'nodes': {...}, 'values': {...}}, ...}

    # Simulation parameters
    simulation_params = {
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

    plot_output_dir = 'plots_value_whittle'

    # Save individual simulation results to CSV files
    save_individual_results(all_individual_results)

    # Plot aggregated results and save to files
    plot_aggregated_results(overall_outperformance, all_performance_metrics, graph_sizes, output_dir=plot_output_dir)

    # Plot individual simulation results and save to files
    plot_individual_results(all_individual_results, output_dir=plot_output_dir)

if __name__ == "__main__":
    main()
