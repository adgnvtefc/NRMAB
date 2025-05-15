# comparisons_big_graphs.py

import numpy as np
import copy
import random

from networkSim import NetworkSim as ns
from algorithms.hillClimb import HillClimb
from algorithms.deepq import train_dqn_agent, select_action_dqn
from algorithms.OLD_doubleq import train_double_dqn_agent, select_action_double_dqn 

# Activation chance in passive action
PASSIVE_ACTIVATION_CHANCE = 0.05
# Deactivation chance in passive action
PASSIVE_DEACTIVATION_CHANCE = 0.3

ACTIVE_ACTIVATION_CHANCE = 0.95
ACTIVE_DEACTIVATION_CHANCE = 0.05

CASCADE_PROB = 0.05

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
    overall_data = {}

    simulation_params = {
        'passive_activation_chance': PASSIVE_ACTIVATION_CHANCE,
        'passive_deactivation_chance': PASSIVE_DEACTIVATION_CHANCE,
        'active_activation_chance': ACTIVE_ACTIVATION_CHANCE,
        'active_deactivation_chance': ACTIVE_DEACTIVATION_CHANCE,
        'cascade_prob': CASCADE_PROB,
        'k': num_actions,
        'discount_factor': 0.99  # Discount factor gamma set to 0.99
    }

    for num_nodes, num_edges in graph_sizes:
        print(f"\n=== Running simulations for Graph with {num_nodes} nodes and {num_edges} edges ===\n")
        # Initialize the graph
        G = ns.init_random_graph(
            num_nodes,
            num_edges,
            PASSIVE_ACTIVATION_CHANCE,
            PASSIVE_DEACTIVATION_CHANCE,
            ACTIVE_ACTIVATION_CHANCE,
            ACTIVE_DEACTIVATION_CHANCE
        )

        # Copy the graph for each algorithm to ensure they start from the same initial state
        graph_hill_climb = copy.deepcopy(G)
        graph_dqn_normal = copy.deepcopy(G)
        graph_double_dqn = copy.deepcopy(G)  # Graph for Double DQN
        graph_random_selection = copy.deepcopy(G)
        graph_no_selection = copy.deepcopy(G)

        # ----------------- DQN AGENT WITH NORMAL REWARD FUNCTION -----------------
        # Set stop_percent based on the number of nodes
        if num_nodes == 50:
            stop_percent = 0.8
        elif num_nodes == 100:
            stop_percent = 0.5
        elif num_nodes == 150:
            stop_percent = 0.3
        elif num_nodes == 200:
            stop_percent = 0.25
        else:
            stop_percent = 0.5  # Default value if not specified

        config_normal = {
            "graph": graph_dqn_normal,
            "num_nodes": num_nodes,
            "cascade_prob": CASCADE_PROB,
            "stop_percent": stop_percent,
            "reward_function": "normal"
        }
        print("Training DQN agent with normal reward function...")
        model_normal, policy_normal = train_dqn_agent(config_normal, num_actions)

        # ----------------- DOUBLE DQN AGENT -----------------
        config_double_dqn = {
            "graph": graph_double_dqn,
            "num_nodes": num_nodes,
            "cascade_prob": CASCADE_PROB,
            "stop_percent": stop_percent,
            "reward_function": "normal"
        }
        print("Training Double DQN agent...")
        model_double_dqn, policy_double_dqn = train_double_dqn_agent(config_double_dqn, num_actions)

        # Initialize data collection structures
        algorithms = ['Hill Climb', 'DQN Normal', 'Double DQN', 'Random Selection', 'No Selection']
        data_collection = {algo: {
            'timestep': [],
            'cumulative_active_nodes': [],
            'discounted_activation': [],
            'percent_activated': [], 
            'activation_efficiency': [],
        } for algo in algorithms}

        gamma = simulation_params['discount_factor']
        cumulative_seeds_used = {algo: 0 for algo in algorithms}
        cumulative_active_nodes_prev = {algo: 0 for algo in algorithms}
        discounted_activation_prev = {algo: 0 for algo in algorithms}

        # Initialize outperformance_counts for this graph size
        outperformance_counts = {
            'DQN Normal vs Hill Climb': 0,
            'DQN Normal vs Random Selection': 0,
            'DQN Normal vs No Selection': 0,
            'Double DQN vs Hill Climb': 0,
            'Double DQN vs Random Selection': 0,
            'Double DQN vs No Selection': 0,
        }

        # Set the number of timesteps for the simulation
        num_timesteps = 50
        update_interval = 5  # Update the table every 5 timesteps
        total_timesteps = num_timesteps

        # Activation thresholds to check (in percentage)
        activation_thresholds_values = [25, 50, 75, 90]

        for timestep in range(1, num_timesteps + 1):
            # ----------------- HILL CLIMB -----------------
            # Select action using Hill Climb
            seeded_nodes_hc = HillClimb.hill_climb(graph_hill_climb, num=num_actions)
            # Apply actions and state transitions
            ns.passive_state_transition_without_neighbors(graph_hill_climb, exempt_nodes=seeded_nodes_hc)
            ns.active_state_transition([node for node in seeded_nodes_hc])
            ns.independent_cascade_allNodes(graph_hill_climb, CASCADE_PROB)
            ns.rearm_nodes(graph_hill_climb)
            # Collect data
            collect_data('Hill Climb', graph_hill_climb, seeded_nodes_hc,
                         cumulative_seeds_used, data_collection, timestep, gamma,
                         cumulative_active_nodes_prev, discounted_activation_prev)

            # ----------------- DQN AGENT WITH NORMAL REWARD FUNCTION -----------------
            # Select action using DQN agent trained with normal reward function
            seeded_nodes_dqn_normal = select_action_dqn(graph_dqn_normal, model_normal, num_actions)
            # Apply actions and state transitions
            exempt_nodes = seeded_nodes_dqn_normal
            ns.passive_state_transition_without_neighbors(graph_dqn_normal, exempt_nodes=exempt_nodes)
            ns.active_state_transition(seeded_nodes_dqn_normal)
            ns.independent_cascade_allNodes(graph_dqn_normal, CASCADE_PROB)
            ns.rearm_nodes(graph_dqn_normal)
            # Collect data
            collect_data('DQN Normal', graph_dqn_normal, seeded_nodes_dqn_normal,
                         cumulative_seeds_used, data_collection, timestep, gamma,
                         cumulative_active_nodes_prev, discounted_activation_prev)

            # ----------------- DOUBLE DQN AGENT -----------------
            # Select action using Double DQN agent
            seeded_nodes_double_dqn = select_action_double_dqn(graph_double_dqn, model_double_dqn, num_actions)
            # Apply actions and state transitions
            exempt_nodes = seeded_nodes_double_dqn
            ns.passive_state_transition_without_neighbors(graph_double_dqn, exempt_nodes=exempt_nodes)
            ns.active_state_transition(seeded_nodes_double_dqn)
            ns.independent_cascade_allNodes(graph_double_dqn, CASCADE_PROB)
            ns.rearm_nodes(graph_double_dqn)
            # Collect data
            collect_data('Double DQN', graph_double_dqn, seeded_nodes_double_dqn,
                         cumulative_seeds_used, data_collection, timestep, gamma,
                         cumulative_active_nodes_prev, discounted_activation_prev)

            # ----------------- RANDOM SELECTION -----------------
            # Select action by randomly choosing nodes
            random_nodes_indices = random.sample(range(len(graph_random_selection.nodes())), num_actions)
            seeded_nodes_random = [graph_random_selection.nodes[node_index]['obj'] for node_index in random_nodes_indices]
            # Apply actions and state transitions
            exempt_nodes = seeded_nodes_random
            ns.passive_state_transition_without_neighbors(graph_random_selection, exempt_nodes=exempt_nodes)
            ns.active_state_transition(seeded_nodes_random)
            ns.independent_cascade_allNodes(graph_random_selection, CASCADE_PROB)
            ns.rearm_nodes(graph_random_selection)
            # Collect data
            collect_data('Random Selection', graph_random_selection, seeded_nodes_random,
                         cumulative_seeds_used, data_collection, timestep, gamma,
                         cumulative_active_nodes_prev, discounted_activation_prev)

            # ----------------- NO SELECTION -----------------
            # No action is taken
            seeded_nodes_none = []
            # Apply actions and state transitions
            ns.passive_state_transition_without_neighbors(graph_no_selection, exempt_nodes=[])
            ns.independent_cascade_allNodes(graph_no_selection, CASCADE_PROB)
            ns.rearm_nodes(graph_no_selection)
            # Collect data
            collect_data('No Selection', graph_no_selection, seeded_nodes_none,
                         cumulative_seeds_used, data_collection, timestep, gamma,
                         cumulative_active_nodes_prev, discounted_activation_prev)

            # ---------------- FIND DQN OUTPERFORMANCE ---------------
            # For both DQN agents
            for dqn_algo in ['DQN Normal', 'Double DQN']:
                dqn_percent_activated = data_collection[dqn_algo]['percent_activated'][-1]

                # Compare with other algorithms
                for other_algo in ['Hill Climb', 'Random Selection', 'No Selection']:
                    other_percent_activated = data_collection[other_algo]['percent_activated'][-1]
                    if dqn_percent_activated > other_percent_activated:
                        key = f'{dqn_algo} vs {other_algo}'
                        outperformance_counts[key] += 1

            # ----------------- PRINT RESULTS -----------------
            # Every update_interval timesteps, print the data
            if timestep % update_interval == 0 or timestep == num_timesteps:
                # Calculate AUAC and activation thresholds
                auac = calculate_auac(data_collection)
                activation_thresholds = calculate_activation_thresholds(data_collection, activation_thresholds_values)
                print_results(data_collection, cumulative_seeds_used, auac, activation_thresholds, timestep, outperformance_counts)

        # After simulation for this graph size is complete
        # Calculate percentages
        outperformance_percentages = {key: (count / total_timesteps) * 100 for key, count in outperformance_counts.items()}
        # Store data for this graph size
        overall_data[(num_nodes, num_edges)] = {
            'data_collection': data_collection,
            'outperformance_percentages': outperformance_percentages
        }

    # After all graph sizes are processed, create combined plots
    plot_combined_results(overall_data, simulation_params)

def collect_data(algorithm, graph, seeded_nodes, cumulative_seeds_used,
                 data_collection, timestep, gamma,
                 cumulative_active_nodes_prev, discounted_activation_prev):
    # Calculate metrics
    total_nodes = len(graph.nodes())
    active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())

    # Update cumulative active nodes (sum over time)
    cumulative_active_nodes = cumulative_active_nodes_prev[algorithm] + active_nodes
    cumulative_active_nodes_prev[algorithm] = cumulative_active_nodes

    # Update cumulative seeds used
    cumulative_seeds_used[algorithm] += len(seeded_nodes)

    # Calculate percentage of network activated (current timestep)
    percent_activated = (active_nodes / total_nodes) * 100

    # Activation Efficiency
    if cumulative_seeds_used[algorithm] > 0:
        activation_efficiency = active_nodes / cumulative_seeds_used[algorithm]
    else:
        activation_efficiency = 0

    # Discounted activation
    discounted_activation = discounted_activation_prev[algorithm] * gamma + active_nodes
    discounted_activation_prev[algorithm] = discounted_activation

    # Record data
    data_collection[algorithm]['timestep'].append(timestep)
    data_collection[algorithm]['cumulative_active_nodes'].append(cumulative_active_nodes)
    data_collection[algorithm]['discounted_activation'].append(discounted_activation)
    data_collection[algorithm]['percent_activated'].append(percent_activated)
    data_collection[algorithm]['activation_efficiency'].append(activation_efficiency)

def calculate_activation_thresholds(data_collection, activation_thresholds):
    thresholds = {algo: {} for algo in data_collection.keys()}
    for algo in data_collection.keys():
        for threshold in activation_thresholds:
            reached = False
            for t, percent in zip(data_collection[algo]['timestep'], data_collection[algo]['percent_activated']):
                if percent >= threshold:
                    thresholds[algo][threshold] = t
                    reached = True
                    break
            if not reached:
                thresholds[algo][threshold] = 'N/A'  # Did not reach threshold
    return thresholds

def calculate_auac(data_collection):
    auac = {}
    for algo in data_collection.keys():
        timesteps = data_collection[algo]['timestep']
        percent_activated = data_collection[algo]['percent_activated']
        auac[algo] = np.trapz(percent_activated, timesteps)
    return auac

def print_results(data_collection, cumulative_seeds_used, auac, activation_thresholds, timestep, outperformance_counts):
    print(f"\n=== Simulation Results at Timestep {timestep} ===")
    header = "{:<20} {:<10} {:<20} {:<20} {:<25} {:<25}".format(
        'Algorithm', 'Timestep', '% Activated', 'AUAC',
        'Activation Efficiency', 'Time to 50% Activation'
    )
    print(header)
    print("=" * len(header))
    for algo in data_collection.keys():
        percent_activated = data_collection[algo]['percent_activated'][-1]
        efficiency = data_collection[algo]['activation_efficiency'][-1]
        time_to_50 = activation_thresholds[algo].get(50, 'N/A')
        print("{:<20} {:<10} {:<20} {:<20} {:<25} {:<25}".format(
            algo,
            timestep,
            f"{percent_activated:.2f}%",
            f"{auac[algo]:.2f}",
            f"{efficiency:.2f}",
            time_to_50
        ))
    print("=" * len(header) + "\n")
    print("\nDQN Outperformance Percentages:")
    for key, count in outperformance_counts.items():
        percentage = (count / timestep) * 100  # Use 'timestep' instead of 'total_timesteps'
        print(f"{key}: {percentage:.2f}%")
    print("=" * len(header) + "\n")

def plot_combined_results(overall_data, simulation_params):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    # Extract parameters
    passive_activation_chance = simulation_params['passive_activation_chance']
    passive_deactivation_chance = simulation_params['passive_deactivation_chance']
    active_activation_chance = simulation_params['active_activation_chance']
    active_deactivation_chance = simulation_params['active_deactivation_chance']
    cascade_prob = simulation_params['cascade_prob']
    k = simulation_params['k']
    gamma = simulation_params['discount_factor']

    # Define metrics to plot
    metrics = ['percent_activated', 'cumulative_active_nodes', 'discounted_activation']
    metric_names = {
        'percent_activated': 'Percent Activated',
        'cumulative_active_nodes': 'Cumulative Activation',
        'discounted_activation': f'Discounted (Gamma={gamma})'
    }

    # Create a figure with subplots for each graph size
    num_graphs = len(overall_data)
    cols = 2
    rows = (num_graphs + 1) // cols  # Calculate the number of rows needed
    fig_height = 5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, fig_height))
    axes = axes.flatten()

    # Initial metric to display
    metric = metrics[0]

    # Store lines for updating
    lines_dict = {}  # keys: ax, values: list of line objects for each algorithm

    for idx, ((num_nodes, num_edges), data) in enumerate(overall_data.items()):
        data_collection = data['data_collection']
        ax = axes[idx]
        timesteps = data_collection['Hill Climb']['timestep']
        lines = []
        for algo in data_collection:
            line, = ax.plot(timesteps, data_collection[algo][metric], label=algo)
            lines.append(line)
        lines_dict[ax] = lines  # Store lines for updating later
        ax.set_xlabel('Timestep')
        ax.set_ylabel(metric_names[metric])
        ax.set_title(f'Graph: {num_nodes} nodes, {num_edges} edges')

        # Add simulation parameters as text box in the plot
        textstr = '\n'.join((
            f'Passive Activation Chance: {passive_activation_chance}',
            f'Passive Deactivation Chance: {passive_deactivation_chance}',
            f'Active Activation Chance: {active_activation_chance}',
            f'Active Deactivation Chance: {active_deactivation_chance}',
            f'Cascade Probability: {cascade_prob}',
            f'k (Nodes Activated per Timestep): {k}'
        ))
        # Adjust the margins based on text length
        max_line_length = max(len(line) for line in textstr.split('\n'))
        bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=bbox_props)

        ax.legend()

    # Hide any unused subplots
    for idx in range(len(overall_data), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust the layout to make room for the widgets
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.05)

    # Add a metric selector
    ax_metric = plt.axes([0.05, 0.6, 0.12, 0.15])  # x, y, width, height
    metric_labels = [metric_names[m] for m in metrics]
    radio_metric = RadioButtons(ax_metric, metric_labels)

    # Update function for metrics
    def update_metric(label):
        # Find the metric corresponding to the label
        for m_key, m_label in metric_names.items():
            if m_label == label:
                selected_metric = m_key
                break
        else:
            return  # Metric not found, do nothing

        # Update all subplots
        for idx, ((num_nodes, num_edges), data) in enumerate(overall_data.items()):
            data_collection = data['data_collection']
            ax = axes[idx]
            timesteps = data_collection['Hill Climb']['timestep']
            lines = lines_dict[ax]
            for line, algo in zip(lines, data_collection):
                ydata = data_collection[algo][selected_metric]
                line.set_ydata(ydata)
            ax.relim()
            ax.autoscale_view()
            ax.set_ylabel(metric_names[selected_metric])
        fig.suptitle(metric_names[selected_metric] + ' Across Different Graph Sizes', fontsize=16)
        plt.draw()

    radio_metric.on_clicked(update_metric)

    # Plot outperformance percentages
    # Create a new figure for outperformance percentages
    fig2, axes2 = plt.subplots(2, 3, figsize=(24, 12))  # Adjusted for more comparisons
    axes2 = axes2.flatten()
    outperformance_keys = [
        'DQN Normal vs Hill Climb', 'DQN Normal vs Random Selection', 'DQN Normal vs No Selection',
        'Double DQN vs Hill Climb', 'Double DQN vs Random Selection', 'Double DQN vs No Selection'
    ]

    for idx, key in enumerate(outperformance_keys):
        ax = axes2[idx]
        graph_labels = []
        percentages = []
        for (num_nodes, num_edges), data in overall_data.items():
            outperformance_percentages = data['outperformance_percentages']
            percentage = outperformance_percentages.get(key, 0)  # Default to 0 if key not found
            graph_labels.append(f"{num_nodes} nodes")
            percentages.append(percentage)
        ax.bar(graph_labels, percentages, color='skyblue')
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Outperformance Percentage (%)')
        ax.set_title(f'{key}')
        for i, v in enumerate(percentages):
            ax.text(i, v + 1, f"{v:.2f}%", color='blue', ha='center')
        ax.set_ylim(0, 100)  # Assuming percentages range from 0 to 100

    # Hide any unused subplots
    for idx in range(len(outperformance_keys), len(axes2)):
        fig2.delaxes(axes2[idx])

    # Adjust layout to prevent overlap
    fig2.tight_layout()

    # Initial title
    fig.suptitle(metric_names[metric] + ' Across Different Graph Sizes', fontsize=16)

    plt.show()

if __name__ == "__main__":
    main()
