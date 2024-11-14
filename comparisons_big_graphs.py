# comparisons_big_graphs.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import torch
import random
import os

from tianshou.policy import BasePolicy
from tianshou.data import Batch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer

from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
from simpleNode import SimpleNode as Node
from hillClimb import HillClimb
from network_env import NetworkEnv
from policy_networks import PolicyNetworkAgent
from deepq import train_dqn_agent, QNet, CustomQPolicy


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
        (50, 75),
        (100, 150),
        (150, 225),
        (200, 300)
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
        'k': num_actions
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
        graph_policy_network = copy.deepcopy(G)
        graph_dqn_agent = copy.deepcopy(G)
        graph_random_selection = copy.deepcopy(G)
        graph_no_selection = copy.deepcopy(G)

        # Initialize agents
        # For Policy Network
        model_path_policy = f'ppo_policy_net_{num_nodes}.pth'
        model_path_value = f'ppo_value_net_{num_nodes}.pth'

        policy_agent = PolicyNetworkAgent(num_nodes, num_actions)
        env = NetworkEnv(graph_policy_network, k=num_actions)

        if os.path.exists(model_path_policy) and os.path.exists(model_path_value):
            # Load the trained models
            policy_agent.load_model(model_path_policy, model_path_value)
            print("Loaded trained Policy Network models.")
        else:
            # Train the models
            print("Training Policy Network models...")
            policy_agent.train(env, num_episodes=500)
            #NOTE: models are not saved for now
            #policy_agent.save_model(model_path_policy, model_path_value)
        
        
        #DeepQ
        config = {
            "graph": graph_dqn_agent,
            "num_nodes": num_nodes,
            "cascade_prob": CASCADE_PROB,
            "stop_percent": 0.8  # Adjust as needed
        }
        print("Training DQN agent...")
        model, policy = train_dqn_agent(config, num_actions)

        # Initialize data collection structures
        algorithms = ['Hill Climb', 'Policy Network', 'DQN Agent', 'Random Selection', 'No Selection']
        data_collection = {algo: {
            'timestep': [],
            'cumulative_active_nodes': [],
            'percent_activated': [],
            'activation_efficiency': [],
        } for algo in algorithms}

        cumulative_seeds_used = {algo: 0 for algo in algorithms}

        # Set the number of timesteps for the simulation
        num_timesteps = 50
        update_interval = 5  # Update the table every 5 timesteps

        # Activation thresholds to check (in percentage)
        activation_thresholds_values = [25, 50, 75, 90]

        for timestep in range(1, num_timesteps + 1):
            # ----------------- HILL CLIMB -----------------
            # Select action using Hill Climb
            seeded_nodes_hc = HillClimb.hill_climb(graph_hill_climb, num=num_actions)
            # Apply actions and state transitions
            transition_nodes_hc = ns.passive_state_transition_without_neighbors(graph_hill_climb, exempt_nodes=seeded_nodes_hc)
            changed_nodes_hc = ns.active_state_transition([node for node in seeded_nodes_hc])
            newly_activated_hc = ns.independent_cascade_allNodes(graph_hill_climb, CASCADE_PROB)
            ns.rearm_nodes(graph_hill_climb)
            # Collect data
            collect_data('Hill Climb', graph_hill_climb, seeded_nodes_hc, changed_nodes_hc, newly_activated_hc,
                         cumulative_seeds_used, data_collection, timestep)

            # ----------------- POLICY NETWORK -----------------
            # Select action using Policy Network
            state = np.array([int(graph_policy_network.nodes[i]['obj'].isActive()) for i in graph_policy_network.nodes()], dtype=np.float32)
            action, logits_np = policy_agent.select_action(state)
            topk_indices = np.nonzero(action)[0]

            # Apply actions and state transitions
            exempt_nodes = [graph_policy_network.nodes[node_index]['obj'] for node_index in topk_indices]
            transition_nodes_pn = ns.passive_state_transition_without_neighbors(graph_policy_network, exempt_nodes=exempt_nodes)
            changed_nodes_pn = ns.active_state_transition_graph_indices(graph_policy_network, topk_indices)
            newly_activated_pn = ns.independent_cascade_allNodes(graph_policy_network, CASCADE_PROB)
            ns.rearm_nodes(graph_policy_network)

            # Collect data
            collect_data('Policy Network', graph_policy_network, [graph_policy_network.nodes[node_index]['obj'] for node_index in topk_indices],
                        changed_nodes_pn, newly_activated_pn, cumulative_seeds_used, data_collection, timestep)

            # ----------------- DQN AGENT -----------------
            # Select action using DQN agent
            state = np.array([int(graph_dqn_agent.nodes[i]['obj'].isActive()) for i in graph_dqn_agent.nodes()], dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            device = state_tensor.device

            # Generate all possible actions
            actions = torch.eye(num_nodes, device=device)
            states = state_tensor.repeat(num_nodes, 1)
            actions_tensor = actions

            # Compute Q-values
            with torch.no_grad():
                q_values = model(states, actions_tensor).squeeze()

            # Mask already active nodes
            active_nodes = [i for i, node in enumerate(graph_dqn_agent.nodes()) if graph_dqn_agent.nodes[node]['obj'].isActive()]
            q_values_np = q_values.cpu().numpy()
            q_values_np[active_nodes] = -np.inf  # Assign negative infinity to active nodes

            # Select top-k actions
            topk_indices = np.argsort(q_values_np)[-num_actions:][::-1]

            # Apply actions and state transitions
            exempt_nodes = [graph_dqn_agent.nodes[node_index]['obj'] for node_index in topk_indices]
            ns.passive_state_transition_without_neighbors(graph_dqn_agent, exempt_nodes=exempt_nodes)
            ns.active_state_transition_graph_indices(graph_dqn_agent, topk_indices)
            ns.independent_cascade_allNodes(graph_dqn_agent, CASCADE_PROB)
            ns.rearm_nodes(graph_dqn_agent)

            # Collect data
            collect_data('DQN Agent', graph_dqn_agent, exempt_nodes, [], [], cumulative_seeds_used, data_collection, timestep)

            # ----------------- RANDOM SELECTION -----------------
            # Select action by randomly choosing nodes
            random_nodes_indices = random.sample(range(len(graph_random_selection.nodes())), num_actions)
            seeded_nodes_random = [graph_random_selection.nodes[node_index]['obj'] for node_index in random_nodes_indices]
            # Apply actions and state transitions
            exempt_nodes = seeded_nodes_random
            transition_nodes_random = ns.passive_state_transition_without_neighbors(graph_random_selection, exempt_nodes=exempt_nodes)
            changed_nodes_random = ns.active_state_transition(seeded_nodes_random)
            newly_activated_random = ns.independent_cascade_allNodes(graph_random_selection, CASCADE_PROB)
            ns.rearm_nodes(graph_random_selection)
            # Collect data
            collect_data('Random Selection', graph_random_selection, seeded_nodes_random, changed_nodes_random,
                         newly_activated_random, cumulative_seeds_used, data_collection, timestep)

            # ----------------- NO SELECTION -----------------
            # No action is taken
            seeded_nodes_none = []
            # Apply actions and state transitions
            transition_nodes_none = ns.passive_state_transition_without_neighbors(graph_no_selection, exempt_nodes=[])
            # Since no nodes are activated, changed_nodes_none is empty
            changed_nodes_none = []
            newly_activated_none = ns.independent_cascade_allNodes(graph_no_selection, CASCADE_PROB)
            ns.rearm_nodes(graph_no_selection)
            # Collect data
            collect_data('No Selection', graph_no_selection, seeded_nodes_none, changed_nodes_none, newly_activated_none,
                         cumulative_seeds_used, data_collection, timestep)

            # ----------------- PRINT RESULTS -----------------
            # Every update_interval timesteps, print the data
            if timestep % update_interval == 0 or timestep == num_timesteps:
                # Calculate AUAC and activation thresholds
                auac = calculate_auac(data_collection)
                activation_thresholds = calculate_activation_thresholds(data_collection, activation_thresholds_values)
                print_results(data_collection, cumulative_seeds_used, auac, activation_thresholds, timestep)

        # Store data for this graph size
        overall_data[(num_nodes, num_edges)] = data_collection

    # After all graph sizes are processed, create combined plots
    plot_combined_results(overall_data, simulation_params)

def collect_data(algorithm, graph, seeded_nodes, changed_nodes, newly_activated_nodes, cumulative_seeds_used,
                 data_collection, timestep):
    # Calculate metrics
    total_nodes = len(graph.nodes())
    active_nodes = sum(1 for node in graph.nodes() if graph.nodes[node]['obj'].isActive())
    # For cumulative activated nodes, we can use active_nodes because it represents all currently active nodes
    cumulative_active_nodes = active_nodes

    # Update cumulative seeds used
    cumulative_seeds_used[algorithm] += len(seeded_nodes)

    # Calculate percentage of network activated
    percent_activated = (cumulative_active_nodes / total_nodes) * 100

    # Activation Efficiency
    if cumulative_seeds_used[algorithm] > 0:
        activation_efficiency = cumulative_active_nodes / cumulative_seeds_used[algorithm]
    else:
        activation_efficiency = 0

    # Record data
    data_collection[algorithm]['timestep'].append(timestep)
    data_collection[algorithm]['cumulative_active_nodes'].append(cumulative_active_nodes)
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
        active_nodes = data_collection[algo]['cumulative_active_nodes']
        auac[algo] = np.trapz(active_nodes, timesteps)
    return auac

def print_results(data_collection, cumulative_seeds_used, auac, activation_thresholds, timestep):
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

def plot_combined_results(overall_data, simulation_params):
    import matplotlib.pyplot as plt

    # Extract parameters
    passive_activation_chance = simulation_params['passive_activation_chance']
    passive_deactivation_chance = simulation_params['passive_deactivation_chance']
    active_activation_chance = simulation_params['active_activation_chance']
    active_deactivation_chance = simulation_params['active_deactivation_chance']
    cascade_prob = simulation_params['cascade_prob']
    k = simulation_params['k']

    # Define metrics to plot
    metrics = ['percent_activated', 'activation_efficiency']
    metric_names = {
        'percent_activated': 'Percentage of Network Activated (%)',
        'activation_efficiency': 'Activation Efficiency'
    }

    for metric in metrics:
        # Create a figure with subplots for each graph size
        num_graphs = len(overall_data)
        cols = 2
        rows = (num_graphs + 1) // cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for idx, ((num_nodes, num_edges), data_collection) in enumerate(overall_data.items()):
            ax = axes[idx]
            timesteps = data_collection['Hill Climb']['timestep']
            for algo in data_collection:
                ax.plot(timesteps, data_collection[algo][metric], label=algo)
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
            # Place a text box in upper left in axes coords
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

            ax.legend()

        # Hide any unused subplots
        for idx in range(len(overall_data), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(metric_names[metric] + ' Across Different Graph Sizes', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle
        plt.show()


if __name__ == "__main__":
    main()
