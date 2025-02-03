# dqn_simulation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from deepq import train_dqn_agent, select_action_dqn  # Ensure main.py is accessible
from networkSim import NetworkSim as ns  # Ensure networkSim.py is accessible


CASCADE_PROB = 0.1

# Number of actions (nodes to activate) per timestep
NUM_ACTIONS = 2

# Simulation Control
NUM_TIMESTEPS = 15  # Maximum number of timesteps

# Initialize the reverse mapping after the graph has been created
def build_node_obj_to_id_mapping(G):
    return {data['obj']: node_id for node_id, data in G.nodes(data=True)}

def load_or_train_dqn_agent(config_dqn, num_actions, num_epochs=3):
    """
    Loads a pre-trained DQN model if available; otherwise, trains a new model.
    
    Args:
        config_dqn (dict): Configuration dictionary for the DQN agent.
        num_actions (int): Number of actions the DQN agent can take.
        model_path (str): File path to load/save the DQN model.
        num_epochs (int): Number of training epochs if training is required.
    
    Returns:
        model (nn.Module): Trained DQN model.
        policy (BasePolicy): Trained DQN policy.
    """

    print("Pre-trained DQN model not found. Training a new model...")
    model, policy = train_dqn_agent(config_dqn, num_actions, num_epochs=num_epochs)
    return model, policy

def visualize_graph(G, pos, node_color_map, edge_color_map, timestep):
    """
    Visualizes the network graph.
    
    Args:
        G (networkx.Graph): The network graph.
        pos (dict): Positions of nodes for visualization.
        node_color_map (list): List of colors for nodes.
        edge_color_map (list): List of colors for edges.
        timestep (int): Current timestep.
    """
    plt.clf()  # Clear the previous plot
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, edge_color=edge_color_map,
            node_size=800, font_color='white', font_weight='bold')
    plt.title(f"Network Simulation - Timestep {timestep}")
    plt.show(block=False)
    plt.pause(0.5)  # Pause to update the plot

def main():
    # Initialize the initial graph
    num_nodes = 30
    num_edges = 50
    initial_graph = ns.init_random_graph(
        num_nodes,
        num_edges,
        value_low=1,
        value_high=10
    )
    
    # Build the reverse mapping
    node_obj_to_id = build_node_obj_to_id_mapping(initial_graph)
    
    # Set up visualization
    pos = nx.spring_layout(initial_graph)  # Positioning of nodes
    
    # Check if the graph has node attribute 'obj' with method isActive()
    # If not, initialize node attributes
    for node in initial_graph.nodes():
        if 'obj' not in initial_graph.nodes[node]:
            # Assuming nodes are initially inactive
            initial_graph.nodes[node]['obj'] = ns.Node()  # Replace with actual node object initialization
        # Initialize node states if not already set
        if not hasattr(initial_graph.nodes[node]['obj'], 'active'):
            initial_graph.nodes[node]['obj'].active = False
    
    # Determine stop_percent based on number of nodes
    stop_percent = determine_stop_percent(num_nodes)
    
    # Configuration for DQN agent
    config_dqn = {
        "graph": copy.deepcopy(initial_graph),
        "num_nodes": num_nodes,
        "cascade_prob": CASCADE_PROB,
        "stop_percent": stop_percent,
        "reward_function": "normal"
    }
    
    # Load or train the DQN agent
    model_dqn, policy_dqn = load_or_train_dqn_agent(config_dqn, NUM_ACTIONS)
    
    # Initialize simulation variables
    G = copy.deepcopy(initial_graph)
    node_obj_to_id = build_node_obj_to_id_mapping(G)
    timestep = 0
    max_timesteps = NUM_TIMESTEPS
    
    # Initialize visualization
    node_color_map = ns.color_nodes(G)  # Function to color nodes based on their state
    edge_color_map = ns.color_edges(G)  # Function to color edges if needed
    
    visualize_graph(G, pos, node_color_map, edge_color_map, timestep)
    
    # Start simulation loop
    while timestep < max_timesteps:
        print(f"\n--- Timestep {timestep} ---")
        
        # Extract current state
        state = np.array([int(G.nodes[node]['obj'].isActive()) for node in G.nodes()], dtype=np.float32)
        
        # Select actions using the DQN agent
        seeded_nodes = select_action_dqn(G, model_dqn, NUM_ACTIONS)
        
        print(f"Seeded Nodes: {[node_obj_to_id[node] for node in seeded_nodes]}")
        
        # Apply state transitions based on selected actions
        changed_nodes_2 = ns.passive_state_transition_without_neighbors(G, exempt_nodes=seeded_nodes)
        changed_nodes = ns.active_state_transition(seeded_nodes)
        
        # Log node activations
        for node in seeded_nodes:
            node_id = node_obj_to_id.get(node)
            print(f"Node {node_id} is activated.")
        
        # Handle other node transitions
        # Assuming transition_nodes are handled within state transitions
        transition_nodes = changed_nodes  # Placeholder for actual function
        for node in transition_nodes:
            node_id = node_obj_to_id.get(node)
            if node_id is not None:
                if node.isActive():
                    print(f"Node {node_id} transitioned to active.")
                else:
                    print(f"Node {node_id} transitioned to inactive.")
            else:
                print("Node object not found in the graph.")
        transition_nodes = changed_nodes_2  # Placeholder for actual function
        for node in transition_nodes:
            node_id = node_obj_to_id.get(node)
            if node_id is not None:
                if node.isActive():
                    print(f"Node {node_id} transitioned to active.")
                else:
                    print(f"Node {node_id} transitioned to inactive.")
            else:
                print("Node object not found in the graph.")
        
        
        # Perform independent cascade
        newlyActivated = ns.independent_cascade_allNodes(G, CASCADE_PROB)
        print(f"Cascade Activated: {newlyActivated}")
        
        # Rearm nodes if necessary
        ns.rearm_nodes(G)
        
        # Update visualization
        node_color_map = ns.color_nodes(G)
        edge_color_map = ns.color_edges(G)
        visualize_graph(G, pos, node_color_map, edge_color_map, timestep)
        
        timestep += 1
        
        # Wait for user input to proceed to the next timestep
        input("Press Enter to proceed to the next timestep...")
    
    print("\n=== Simulation Completed ===")

def determine_stop_percent(num_nodes):
    """
    Determines the stop_percent based on the number of nodes.
    """
    if num_nodes == 30:
        return 0.2
    elif num_nodes == 100:
        return 0.5
    elif num_nodes == 150:
        return 0.3
    elif num_nodes == 200:
        return 0.25
    else:
        return 0.5  # Default value if not specified

if __name__ == "__main__":
    main()
