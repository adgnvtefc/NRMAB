import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import heapq

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns
from algorithms.whittle import WhittleIndexPolicy

def main():
    """
    Run the Whittle Index simulation and visualization.
    """
    # Whittle Index parameters
    WHITTLE_THRESHOLD = 1e-4  # Convergence threshold for binary search
    VALUE_ITERATION_THRESHOLD = 1e-2  # Threshold for value iteration
    DISCOUNT = 0.99  # Discount factor
    NUM_ACTIONS = 2  # Number of nodes to activate per timestep (k)
    CASCADE_PROB = 0.05  # Probability of cascade per edge

    # Graph parameters
    NUM_NODES = 50
    NUM_EDGES = 50
    VALUE_LOW = 10
    VALUE_HIGH = 100
    RANDOM_SEED = 42  # For reproducibility

    print("Initializing the network graph...")
    G = ns.init_random_graph(
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        value_low=VALUE_LOW,
        value_high=VALUE_HIGH
    )

    print("Defining transition matrices and node values for Whittle Index computation...")


    transitions_whittle = {}
    node_values = {}
    for node_id in G.nodes():
        node_obj = G.nodes[node_id]['obj']
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

    print("Initializing WhittleIndexPolicy...")
    whittle_policy = WhittleIndexPolicy(
        transitions=transitions_whittle,
        node_values=node_values,
        discount=DISCOUNT,
        subsidy_break=0.0,
        eps=WHITTLE_THRESHOLD
    )

    current_states = {node_id: 0 for node_id in G.nodes()}  # Start inactive
    pos = nx.spring_layout(G, seed=RANDOM_SEED)

    MAX_TIMESTEPS = 10
    timestep = 0

    print("Starting simulation...")
    while timestep < MAX_TIMESTEPS:
        plt.clf()

        # Compute Whittle Indices
        whittle_indices = whittle_policy.compute_whittle_indices(current_states)

        # Select top-k nodes
        seeded_nodes_whittle_ids = whittle_policy.select_top_k(whittle_indices, NUM_ACTIONS)

        # Apply actions and transitions
        ns.active_state_transition_graph_indices(G, seeded_nodes_whittle_ids)
        ns.passive_state_transition_without_neighbors(G, exempt_nodes=set(seeded_nodes_whittle_ids))
        newly_activated = ns.independent_cascade_allNodes(G, edge_weight=CASCADE_PROB)
        ns.rearm_nodes(G)

        # Update states
        current_states = {node_id: int(G.nodes[node_id]['obj'].isActive()) for node_id in G.nodes()}

        # Compute reward at this timestep
        reward = ns.reward_function(G, seed=None)  # Assumes ns.reward_function exists and returns a float

        # Visualization
        node_color_map = ns.color_nodes(G)
        edge_color_map = ns.color_edges(G)

        # Label nodes with node value and Whittle Index
        # Node value from node_values[node_id]
        # Whittle index from whittle_indices[node_id]
        labels = {node_id: f"{node_id}\nVal:{node_values[node_id]:.2f}\nWI:{whittle_indices[node_id]:.2f}"
                  for node_id in G.nodes()}

        nx.draw(
            G, pos, labels=labels, with_labels=True,
            node_color=node_color_map, edge_color=edge_color_map,
            node_size=800, font_color='white'
        )

        plt.title(f"Whittle Index Simulation - Timestep {timestep}")
        plt.show(block=False)
        plt.pause(0.5)

        # Print status including reward
        print(f"Timestep {timestep}:")
        print(f"Seeded Nodes (Whittle): {seeded_nodes_whittle_ids}")
        print(f"Newly Activated via Cascade: {newly_activated}")
        print(f"Whittle Indices: {whittle_indices}")
        print(f"Reward this timestep: {reward}\n")

        timestep += 1

        # Wait for user input to proceed
        input("Press Enter to proceed to the next timestep...")

    plt.show()

if __name__ == "__main__":
    main()