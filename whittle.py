# whittle.py

"""
Whittle Index Computation Module

This module provides functions and the WhittleIndexPolicy class to compute the Whittle Index
for each arm (node) in a network and select actions based on these indices. It leverages
value iteration and binary search to determine optimal subsidies for activating nodes.

Dependencies:
    - numpy
    - heapq
"""

import numpy as np
import heapq

# ================================
# Whittle Index Computation Functions
# ================================

def arm_value_iteration(transitions, state, lamb_val, discount, node_value, threshold=1e-2):
    """ 
    Value iteration for a single arm (node).

    Args:
        transitions (np.ndarray): Transition probabilities matrix of shape (n_states, n_actions, n_states).
        state (int): The current state (0: Passive, 1: Active).
        lamb_val (float): The current lambda value in binary search.
        discount (float): Discount factor for future rewards.
        node_value (float): The value of the node when active.
        threshold (float): Convergence threshold for value iteration.

    Returns:
        int: The optimal action (0: Passive, 1: Active) for the given state.
    """
    assert discount < 1, "Discount factor must be less than 1."

    n_states, n_actions, _ = transitions.shape
    value_func = np.zeros(n_states)
    difference = np.ones(n_states)
    iters = 0

    # Lambda-adjusted reward function
    def reward(s, a):
        if s == 1:
            return node_value - (a * lamb_val)
        else:
            return 0  # Assuming no reward when passive

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # Calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                for next_s in range(n_states):
                    Q_func[s, a] += transitions[s, a, next_s] * (reward(s, a) + discount * value_func[next_s])

        # Update value function
        value_func = np.max(Q_func, axis=1)

        # Calculate difference for convergence
        difference = np.abs(orig_value_func - value_func)

    # Return the action with the highest Q-value for the specified state
    return np.argmax(Q_func[state, :])

def get_init_bounds(transitions):
    """ 
    Initialize the lower and upper bounds for the Whittle Index.

    Args:
        transitions (np.ndarray): Transition probabilities matrix.

    Returns:
        tuple: Lower and upper bounds (lb, ub).
    """
    lb = -10.0  # Lower bound for lambda
    ub = 10.0   # Upper bound for lambda
    return lb, ub

def arm_compute_whittle(transitions, state, discount, node_value, subsidy_break=0.0, eps=1e-4):
    """
    Compute the Whittle Index for a single arm using binary search.

    Args:
        transitions (np.ndarray): Transition probabilities matrix of shape (n_states, n_actions, n_states).
        state (int): The current state for which to compute the Whittle Index (0 or 1).
        discount (float): Discount factor for future rewards.
        node_value (float): The value of the node when active.
        subsidy_break (float): The minimum subsidy value at which to stop iterating.
        eps (float): Convergence threshold for binary search.

    Returns:
        float: The computed Whittle Index subsidy.
    """
    lb, ub = get_init_bounds(transitions)  # Initialize bounds on WI

    while abs(ub - lb) > eps:
        lamb_val = (lb + ub) / 2

        # Compute the optimal action using value iteration
        action = arm_value_iteration(
            transitions=transitions,
            state=state,
            lamb_val=lamb_val,
            discount=discount,
            node_value=node_value,
            threshold=1e-2
        )

        if action == 0:
            # Optimal action is passive: subsidy is too high
            ub = lamb_val
        elif action == 1:
            # Optimal action is active: subsidy is too low
            lb = lamb_val
        else:
            raise ValueError(f"Action not binary: {action}")

    subsidy = (ub + lb) / 2
    return subsidy

# ================================
# WhittleIndexPolicy Class
# ================================

class WhittleIndexPolicy:
    """
    Whittle Index Policy for selecting actions based on computed indices.

    Attributes:
        transitions (dict): Dictionary mapping node IDs to their transition matrices.
        node_values (dict): Dictionary mapping node IDs to their values.
        discount (float): Discount factor for future rewards.
        subsidy_break (float): Subsidy threshold for early termination.
        eps (float): Convergence threshold for binary search.
    """

    def __init__(self, transitions, node_values, discount=0.99, subsidy_break=0.0, eps=1e-4):
        """
        Initialize the WhittleIndexPolicy.

        Args:
            transitions (dict): Dictionary where keys are node IDs and values are transition matrices of shape (n_states, n_actions, n_states).
            node_values (dict): Dictionary where keys are node IDs and values are the node's value.
            discount (float): Discount factor for future rewards.
            subsidy_break (float): Subsidy threshold for early termination.
            eps (float): Convergence threshold for binary search.
        """
        self.transitions = transitions  # {node_id: transition_matrix}
        self.node_values = node_values  # {node_id: node_value}
        self.discount = discount
        self.subsidy_break = subsidy_break
        self.eps = eps

    def compute_whittle_indices(self, current_states):
        """
        Compute Whittle Indices for all nodes based on their transition matrices and current states.

        Args:
            current_states (dict): Dictionary mapping node IDs to their current states (0: Passive, 1: Active).

        Returns:
            dict: Dictionary mapping node IDs to their computed Whittle Indices.
        """
        whittle_indices = {}
        for node_id, transition_matrix in self.transitions.items():
            state = current_states.get(node_id, 0)  # Default to 0 if not specified
            node_value = self.node_values.get(node_id, 1)  # Default node value to 1 if not specified
            wi = arm_compute_whittle(
                transitions=transition_matrix,
                state=state,
                discount=self.discount,
                node_value=node_value,
                subsidy_break=self.subsidy_break,
                eps=self.eps
            )
            whittle_indices[node_id] = wi
        return whittle_indices

    def select_top_k(self, whittle_indices, k):
        """
        Select the top-k nodes with the highest Whittle Indices.

        Args:
            whittle_indices (dict): Dictionary mapping node IDs to Whittle Indices.
            k (int): Number of nodes to select.

        Returns:
            list: List of selected node IDs.
        """
        if k <= 0:
            return []
        top_k = heapq.nlargest(k, whittle_indices.items(), key=lambda x: x[1])
        selected_nodes = [node_id for node_id, _ in top_k]
        return selected_nodes


# whittle_test.py

"""
Whittle Index Simulation and Visualization

This script initializes a network graph, computes the Whittle Index for each node,
selects top-k nodes based on the indices, applies actions, and visualizes the network over time.
"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from whittle import WhittleIndexPolicy
from networkSim import NetworkSim as ns
from hillClimb import HillClimb as hc
import copy
import heapq


# ================================
# Simulation and Visualization
# ================================

def main():
    """
    Main function to run the Whittle Index simulation and visualization.
    """
    # ========================
    # Configuration Parameters
    # ========================

    # Whittle Index parameters
    WHITTLE_THRESHOLD = 1e-4  # Convergence threshold for binary search
    VALUE_ITERATION_THRESHOLD = 1e-2  # Threshold for value iteration
    DISCOUNT = 0.99  # Discount factor
    NUM_ACTIONS = 2  # Number of nodes to activate per timestep (k)
    CASCADE_PROB = 0.05  # Probability of cascade per edge

    # Graph parameters
    NUM_NODES = 10
    NUM_EDGES = 30
    VALUE_LOW = 10
    VALUE_HIGH = 100
    RANDOM_SEED = 42  # For reproducibility

    # ========================
    # Initialize the Graph
    # ========================

    print("Initializing the network graph...")
    G = ns.init_random_graph(
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        value_low=VALUE_LOW,
        value_high=VALUE_HIGH
    )

    # ========================
    # Define Transition Matrices and Node Values
    # ========================

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
        transition_matrix[0, 0, 1] = node_obj.passive_activation_passive  # P(P→A | a=0)
        transition_matrix[0, 0, 0] = 1 - node_obj.passive_activation_passive  # P(P→P | a=0)

        # Action Active (a=1)
        transition_matrix[0, 1, 1] = node_obj.passive_activation_active  # P(P→A | a=1)
        transition_matrix[0, 1, 0] = 1 - node_obj.passive_activation_active  # P(P→P | a=1)

        # From Active state (s=1)
        # Action Passive (a=0)
        transition_matrix[1, 0, 1] = node_obj.active_activation_passive  # P(A→A | a=0)
        transition_matrix[1, 0, 0] = 1 - node_obj.active_activation_passive  # P(A→P | a=0)

        # Action Active (a=1)
        transition_matrix[1, 1, 1] = node_obj.active_activation_active  # P(A→A | a=1)
        transition_matrix[1, 1, 0] = 1 - node_obj.active_activation_active  # P(A→P | a=1)

        transitions_whittle[node_id] = transition_matrix
        node_values[node_id] = node_obj.getValue()

    # ========================
    # Initialize WhittleIndexPolicy
    # ========================

    print("Initializing WhittleIndexPolicy...")
    whittle_policy = WhittleIndexPolicy(
        transitions=transitions_whittle,
        node_values=node_values,
        discount=DISCOUNT,
        subsidy_break=0.0,  # Adjust based on your problem
        eps=WHITTLE_THRESHOLD
    )

    # ========================
    # Initialize Current States
    # ========================

    current_states = {node_id: 0 for node_id in G.nodes()}  # Assuming all start inactive

    # ========================
    # Initialize Visualization Layout
    # ========================

    pos = nx.spring_layout(G, seed=RANDOM_SEED)  # Fixed seed for consistent layout

    # ========================
    # Simulation Loop Parameters
    # ========================

    MAX_TIMESTEPS = 10  # For testing, run for 10 timesteps
    timestep = 0

    # ========================
    # Simulation Loop
    # ========================

    print("Starting simulation...")
    while timestep < MAX_TIMESTEPS:
        plt.clf()  # Clear the previous plot

        # Compute Whittle Indices for all nodes based on current states
        whittle_indices = whittle_policy.compute_whittle_indices(current_states)

        # Select top-k nodes with highest Whittle Indices
        seeded_nodes_whittle_ids = whittle_policy.select_top_k(whittle_indices, NUM_ACTIONS)
        seeded_nodes_whittle = seeded_nodes_whittle_ids  # List of node IDs

        # Apply actions and state transitions
        # Activate selected nodes
        changed_seeded = ns.active_state_transition_graph_indices(G, seeded_nodes_whittle_ids)
        # Apply passive transitions to all other nodes
        changed_passive = ns.passive_state_transition_without_neighbors(G, exempt_nodes=set(seeded_nodes_whittle_ids))
        # Handle cascades
        newly_activated = ns.independent_cascade_allNodes(G, edge_weight=CASCADE_PROB)
        # Rearm nodes for the next timestep
        ns.rearm_nodes(G)

        # Update current states based on node statuses
        current_states = {node_id: int(G.nodes[node_id]['obj'].isActive()) for node_id in G.nodes()}

        # Visualization
        node_color_map = ns.color_nodes(G)
        edge_color_map = ns.color_edges(G)

        # Label nodes with their Whittle Indices
        labels = {node_id: f"{node_id}\nWI:{whittle_indices[node_id]:.2f}" for node_id in G.nodes()}

        # Draw the updated graph
        nx.draw(
            G, pos, labels=labels, with_labels=True,
            node_color=node_color_map, edge_color=edge_color_map,
            node_size=800, font_color='white'
        )

        plt.title(f"Whittle Index Simulation - Timestep {timestep}")
        plt.show(block=False)
        plt.pause(0.5)  # Pause for half a second to visualize changes

        # Print status
        print(f"Timestep {timestep}:")
        print(f"Seeded Nodes (Whittle): {seeded_nodes_whittle_ids}")
        print(f"Newly Activated via Cascade: {newly_activated}")
        print(f"Whittle Indices: {whittle_indices}\n")

        timestep += 1

        # Optionally, wait for user input to proceed to the next step
        # Uncomment the line below if you prefer step-by-step execution
        # input("Press Enter to proceed to the next timestep...")

    plt.show()

# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    main()
