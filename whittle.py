import numpy as np
import heapq
from multiprocessing import Pool, cpu_count

# Default thresholds (adjust as needed)
WHITTLE_THRESHOLD = 1e-4
VALUE_ITERATION_THRESHOLD = 1e-2

def arm_value_iteration(transitions, state, lamb_val, discount, node_value, threshold=VALUE_ITERATION_THRESHOLD):
    """ 
    Value iteration for a single arm (node).

    Args:
        transitions (np.ndarray): Transition probability matrix of shape (n_states, n_actions, n_states).
            transitions[s,a,next_s] = P(next_s | s, a)
        state (int): Current state (0 or 1 for a two-state system).
        lamb_val (float): Current lambda value for binary search.
        discount (float): Discount factor (<1).
        node_value (float): Intrinsic value of the node when active.
        threshold (float): Convergence threshold for value iteration.

    Returns:
        int: Optimal action for the given state (0 or 1).
    """
    assert discount < 1, "Discount factor must be less than 1."
    n_states, n_actions, _ = transitions.shape
    value_func = np.zeros(n_states)
    difference = np.ones(n_states)

    def reward(s, a):
        # If state is active (s=1), reward = node_value minus subsidy if action=1.
        # If state is inactive (s=0), no direct reward.
        if s == 1:
            return node_value - (a * lamb_val)
        else:
            return 0.0

    while np.max(difference) >= threshold:
        orig_value_func = value_func.copy()
        Q_func = np.zeros((n_states, n_actions))

        for s in range(n_states):
            for a in range(n_actions):
                val = 0.0
                r = reward(s, a)
                for next_s in range(n_states):
                    val += transitions[s, a, next_s] * (r + discount * value_func[next_s])
                Q_func[s, a] = val

        value_func = np.max(Q_func, axis=1)
        difference = np.abs(orig_value_func - value_func)

    return np.argmax(Q_func[state, :])

def get_init_bounds(transitions):
    """
    Initialize bounds for the Whittle index.
    """
    lb = -10.0
    ub = 10.0
    return lb, ub

def arm_compute_whittle(transitions, state, discount, node_value, subsidy_break=0.0, eps=WHITTLE_THRESHOLD):
    """
    Compute the Whittle Index for a single arm using binary search.

    Args:
        transitions (np.ndarray): (n_states, n_actions, n_states) transition probabilities.
        state (int): Current state (0 or 1).
        discount (float): Discount factor for future rewards.
        node_value (float): Value of the node when active.
        subsidy_break (float): Early break threshold for subsidies (optional).
        eps (float): Convergence threshold.

    Returns:
        float: The computed Whittle Index.
    """
    lb, ub = get_init_bounds(transitions)

    while abs(ub - lb) > eps:
        lamb_val = (lb + ub) / 2.0

        if ub < subsidy_break:
            # Early stopping if desired
            return -10.0

        action = arm_value_iteration(
            transitions=transitions,
            state=state,
            lamb_val=lamb_val,
            discount=discount,
            node_value=node_value,
            threshold=VALUE_ITERATION_THRESHOLD
        )

        if action == 0:
            # Optimal action is passive, subsidy too high
            ub = lamb_val
        elif action == 1:
            # Optimal action is active, subsidy too low
            lb = lamb_val
        else:
            raise ValueError("Action not binary.")

    return (ub + lb) / 2.0


# Helper function for parallel processing
def compute_whittle_for_node(args):
    node_id, transition_matrix, state, node_value, discount, subsidy_break, eps = args
    wi = arm_compute_whittle(
        transitions=transition_matrix,
        state=state,
        discount=discount,
        node_value=node_value,
        subsidy_break=subsidy_break,
        eps=eps
    )
    return (node_id, wi)

class WhittleIndexPolicy:
    """
    Whittle Index Policy for selecting actions based on computed indices.

    Attributes:
        transitions (dict): {node_id: transition_matrix}
        node_values (dict): {node_id: node_value}
        discount (float): Discount factor
        subsidy_break (float): Subsidy threshold
        eps (float): Convergence threshold
    """

    def __init__(self, transitions, node_values, discount=0.99, subsidy_break=0.0, eps=1e-4):
        self.transitions = transitions
        self.node_values = node_values
        self.discount = discount
        self.subsidy_break = subsidy_break
        self.eps = eps

    def compute_whittle_indices(self, current_states):
        """
        Compute Whittle Indices for all nodes in parallel.

        Args:
            current_states (dict): {node_id: state (0 or 1)}

        Returns:
            dict: {node_id: whittle_index}
        """
        whittle_indices = {}
        args_list = []
        for node_id, transition_matrix in self.transitions.items():
            state = current_states.get(node_id, 0)
            node_value = self.node_values.get(node_id, 1)
            args = (node_id, transition_matrix, state, node_value, self.discount, self.subsidy_break, self.eps)
            args_list.append(args)

        # Use as many processes as we have CPU cores or the number of nodes, whichever is smaller
        num_processes = min(cpu_count(), len(args_list))

        with Pool(processes=num_processes) as pool:
            results = pool.map(compute_whittle_for_node, args_list)

        for node_id, wi in results:
            whittle_indices[node_id] = wi

        return whittle_indices

    def select_top_k(self, whittle_indices, k):
        """
        Select the top-k nodes with the highest Whittle Indices.

        Args:
            whittle_indices (dict): {node_id: whittle_index}
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

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkSim import NetworkSim as ns
import copy
import heapq

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