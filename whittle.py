# whittle.py (modified)

import numpy as np
import heapq
from multiprocessing import Pool, cpu_count

WHITTLE_THRESHOLD = 1e-4
VALUE_ITERATION_THRESHOLD = 1e-2

def arm_value_iteration(transitions, state, lamb_val, discount, node_value, threshold=VALUE_ITERATION_THRESHOLD):
    assert discount < 1, "Discount factor must be less than 1."
    n_states, n_actions, _ = transitions.shape
    value_func = np.zeros(n_states)
    difference = np.ones(n_states)

    def reward(s, a):
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
    lb = -10.0
    ub = 10.0
    return lb, ub

def arm_compute_whittle(transitions, state, discount, node_value, subsidy_break=0.0, eps=WHITTLE_THRESHOLD):
    lb, ub = get_init_bounds(transitions)

    while abs(ub - lb) > eps:
        lamb_val = (lb + ub) / 2.0

        if ub < subsidy_break:
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
            ub = lamb_val
        elif action == 1:
            lb = lamb_val
        else:
            raise ValueError("Action not binary.")

    return (ub + lb) / 2.0

def compute_whittle_for_node(args):
    node_id, transition_matrix, discount, node_value, subsidy_break, eps = args
    # Compute WI for both states: 0 and 1
    wi_state_0 = arm_compute_whittle(transition_matrix, 0, discount, node_value, subsidy_break, eps)
    wi_state_1 = arm_compute_whittle(transition_matrix, 1, discount, node_value, subsidy_break, eps)
    return (node_id, wi_state_0, wi_state_1)

class WhittleIndexPolicy:
    def __init__(self, transitions, node_values, discount=0.99, subsidy_break=0.0, eps=1e-4):
        self.transitions = transitions
        self.node_values = node_values
        self.discount = discount
        self.subsidy_break = subsidy_break
        self.eps = eps

        # Precompute Whittle indices for both states for each node
        self.precomputed_wi = self._precompute_whittle_indices()

    def _precompute_whittle_indices(self):
        args_list = []
        for node_id, transition_matrix in self.transitions.items():
            node_value = self.node_values.get(node_id, 1)
            args = (node_id, transition_matrix, self.discount, node_value, self.subsidy_break, self.eps)
            args_list.append(args)

        num_processes = min(cpu_count(), len(args_list))

        with Pool(processes=num_processes) as pool:
            results = pool.map(compute_whittle_for_node, args_list)

        # Store results in a dict: {node_id: {0: wi0, 1: wi1}}
        precomputed = {}
        for node_id, wi0, wi1 in results:
            precomputed[node_id] = {0: wi0, 1: wi1}
        return precomputed

    def compute_whittle_indices(self, current_states):
        # Now we just lookup from precomputed_wi instead of recomputing
        whittle_indices = {}
        for node_id in self.transitions.keys():
            state = current_states.get(node_id, 0)
            # Get the whittle index for this state from precomputed dictionary
            whittle_indices[node_id] = self.precomputed_wi[node_id][state]
        return whittle_indices

    def select_top_k(self, whittle_indices, k):
        if k <= 0:
            return []
        top_k = heapq.nlargest(k, whittle_indices.items(), key=lambda x: x[1])
        selected_nodes = [node_id for node_id, _ in top_k]
        return selected_nodes
