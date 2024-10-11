import networkx as nx
from simpleNode import SimpleNode as Node
import itertools
import copy
from networkSim import NetworkSim as ns
import numpy as np
from math import comb
from tqdm import tqdm


#how the Q table works:
#each entry in the table represents the Q value -- the action value function, in a particular state
#we fill in the Q table and use it to determine the best action
#We can derive the state value function V(s) from taking the max of a row of the Q table corresponding to the value function


#class to implement bellman's equation using Q-Table
class TabularBellman:

    #you should pass in a complete graph to this function
    def __init__(self, graph, num_actions = 1, gamma = 0.9, alpha=0.1):
        self.graph = graph

        self.num_nodes = len(graph.nodes())
        self.num_actions = num_actions
        
        self.gamma = gamma
        self.alpha = alpha
        
        #all possible actions
        self.all_action_combinations = list(itertools.combinations(range(self.num_nodes), self.num_actions))

        self.num_states = 2 ** self.num_nodes
        self.num_actions_total = comb(self.num_nodes, num_actions)
        self.qtable = np.zeros((self.num_states, self.num_actions_total))

    
    #implememted using expected reward for each state
    def update_q_table(self, num_iterations=1, num_samples=5):
        with tqdm(total=num_iterations, desc="Training Q-table", unit="iteration") as outer_pbar:
            for iteration in range(num_iterations):
                # Create a progress bar for the states within each iteration
                with tqdm(total=self.num_states, desc=f"Updating states for iteration {iteration + 1}/{num_iterations}", unit="state", leave=False) as pbar:
                    for state_index in range(self.num_states):
                        graph = self.get_graph_from_state(state_index)
                        for action_index, action in enumerate(self.all_action_combinations):
                            total_future_value = 0
                            reward_total = 0
                            for _ in range(num_samples):
                                next_graph = copy.deepcopy(graph)

                                # Apply the action to the graph
                                for node_index in action:
                                    next_graph.nodes[node_index]['obj'].activate()

                                # Exempt nodes that were just activated
                                exempt_nodes = {next_graph.nodes[node_index]['obj'] for node_index in action}

                                # Simulate transitions
                                ns.passive_state_transition_without_neighbors(next_graph, exempt_nodes)
                                ns.active_state_transition_graph_indices(next_graph, action)
                                ns.independent_cascade_allNodes(next_graph, 0.05)
                                ns.rearm_nodes(next_graph)

                                # Calculate the next state index
                                next_state_index = self.calculate_state_table_pos(next_graph)

                                # Compute the reward for this sample
                                reward = self.get_reward(next_graph, action)

                                # Get max_future_q
                                max_future_q = np.max(self.qtable[next_state_index])

                                total_future_value += max_future_q
                                reward_total += reward

                            # Compute the expected future value and expected reward
                            expected_future_value = total_future_value / num_samples
                            expected_reward = reward_total / num_samples

                            # Current Q-value
                            current_q = self.qtable[state_index, action_index]

                            # Q-learning update
                            new_q = (1 - self.alpha) * current_q + self.alpha * (expected_reward + self.gamma * expected_future_value)
                            self.qtable[state_index, action_index] = new_q

                        # Update the progress bar after processing each state
                        pbar.update(1)

                # Update the outer progress bar after each iteration
                outer_pbar.update(1)
    
    #NOTE: When the graph is almost filled up, this algorithm often displays suboptimal behavior
    def get_best_action(self, graph):
        state_index = self.calculate_state_table_pos(graph)

        action_index = np.argmax(self.qtable[state_index])
        best_utility = np.max(self.qtable[state_index])
        
        best_action = self.get_action_from_index(action_index)
        return (best_action, best_utility)
    
    # Helper function to reconstruct the graph from a state index
    def get_graph_from_state(self, state_index):
        # Create a deep copy of the initial graph to avoid modifying it
        new_graph = copy.deepcopy(self.graph)
        node_list = list(new_graph.nodes())
        for i, node in enumerate(node_list):
            is_active = (state_index >> i) & 1
            new_graph.nodes[node]['obj'].active = bool(is_active)
        return new_graph

    #helper function to "hash" each graph state to achieve O(n) access time in Q table (i don't think you can get better than this?)
    def calculate_state_table_pos(self, graph):
        state_value = 0
        # Iterate through the nodes and calculate the binary number directly
        for i, node in enumerate(graph.nodes()):
            if graph.nodes[node]['obj'].isActive():
                # Shift 1 to the correct position and add it to the state value
                state_value |= (1 << i)
        return state_value
        
    #helper function that, given an action, gives its column in the Q table
    def get_action_pos(self, actions):
        #actions is a tuple of len(num_actions), each of it which is an int between [1, num_nodes]
        #how do?
        sorted_actions = tuple(sorted(actions))
        return self.all_action_combinations.index(sorted_actions)

    #helper function that, given a column in the Q table, finds the action
    def get_action_from_index(self, action_index):
        return self.all_action_combinations[action_index]

    def get_reward(self, graph, action):
        # Compute the reward as per the reward function
        return ns.reward_function(graph, set(action))
