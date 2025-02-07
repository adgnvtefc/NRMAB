import networkx as nx
from simpleNode import SimpleNode as Node
import itertools
import copy
from networkSim import NetworkSim as ns
import numpy as np
from math import comb
from tqdm import tqdm
import random

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
    def update_q_table(self, num_episodes=1, steps_per_episode = 100, epsilon = 0.1):
        with tqdm(total=num_episodes, desc="Training Q-table", unit="iteration") as outer_pbar:
            for episode in range(num_episodes):
                # Start from a random state (you could choose a fixed one if desired)
                state_index = random.randrange(self.num_states)

                # Inner loop for steps within an episode
                with tqdm(total=steps_per_episode, desc=f"Episode {episode+1}/{num_episodes}", unit="step", leave=False) as pbar:
                    for step in range(steps_per_episode):
                        # Epsilon-greedy action selection
                        if random.random() < epsilon:
                            action_index = random.randrange(self.num_actions_total)
                        else:
                            action_index = np.argmax(self.qtable[state_index])

                        action = self.all_action_combinations[action_index]

                        # Simulate one step in the environment
                        next_graph, reward = self.simulate_step(state_index, action)

                        # Determine next state
                        next_state_index = self.calculate_state_table_pos(next_graph)

                        # Q-learning update #MEASURE THIS COMPUTATIONAL COST
                        best_next_Q = np.max(self.qtable[next_state_index])
                        current_Q = self.qtable[state_index, action_index]
                        self.qtable[state_index, action_index] = current_Q + self.alpha * (reward + self.gamma * best_next_Q - current_Q)

                        # Move to next state
                        state_index = next_state_index

                        # Update step progress
                        pbar.update(1)

                # Update episode progress
                outer_pbar.update(1)
    
    def simulate_step(self, state_index, action):
        """
        Simulate one step given a state (via state_index) and an action.
        Returns:
            next_graph: graph after applying the action and transitions
            reward: the observed reward from this transition
        """
        graph = self.get_graph_from_state(state_index)

        # Activate chosen nodes
        action_nodes = [graph.nodes[node_idx]['obj'] for node_idx in action]

        # Exempt nodes that were just activated
        exempt_nodes = set(action_nodes)
        ns.passive_state_transition_without_neighbors(graph, exempt_nodes)
        ns.active_state_transition(action_nodes)
        ns.independent_cascade_allNodes(graph, 0.1)
        ns.rearm_nodes(graph)

        reward = ns.reward_function(graph, set(action))
        return graph, reward
    
    def get_best_action(self, graph):
        state_index = self.calculate_state_table_pos(graph)

        action_index = np.argmax(self.qtable[state_index])
        best_utility = np.max(self.qtable[state_index])
        
        best_action = self.get_action_from_index(action_index)
        return (best_action, best_utility)

    #same as above but returns the node objects
    def get_best_action_nodes(self, graph):
        # Get the state index for the current graph
        state_index = self.calculate_state_table_pos(graph)

        # Find the action with the highest utility in the Q-table for the current state
        action_index = np.argmax(self.qtable[state_index])
        best_utility = np.max(self.qtable[state_index])

        # Convert the action index to node indices
        best_action_indices = self.get_action_from_index(action_index)

        # Convert the node indices to node objects
        best_action_nodes = [graph.nodes[node_index]['obj'] for node_index in best_action_indices]

        return (best_action_nodes, best_utility)
    

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