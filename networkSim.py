import networkx as nx
import random
from simpleNode import SimpleNode as Node
import heapq
import itertools
import copy


class NetworkSim:
    @staticmethod
    def generate_random_nodes(num, passive_activation_chance, passive_deactivation_chance, active_activation_change, active_deactivation_chance):
        return {i: {"obj": Node(passive_activation_chance, passive_deactivation_chance, active_activation_change, active_deactivation_chance)} for i in range(num)}

    @staticmethod
    def generate_random_edges(num, graph):
        edges = set()
        while len(edges) < num:
            node1, node2 = random.sample(list(graph.nodes()), 2)
            edges.add((node1, node2))  # Add edge, automatically avoiding duplicates
        return edges

    @staticmethod
    def determine_edge_activation(graph):
        for edge in graph.edges():
            node1, node2 = edge
            node1_obj = graph.nodes[node1]['obj']
            node2_obj = graph.nodes[node2]['obj']
            graph.edges[edge]['active'] = node1_obj.isActive() or node2_obj.isActive()
    
    #hypothethical_seed is a set of nodes where we override the current node values in the graph with the hypothesized. This seed is assumed to be active.
    @staticmethod
    def get_exclusive_active_edges(graph, hypothetical_seed = None):
        edges = set()
        hypothetical_seed = hypothetical_seed or set()  # Ensure hypothetical_seed is a set or an empty set

        for edge in graph.edges():
            #these are graph indices
            node1, node2 = edge
            node1_obj = graph.nodes[node1]['obj']
            node2_obj = graph.nodes[node2]['obj']

            node1_active = node1 in hypothetical_seed or node1_obj.isActive()
            node2_active = node2 in hypothetical_seed or node2_obj.isActive()

            # Add edge if one of the nodes is active but not both
            if node1_active != node2_active:
                edges.add(edge)

        return edges

    @staticmethod
    def color_nodes(graph):
        return ['green' if graph.nodes[node]['obj'].isActive() else 'red' for node in graph.nodes()]

    @staticmethod
    def color_edges(graph):
        NetworkSim.determine_edge_activation(graph)
        return ['blue' if graph.edges[edge]['active'] else 'gray' for edge in graph.edges()]

    @staticmethod
    def init_random_graph(num_nodes, num_edges, activation_chance, deactivation_chance, active_activation_chance, active_deactivation_chance):
        G = nx.Graph()
        G.add_nodes_from(NetworkSim.generate_random_nodes(num_nodes, activation_chance, deactivation_chance, active_activation_chance, active_deactivation_chance).items())
        G.add_edges_from(NetworkSim.generate_random_edges(num_edges, G))
        return G

  
    @staticmethod
    def passive_state_transition_without_neighbors(graph, exempt_nodes = None):
        changed = set()
        for node in graph.nodes():
            node_obj = graph.nodes[node]['obj']  # Get the SimpleNode object
            if node_obj in exempt_nodes:
                continue
            if node_obj.isActive():
                if random.random() < node_obj.getPassiveDeactivationChance():
                    graph.nodes[node]['obj'].deactivate()
                    changed.add(node_obj)
            else:
                if random.random() < node_obj.getPassiveActivationChance():
                    graph.nodes[node]['obj'].activate()
                    changed.add(node_obj)
        return changed
    
    #note that nodes is a list of nodes here
    @staticmethod
    def active_state_transition(nodes):
        changed = set()
        for node in nodes:
            if node.isActive():
                if random.random() < node.getActiveDeactivationChance():
                    node.deactivate()
                    changed.add(node)
            else:
                if random.random() < node.getActiveActivationChance():
                    node.activate()
                    changed.add(node)
        return changed
    
    #same as above, but takes list of indices of a graph
    @staticmethod
    def active_state_transition_graph_indices(graph, node_indices):
        changed = set()
        for index in node_indices:
            node = graph.nodes[index]['obj']
            if node.isActive():
                if random.random() < node.getActiveDeactivationChance():
                    node.deactivate()
                    changed.add(node)
            else:
                if random.random() < node.getActiveActivationChance():
                    node.activate()
                    changed.add(node)
        return changed


    @staticmethod
    def seed_and_passive_transition(graph, seed_function, **kwargs):
        seeded_set = seed_function(graph, **kwargs)
        return (seeded_set, NetworkSim.passive_state_transition_without_neighbors(graph, exempt_nodes = seeded_set))

    @staticmethod
    def independent_cascade_allNodes(graph, edge_weight):
        cascadeNodes = set()
        newlyActivated = set()
        for node in graph:
            if graph.nodes[node]['obj'].isActive() and graph.nodes[node]['obj'].cascade():
                cascadeNodes.add(node)
        
        for node in cascadeNodes:
            neighbors = set(graph.neighbors(node)) - cascadeNodes
            for neighbor in neighbors:
                #if using different edge weights, just query the list of edges from each neigbor, and query the weight of each edge
                #relative to the neighbor
                if not graph.nodes[neighbor]['obj'].isActive():
                    # Attempt to activate neighbor with probability 'edge_weight'
                    if random.random() <= edge_weight:
                        graph.nodes[neighbor]['obj'].activate()
                        newlyActivated.add(neighbor)
        #currently just recursively call cascade on the newly updated nodes
        #efficiency can probably be improved using subgraphs
        if len(newlyActivated) > 0:
            newlyActivated.update(NetworkSim.independent_cascade_allNodes(graph, edge_weight))
        
        return newlyActivated
    
    #rearm the nodes for firing off in the next casade every timestep
    @staticmethod
    def rearm_nodes(graph):
        for node in graph:
            graph.nodes[node]['obj'].rearm()
        return True


    
    #also consider future reward for that particular activation -- in the next timestep
    #usually -- bellman equation -- define value function of a certain state, and use that to write bellman equation
    #hill climbing considered with respect to this value function -- if a node is in center of graph, this node should have much higher activation
    #node based on self activation / deactivation probabilities
    #use dp
    #careful -- state and actions can be very combinatorial, reach very high levels

    #value -- take into account active/inactive nodes at current state, and use that to estimate total value of graph

    #bellman's equation -- https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/bellman_mdp/#key-recap-on-value-functions

    #finds the rewards of a function given seed
    @staticmethod
    def reward_function(graph, seed):
        #defined by the number of active nodes and active edges that have unactivated nodes at either end
        len_edges = len(NetworkSim.get_exclusive_active_edges(graph, seed))

        # Count the number of active nodes
        num_active = 0
        for node in graph.nodes():
            node_obj = graph.nodes[node]['obj']
            # A node is active if it's in the seed or if its object reports it as active
            if node in seed or node_obj.isActive():
                num_active += 1

        #0.2 here is arbitrarily chosen as a placeholder
        reward = num_active + 0.2 * len_edges  # Sum of active nodes and active edges as the reward

        return reward

    #defines the value of being in a certain state, denoted as V(s)
    #num is k, the number of nodes to activate in the subsequent action
    #note: in current implementation, state space gets very very big
    #horizon: how far in the future to look

    #alternative -- make DP table of state/action value function, and repeatedly update state value and action value
    # where V(s) is a table of ALL possible states and Q(s, a) is a table of ALL possible state-action pairs
    # V(s) can be a 1-D list and Q(s) can be a 2-D matrix with s as one indices and A as other indices
    # ACTION: VERIFY SUBMODULARITY F(A + x) - F(A) > f(B + x) - f(B) given A < B
    @staticmethod
    def state_value_function(graph, num=1, gamma=0.7, horizon=1, max_horizon = None, num_samples=5):
        
        if horizon == 0:
            # Base case: horizon is 0, return 0
            return (0, None)
        
        #keep track of our max horizon value
        if max_horizon == None:
            max_horizon = horizon
        
        # "consume" one horizon point to look a step into the future
        horizon -= 1

        #generate tuples of possible actions to take in this state
        possible_actions = NetworkSim.generate_possible_actions(graph, num)

        max_value = float('-inf')
        optimal_action = None
        for action in possible_actions:
            value = NetworkSim.action_value_function(graph, action, num, gamma, horizon, max_horizon, num_samples)
            if value > max_value:
                max_value = value
                #note that this is INDICES
                optimal_action = action
        
        return (max_value, optimal_action)


    #helper method
    #this component consists of four components: 
    # (1) the reward at the current state
    # (2) the probability of passive transitions to the next state
    # (3) the probability cascades from that transformaton
    # (4) the value of that state
    # since exploring the full state space will take too long, we take a separate approach:
    # instead of summing over state and action space multiplying by value function, we simulate the graph for multiple iterations,
    # and take the average reward of each state
    # this explanation does not make much sense, but you will see what i mean
    @staticmethod
    def action_value_function(graph, action, num, gamma, horizon, max_horizon, num_samples):
        #the rewards for performing the actions specified on the current state
        immediate_reward = NetworkSim.reward_function(graph, action)

        # Simulate possible next states and compute expected future value
        total_future_value = 0
        
        #sample num_samples numbers of future states
        for _ in range(num_samples):
            #simulate next state based on the graph and action taken
            next_state = NetworkSim.simulate_next_state(graph, action)

            future_value = NetworkSim.state_value_function(
                next_state, num, gamma, horizon, max_horizon, num_samples)[0]
            total_future_value += future_value

        expected_future_value = total_future_value / num_samples

        #gamma should change properly now (ex on first of horizon=2, max_horizon = 3, gamma**1, on second, horizon=1, max_horizon = 3, gamma ** 2)
        total_value = immediate_reward + (gamma ** max_horizon - horizon) * expected_future_value
        return total_value
    
    #helper function to simulate the full next state of the graph
    #wow the fact that this is so succinct is actually so sexy
    @staticmethod
    def simulate_next_state(graph, action):
        # Returns a deep copy of the graph representing the next state after simulating transitions
        new_graph = copy.deepcopy(graph)

        #simulate passive transition
        NetworkSim.passive_state_transition_without_neighbors(new_graph, action)
        #simulate active transition
        NetworkSim.active_state_transition_graph_indices(new_graph, action)
        
        #0.1 is a placeholder. more advanced functionality will be implemented *eventually*
        NetworkSim.independent_cascade_allNodes(new_graph, 0.1)

        NetworkSim.rearm_nodes(graph)

        # Update edge activations (i dont think you actually need this function)
        NetworkSim.determine_edge_activation(new_graph)

        return new_graph
    
    #helper function for selecting tuples of num actions among INACTIVE nodes for activation
    #returns list of tuples of graph indices
    @staticmethod
    def generate_possible_actions(graph, num):
        #node that these should be node indices in the graph, not node objects
        nodes = [node for node in graph.nodes()]

        #get a list of num-tuples of inactive nodes to attempt activate. note that there is ZERO heuristic in this step
        return list(itertools.combinations(nodes, num))


    #same as above but returns node objects instead
    @staticmethod
    def generate_possible_actions_nodes(graph, num):
        #node that these should be node indices in the graph, not node objects
        nodes = [graph[node]['obj'] for node in graph.nodes()]

        #get a list of num-tuples of inactive nodes to attempt activate. note that there is ZERO heuristic in this step
        return list(itertools.combinations(nodes, num))


    #okay this one isn't really hill climb anymore, more like just pure bellman
    @staticmethod
    def hill_climb_with_bellman(graph, num=1, gamma=0.7, horizon=3, num_samples=5):
        seeded_set = set()
        (utility, action) = NetworkSim.state_value_function(graph, num, gamma, horizon, num_samples)

        #inefficiency carried over from previous hill climbing algo
        print("Predicted utility gain:" + str(utility))

        for node_index in action:
            seeded_set.add(graph.nodes[node_index]['obj'])
        
        return seeded_set
        

    @staticmethod
    def hill_climb(graph, num=1):
        seeded_set = set()
        node_values = []

        #iterate thru graph to see nodes that can pick
        for node in graph:
            #arbitrarily chosen discount factor
            discount_factor = 0.5

            #replace this value with the value function
            value = 0

            #bootstrap solution to not double activate node under guaranteed activation
            if graph.nodes[node]['obj'].isActive():
                value -= 10

            if not graph.nodes[node]['obj'].canCascade():
                value += 1
            else:
                queue = [(node, 0)]
                visited_set = set()
                visited_set.add(node)
                while queue:
                    current_node, depth = queue.pop()                    
                    current_node_obj = graph.nodes[current_node]['obj']
                    #value of nodes decrease as u get further from the activated node
                    value += 1 * discount_factor ** depth

                    #extension -- different edge weights -- multiply discount factor by all edge weights on the path

                    #get the nodes this node can cascade to
                    if not current_node_obj.canCascade():
                        continue
                    for neighbor in graph.neighbors(current_node):
                        neighbor_obj = graph.nodes[neighbor]['obj']
                        if neighbor not in visited_set and neighbor_obj.canCascade():
                            visited_set.add(neighbor)
                            next_depth = depth + 1
                            queue.append((neighbor, next_depth))

            node_values.append((value, node))
        top_nodes = heapq.nlargest(num, node_values)
        selected_nodes = [node for value, node in top_nodes]

        for node in selected_nodes:
            seeded_set.add(graph.nodes[node]['obj'])
            #the actual activation of nodes has been moved to a separate function; this function now only returns nodes to activate

        print(top_nodes)
        
        return seeded_set

    

#implement better selection algorithm here...

#hill climbing algorithm

#whittle index definition 

#algorithm w/ whittle index
#might be more challenging -- assume it doesn't depend on neighbors
#fancy way -- use neural network to parameterize index? whittle index with dependency

#simulate partially observable graph (just change algo to account for only the visible parts)
#keep belief for the unknown nodes -- use this for algorithm
#keep a variable belief -- every iteration in the simulation, run code or function with belief as the input and
#some other neighbors' states as input / output -- gives them new belief