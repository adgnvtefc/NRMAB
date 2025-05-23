import networkx as nx
import random
from simpleNode import SimpleNode as Node
import itertools
import copy
import math

class NetworkSim:
    @staticmethod    
    def generate_random_nodes(num, value_low, value_high):
        
        nodes = {}
        for i in range(num):
            # Define ranges for transition probabilities
            # Active actions have better transition probabilities

            #ensure active nodes don't deactivate too often
            active_activation_active_action = round(random.uniform(0.8, 1.0), 4)
            active_activation_passive_action = round(random.uniform(0.7, active_activation_active_action), 4)

            passive_activation_active_action = round(random.uniform(0.5, 1), 4)
            passive_activation_passive_action = round(random.uniform(0.0, passive_activation_active_action), 4)

            node_value = random.uniform(value_low, value_high)

            # Instantiate the SimpleNode with generated probabilities and value
            node = Node(
                active_activation_active_action=active_activation_active_action,
                active_activation_passive_action=active_activation_passive_action,
                passive_activation_active_action=passive_activation_active_action,
                passive_activation_passive_action=passive_activation_passive_action,
                value=node_value
            )

            # Add the node to the dictionary
            nodes[i] = {"obj": node}

        return nodes
    
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
    def init_random_graph(num_nodes, num_edges, value_low, value_high):
        G = nx.Graph()
        G.add_nodes_from(NetworkSim.generate_random_nodes(num_nodes, value_low, value_high).items())
        G.add_edges_from(NetworkSim.generate_random_edges(num_edges, G))
        return G

  
    @staticmethod
    def passive_state_transition_without_neighbors(graph, exempt_nodes = None):
        changed = set()
        if exempt_nodes is None:
            exempt_nodes = set()
        if not hasattr(exempt_nodes, '__iter__'):
            exempt_nodes = set([exempt_nodes])
        for node in graph.nodes():
            node_obj = graph.nodes[node]['obj']  # Get the SimpleNode object
            if node_obj in exempt_nodes:
                continue
            original_state = node_obj.isActive()
            new_state = node_obj.transition(action=False)
            if new_state != original_state:
                changed.add(node_obj)
        return changed
    
    #note that nodes is a list of nodes here
    @staticmethod
    def active_state_transition(nodes):
        changed = set()
        for node in nodes:
            original_state = node.isActive()
            new_state = node.transition(action=True)
            if new_state != original_state:
                changed.add(node)
        return changed
    
    #same as above, but takes list of indices of a graph
    @staticmethod
    def active_state_transition_graph_indices(graph, node_indices):
        changed = set()
        if not hasattr(node_indices, "__iter__"):
                node_indices = [node_indices]
        for index in node_indices:
            node = graph.nodes[index]['obj']
            original_state = node.isActive()
            new_state = node.transition(action=True)
            if new_state != original_state:
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
                if not graph.nodes[neighbor]['obj'].isActive():
                    # Attempt to activate neighbor with probability 'edge_weight'
                    if random.random() <= edge_weight:
                        graph.nodes[neighbor]['obj'].activate()
                        newlyActivated.add(neighbor)
        if len(newlyActivated) > 0:
            newlyActivated.update(NetworkSim.independent_cascade_allNodes(graph, edge_weight))
        
        return newlyActivated
    
    #rearm the nodes for firing off in the next casade every timestep
    @staticmethod
    def rearm_nodes(graph):
        for node in graph:
            graph.nodes[node]['obj'].rearm()
        return True


    #helper function for selecting tuples of num actions among all nodes
    #returns list of tuples of graph indices
    @staticmethod
    def generate_possible_actions(graph, num):
        nodes = [node for node in graph.nodes() if not graph.nodes[node]['obj'].isActive()]
        return list(itertools.combinations(nodes, num))


    #same as above but returns node objects instead
    @staticmethod
    def generate_possible_actions_nodes(graph, num):
        nodes = [graph[node]['obj'] for node in graph.nodes()]
        return list(itertools.combinations(nodes, num))

    
    #finds the rewards of a function given seed
    @staticmethod
    def reward_function(graph, seed):
        value = 0
        for node in graph.nodes():
            node_obj = graph.nodes[node]['obj']
            if not hasattr(seed, '__iter__'):
                seed = [seed]
            if (seed is not None and node in seed) or node_obj.isActive():
                value += node_obj.getValue()
        
        return value


    #defines the value of being in a certain state, denoted as V(s)
    @staticmethod
    def state_value_function(graph, num=1, gamma=0.7, horizon=1, max_horizon = None, num_samples=5):
        
        if horizon == 0:
            return (0, None)
        
        if max_horizon == None:
            max_horizon = horizon
        
        horizon -= 1

        possible_actions = NetworkSim.generate_possible_actions(graph, num)
        sampled_actions = random.sample(possible_actions, min(10, len(possible_actions)))

        max_value = float('-inf')
        optimal_action = None
        for action in sampled_actions:
            value = NetworkSim.action_value_function(graph, action, num_actions=num, gamma=gamma, horizon=horizon, max_horizon=max_horizon, num_samples=num_samples)
            if value > max_value:
                max_value = value
                optimal_action = action
        
        return (max_value, optimal_action)

    @staticmethod
    def action_value_function(graph, action, num_actions=1, cascade_prob = 0.1, gamma=0.99, horizon=3, max_horizon=3, num_samples=10):
        #the rewards for performing the actions specified on the current state
        immediate_reward = NetworkSim.reward_function(graph, action)

        # Simulate possible next states and compute expected future value
        total_future_value = 0
        
        #sample num_samples numbers of future states
        for _ in range(num_samples):
            #simulate next state based on the graph and action taken
            next_state = NetworkSim.simulate_next_state(graph, action, cascade_prob)

            future_value = NetworkSim.state_value_function(
                next_state, num=num_actions, gamma=gamma, horizon=horizon, max_horizon=max_horizon, num_samples=num_samples)[0]
            total_future_value += future_value

        expected_future_value = total_future_value / num_samples

        #gamma should change properly now (ex on first of horizon=2, max_horizon = 3, gamma**1, on second, horizon=1, max_horizon = 3, gamma ** 2)
        total_value = immediate_reward + (gamma ** (max_horizon - horizon)) * expected_future_value
        return total_value
    
    #helper function to simulate the full next state of the graph
    @staticmethod
    def simulate_next_state(graph, action, cascade_prob):
        # Returns a deep copy of the graph representing the next state after simulating transitions
        new_graph = copy.deepcopy(graph)
        #simulate passive transition
        NetworkSim.passive_state_transition_without_neighbors(new_graph, action)
        #simulate active transition
        NetworkSim.active_state_transition_graph_indices(new_graph, action)
        NetworkSim.independent_cascade_allNodes(new_graph, cascade_prob)
        NetworkSim.rearm_nodes(new_graph)
        # Update edge activations (i dont think you actually need this function)
        NetworkSim.determine_edge_activation(new_graph)
        return new_graph

    @staticmethod    
    def build_graph_from_edgelist(edgelist_path, value_low, value_high):
        edges = []
        with open(edgelist_path, 'r') as f:
            for line in f:
                # each line has "source destination"
                s, d = line.strip().split()
                s, d = int(s), int(d)
                edges.append((s, d))
        # 2) Identify all unique nodes
        unique_nodes = set()
        for (src, dst) in edges:
            unique_nodes.add(src)
            unique_nodes.add(dst)

        # Sort them so we can index consistently
        unique_nodes = sorted(unique_nodes)
        num_nodes = len(unique_nodes)

        random_nodes = NetworkSim.generate_random_nodes(num_nodes, value_low, value_high)

        G = nx.Graph()

        node_id_map = {}  # real_node_id -> index in [0..num_nodes-1]
        for idx, real_node_id in enumerate(unique_nodes):
            node_id_map[real_node_id] = idx
            # `random_nodes[idx]` is a dict: {"obj": <Node>}
            G.add_node(real_node_id, obj=random_nodes[idx]["obj"])

        for (s, d) in edges:
            G.add_edge(s, d)

        return G

    
