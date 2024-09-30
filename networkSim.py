import networkx as nx
import matplotlib.pyplot as plt
import random
from simpleNode import SimpleNode as Node
import heapq


class NetworkSim:
    @staticmethod
    def generate_random_nodes(num, activation_chance, deactivation_chance):
        return {i: {"obj": Node(activation_chance, deactivation_chance)} for i in range(num)}

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

    @staticmethod
    def color_nodes(graph):
        return ['green' if graph.nodes[node]['obj'].isActive() else 'red' for node in graph.nodes()]

    @staticmethod
    def color_edges(graph):
        NetworkSim.determine_edge_activation(graph)
        return ['blue' if graph.edges[edge]['active'] else 'gray' for edge in graph.edges()]

    @staticmethod
    def init_random_graph(num_nodes, num_edges, activation_chance, deactivation_chance):
        G = nx.Graph()
        G.add_nodes_from(NetworkSim.generate_random_nodes(num_nodes, activation_chance, deactivation_chance).items())
        G.add_edges_from(NetworkSim.generate_random_edges(num_edges, G))
        return G

    @staticmethod
    def modify_activation(count):
        return 0.05 * count

    @staticmethod
    def modify_deactivation(count):
        return 0.05 * count

    @staticmethod
    def calculate_next_node_states(graph):
        next_states = []
        for node in graph.nodes():
            node_obj = graph.nodes[node]['obj']  # Get the SimpleNode object
            active_edge_count = sum(
                1 for neighbor in graph.neighbors(node)
                if graph.get_edge_data(node, neighbor)['active'] == True
            )

            adjusted_activation_chance = node_obj.activation_chance + NetworkSim.modify_activation(active_edge_count)
            adjusted_deactivation_chance = node_obj.deactivation_chance - NetworkSim.modify_deactivation(active_edge_count)

            adjusted_activation_chance = min(max(adjusted_activation_chance, 0), 1)
            adjusted_deactivation_chance = min(max(adjusted_deactivation_chance, 0), 1)

            if node_obj.isActive():
                if random.random() < adjusted_deactivation_chance:
                    next_states.append(False)
                else:
                    next_states.append(True)
            else:
                if random.random() < adjusted_activation_chance:
                    next_states.append(True)
                else:
                    next_states.append(False)
        return next_states
    
    @staticmethod
    def passive_state_transition_without_neighbors(graph, exempt_nodes = None):
        changed = set()
        for node in graph.nodes():
            node_obj = graph.nodes[node]['obj']  # Get the SimpleNode object
            if node_obj in exempt_nodes:
                continue
            if node_obj.isActive():
                if random.random() < node_obj.deactivation_chance:
                    graph.nodes[node]['obj'].deactivate()
                    changed.add(node_obj)
            else:
                if random.random() < node_obj.activation_chance:
                    graph.nodes[node]['obj'].activate()
                    changed.add(node_obj)
        return changed
    
    @staticmethod
    def seed_and_transition(graph, seed_function, **kwargs):
        seeded_set = seed_function(graph, **kwargs)
        return (seeded_set, NetworkSim.passive_state_transition_without_neighbors(graph, exempt_nodes = seeded_set))

    ###change -- within each timestep, each node can only activate once
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


                # Attempt to activate neighbor with probability 'edge_weight'
                if random.random() <= edge_weight:
                    graph.nodes[neighbor]['obj'].active = True
                    newlyActivated.add(neighbor)
        #currently just recursively call cascade on the newly updated nodes
        #efficiency can probably be improved using subgraphs
        if len(newlyActivated) > 0:
            newlyActivated.update(NetworkSim.independent_cascade_allNodes(graph, edge_weight))
        
        return newlyActivated
    
    #also consider future reward for that particular activation -- in the next timestep
    #usually -- bellman equation -- define value function of a certain state, and use that to write bellman equation
    #hill climbing considered with respect to this value function -- if a node is in center of graph, this node should have much higher activation
    #node based on self activation / deactivation probabilities
    #use dp
    #careful -- state and actions can be very combinatorial, reach very high levels
    
    #can append the write up you have to the application -- may not have it submitted in time

    #value -- take into account active/inactive nodes at current state, and use that to estimate total value of graph
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

            #can change this activation function here to instead transition based on active transition probabilities
            graph.nodes[node]['obj'].activate()

        print(top_nodes)
        
        return seeded_set

    

    @staticmethod
    def random_seed(graph, num=1):
        inactive_nodes = [node for node in graph.nodes() if not graph.nodes[node]['obj'].isActive()]

        if len(inactive_nodes) < num:
            num = len(inactive_nodes)

        selected_nodes = random.sample(inactive_nodes, num)

        seeded_set = set()

        for node in selected_nodes:
            graph.nodes[node]['obj'].active = True  # Assuming the 'SimpleNode' class has an 'active' attribute
            #seeded_set contains simplenodes
            seeded_set.add(graph.nodes[node]['obj'])

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