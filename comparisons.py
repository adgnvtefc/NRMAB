from networkSim import NetworkSim as ns
import networkx as nx
import matplotlib.pyplot as plt
from tabularbellman import TabularBellman as tb
from networkvis import NetworkVis as nv
import copy
import random

from hillClimb import HillClimb

#activation chance in passive action
PASSIVE_ACTIVATION_CHANCE = 0.1
#deactivation chance in passive action
PASSIVE_DEACTIVATION_CHANCE = 0.3

ACTIVE_ACTIVATION_CHANCE = 0.95
ACTIVE_DEACTIVATION_CHANCE = 0.05

#sets up singular graph for comparison
G = ns.init_random_graph(10, 20, PASSIVE_ACTIVATION_CHANCE, PASSIVE_DEACTIVATION_CHANCE, ACTIVE_ACTIVATION_CHANCE, ACTIVE_DEACTIVATION_CHANCE)
node_obj_to_id = {data['obj']: node_id for node_id, data in G.nodes(data=True)}


first_itr = True
next_states = []

pos = nx.spring_layout(G)  # Positioning of nodes
timestep = 0

q_graph = copy.deepcopy(G)
#Sets up tabular bellman
tab = tb(G, num_actions=2)
tab.update_q_table(num_iterations=3, num_samples=3)

#hill climbing
hillclimb_graph = copy.deepcopy(G)

#hill climbing but with Bellman's Equation
#this is basically q graph but bad because of time constraints
bellman_hillclimb_graph = copy.deepcopy(G)

random_graph = copy.deepcopy(G)

none_graph = copy.deepcopy(G)

graphs = [q_graph, hillclimb_graph, bellman_hillclimb_graph, random_graph, none_graph]

timestep = 0

# Start simulation loop, but no graphics
while timestep < 100:
    #get the seeded nodes for all the graphs
    q_seeded_nodes, utility = tab.get_best_action_nodes(q_graph)

    #for hill climbing; seeded nodes are node objects
    hillclimb_seeded_nodes = HillClimb.hill_climb(hillclimb_graph, 2)

    bellman_hillclimb_seeded_nodes = HillClimb.hill_climb_with_bellman(graph=bellman_hillclimb_graph, num=2, horizon=1, num_samples=1)

    random_seeded_nodes = random.choice(ns.generate_possible_actions_nodes(random_graph, 2))

    none_seeded_nodes = None

    graph_seeds = [q_seeded_nodes, hillclimb_graph, bellman_hillclimb_graph, random_graph, none_seeded_nodes]

    i = 0
    for graph in graphs:
        #transition step with active and passive transitions
        transition_nodes = ns.passive_state_transition_without_neighbors(graph=graph, exempt_nodes=graph_seeds[i])
        changed_nodes = ns.active_state_transition(graph_seeds[i])

        ns.independent_cascade_allNodes(graph, 0.05)

        ns.rearm_nodes(graph)

    timestep += 1