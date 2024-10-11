from networkSim import NetworkSim as ns
import networkx as nx
import matplotlib.pyplot as plt
from tabularbellman import TabularBellman as tb
from networkvis import NetworkVis as nv

#activation chance in passive action
PASSIVE_ACTIVATION_CHANCE = 0.1
#deactivation chance in passive action
PASSIVE_DEACTIVATION_CHANCE = 0.3

ACTIVE_ACTIVATION_CHANCE = 0.95
ACTIVE_DEACTIVATION_CHANCE = 0.05

G = ns.init_random_graph(12, 20, PASSIVE_ACTIVATION_CHANCE, PASSIVE_DEACTIVATION_CHANCE, ACTIVE_ACTIVATION_CHANCE, ACTIVE_DEACTIVATION_CHANCE)
node_obj_to_id = {data['obj']: node_id for node_id, data in G.nodes(data=True)}


first_itr = True
next_states = []

pos = nx.spring_layout(G)  # Positioning of nodes
timestep = 0

#this current configuration gives decent results -- after lowing spontaneous activation chance and cascade chance
#higher cascade chances lead to, interestingly enough, worse algorithm results
tab = tb(G, num_actions=2)
tab.update_q_table(num_iterations=3, num_samples=3)

# Start simulation loop
while True:
    plt.clf()  # Clear the previous plot

    seeded_nodes, utility = tab.get_best_action(G)
    print("Best action: " + str(seeded_nodes))
    print("Value gain: " + str(utility))
    nv.do_things(G, pos, seeded_nodes, node_obj_to_id, timestep)
    timestep += 1