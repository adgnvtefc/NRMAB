from networkSim import NetworkSim as ns
import networkx as nx
import matplotlib.pyplot as plt
from algorithms.hillClimb import HillClimb as hc

#activation chance in passive action
PASSIVE_ACTIVATION_CHANCE = 0.1
#deactivation chance in passive action
PASSIVE_DEACTIVATION_CHANCE = 0.2

ACTIVE_ACTIVATION_CHANCE = 0.95
ACTIVE_DEACTIVATION_CHANCE = 0.05

# Initialize the graph
# NOTE: THE CURRENT IMPLEMENTATION OF BELLMANS IS HORRIFYINGLY INEFFICIENT AND WILL EXPLODE 
# for an idea of how bad it is:
# we select k actions among all inactive nodes. in the worst case, for 10 nodes, that is 10 c 2 = 45 possible actions.
# for each action, we run bellman's equation, so we do the whole thing... again. That means that when horizon=3 (we peek 3 states into the future), we expand 45^3 states
# now, if num_samples != 1, we multiply the expansions at EACH STEP by that number. so for num_samples=3, we have:
# 45, 45^2 * 2, 45^3 * 2^2
# you get the idea...
G = ns.init_random_graph(10, 30, PASSIVE_ACTIVATION_CHANCE, PASSIVE_DEACTIVATION_CHANCE, ACTIVE_ACTIVATION_CHANCE, ACTIVE_DEACTIVATION_CHANCE)

# Build the reverse mapping after the graph has been created (can be functionlized later)
node_obj_to_id = {data['obj']: node_id for node_id, data in G.nodes(data=True)}


first_itr = True
next_states = []

pos = nx.spring_layout(G)  # Positioning of nodes
timestep = 0

# Start simulation loop
while True:
    plt.clf()  # Clear the previous plot

    # seed nodes according to our function and transition them.
    #(seeded_nodes, transition_nodes) = ns.seed_and_passive_transition(G, ns.hill_climb, num=2)
    (seeded_nodes, transition_nodes) = ns.seed_and_passive_transition(G, hc.hill_climb_with_bellman, num=2, horizon=1, num_samples=1)

    #transition seeded nodes based on active transition probabilities
    changed_nodes = ns.active_state_transition(seeded_nodes)

    for node in changed_nodes:
        node_id = node_obj_to_id.get(node)
        print(f"Node {node_id} is activated.")


    for node in transition_nodes:
        # Get the node identifier using the reverse mapping
        node_id = node_obj_to_id.get(node)
        if node_id is not None:
            # Check if the node is active
            if node.isActive():
                print(f"Node {node_id} transitioned to active.")
            else:
                print(f"Node {node_id} transitioned to inactive.")
        else:
            print("Node object not found in the graph.")


    newlyActivated = ns.independent_cascade_allNodes(G, 0.1)

    ns.rearm_nodes(G)

    print("Cascade Activated " + str(newlyActivated))
    # Update node and edge colors
    node_color_map = ns.color_nodes(G)
    edge_color_map = ns.color_edges(G)

    # Draw the updated graph
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, edge_color=edge_color_map, node_size=800, font_color='white')

    # Show the graph
    plt.show(block=False)
    print("T = " + str(timestep))
    timestep += 1

    # Wait for user input to proceed to the next step
    input("Press Enter to proceed to the next timestep...")