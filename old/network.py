from networkSim import NetworkSim as ns
import networkx as nx
import matplotlib.pyplot as plt

DEFAULT_ACTIVATION_CHANCE = 0.0
DEFAULT_DEACTIVATION_CHANCE = 0.2

# Initialize the graph
G = ns.init_random_graph(40, 100, DEFAULT_ACTIVATION_CHANCE, DEFAULT_DEACTIVATION_CHANCE)

first_itr = True
next_states = []

pos = nx.spring_layout(G)  # Positioning of nodes
timestep = 0
# Start simulation loop
while True:
    plt.clf()  # Clear the previous plot


    if not first_itr:
        # Update node activity based on next_states
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['obj'].active = next_states[i]
    else:
        first_itr = False

    ns.random_seed(G, 2)

    # Update node and edge colors
    node_color_map = ns.color_nodes(G)
    edge_color_map = ns.color_edges(G)

    # Draw the updated graph
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, edge_color=edge_color_map, node_size=800, font_color='white')

    # Show the graph
    plt.show(block=False)
    print("T = " + str(timestep))
    timestep += 1

    # Calculate the next node states
    next_states = ns.calculate_next_node_states(G)

    # Wait for user input to proceed to the next step
    input("Press Enter to proceed to the next timestep...")