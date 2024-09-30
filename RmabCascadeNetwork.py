from networkSim import NetworkSim as ns
import networkx as nx
import matplotlib.pyplot as plt

#activation chance in passive action
DEFAULT_ACTIVATION_CHANCE = 0.1
#deactivation chance in passive action
DEFAULT_DEACTIVATION_CHANCE = 0.2
#assume 100% activation chance in selection action

# Initialize the graph
G = ns.init_random_graph(40, 100, DEFAULT_ACTIVATION_CHANCE, DEFAULT_DEACTIVATION_CHANCE)

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
    (seeded_nodes, transition_nodes) = ns.seed_and_transition(G, ns.hill_climb, num=2)

    for node in seeded_nodes:
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

    ###add step here -- 'rearm' all nodes for cascade for the next timestep

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