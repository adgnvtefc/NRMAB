from networkSim import NetworkSim as ns
import networkx as nx
import matplotlib.pyplot as plt

class NetworkVis:
    @staticmethod
    def do_things(G, graph_pos, seeded_nodes, obj_to_id_mapping, timestep=0):
        transition_nodes = ns.passive_state_transition_without_neighbors(G, exempt_nodes=[G.nodes[node_index]['obj'] for node_index in seeded_nodes])
        changed_nodes = ns.active_state_transition_graph_indices(G, seeded_nodes)

        for node in changed_nodes:
            node_id = obj_to_id_mapping.get(node)
            print(f"Node {node_id} is activated.")


        for node in transition_nodes:
            # Get the node identifier using the reverse mapping
            node_id = obj_to_id_mapping.get(node)
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
        nx.draw(G, graph_pos, with_labels=True, node_color=node_color_map, edge_color=edge_color_map, node_size=800, font_color='white')

        # Show the graph
        plt.show(block=False)
        print("T = " + str(timestep))
        timestep += 1

        # Wait for user input to proceed to the next step
        input("Press Enter to proceed to the next timestep...")
