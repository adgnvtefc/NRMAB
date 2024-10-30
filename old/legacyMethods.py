import random
class LegacyNetworkSimMethods:
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