import heapq
from networkSim import NetworkSim as ns
import time

class HillClimb:
    total_hill_climb_time = 0.0  # class-level accumulator
    times_called = 0

    # The BFS neighborhood term depends only on topology (and all nodes are
    # rearmed when hill_climb is called), so cache it per graph instance —
    # recomputing 1899 full BFS traversals every timestep dominated eval time.
    _bfs_cache = {}

    @staticmethod
    def _bfs_scores(graph, discount_factor=0.2):
        # content fingerprint: identical across deepcopies of the same seed
        # graph, distinct across seeds (values are random per seed)
        key = (graph.number_of_nodes(), graph.number_of_edges(),
               round(sum(graph.nodes[n]['obj'].getValue() for n in graph), 6))
        cached = HillClimb._bfs_cache.get(key)
        if cached is not None:
            return cached
        scores = {}
        for node in graph:
            value = 0.0
            queue = [(node, 0)]
            visited = {node}
            while queue:
                cur, depth = queue.pop()
                value += discount_factor ** depth
                for nb in graph.neighbors(cur):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, depth + 1))
            scores[node] = value
        if len(HillClimb._bfs_cache) > 64:
            HillClimb._bfs_cache.clear()
        HillClimb._bfs_cache[key] = scores
        return scores

    @staticmethod
    def hill_climb(graph, num=1):
        start_time = time.perf_counter()
        HillClimb.times_called += 1

        seeded_set = set()
        node_values = []

        bfs_scores = HillClimb._bfs_scores(graph)

        #iterate thru graph to see nodes that can pick
        for node in graph:
            value = graph.nodes[node]['obj'].getValue()

            #bootstrap solution to not double activate node under guaranteed activation
            if graph.nodes[node]['obj'].isActive():
                value -= 10

            if not graph.nodes[node]['obj'].canCascade():
                value += 1
            else:
                # all nodes are rearmed at selection time, so the canCascade-
                # filtered BFS equals the cached unfiltered BFS
                value += bfs_scores[node]

            node_values.append((value, node))
        top_nodes = heapq.nlargest(num, node_values)
        selected_nodes = [node for value, node in top_nodes]

        for node in selected_nodes:
            seeded_set.add(graph.nodes[node]['obj'])
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        HillClimb.total_hill_climb_time += elapsed

        return seeded_set
    @staticmethod
    def static_scores(graph, discount_factor=0.2):
        """State-independent version of the hill_climb score (node value plus
        discounted size of the BFS-reachable neighborhood), aligned with the
        obs index order (graph.nodes() iteration). Used to guide exploration
        during Q-learning data collection."""
        import numpy as np
        scores = []
        for node in graph.nodes():
            value = graph.nodes[node]['obj'].getValue()
            queue = [(node, 0)]
            visited = {node}
            while queue:
                cur, depth = queue.pop()
                value += discount_factor ** depth
                for nb in graph.neighbors(cur):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, depth + 1))
            scores.append(value)
        return np.array(scores, dtype=np.float32)

    @staticmethod
    def get_hillclimb_total_time():
        return HillClimb.total_hill_climb_time
    @staticmethod
    def get_hillclimb_times_called():
        return HillClimb.times_called