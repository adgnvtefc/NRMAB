import networkx as nx
import random
from networkSim import NetworkSim  as ns
from networkvis import NetworkVis as nv
#add: graph neural network

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparisons import Comparisons 
from plotting import plot_trials

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

    random_nodes = ns.generate_random_nodes(num_nodes, value_low, value_high)

    G = nx.Graph()

    node_id_map = {}  # real_node_id -> index in [0..num_nodes-1]
    for idx, real_node_id in enumerate(unique_nodes):
        node_id_map[real_node_id] = idx
        # `random_nodes[idx]` is a dict: {"obj": <Node>}
        G.add_node(real_node_id, obj=random_nodes[idx]["obj"])

    for (s, d) in edges:
        G.add_edge(s, d)

    return G

graph = build_graph_from_edgelist("./graphs/India.txt", value_low=1, value_high=2)
pos = nx.spring_layout(graph)  # Positioning of nodes


algorithms = ['dqn', 'hillclimb', 'none']
NUM_ACTIONS = 30

comp = Comparisons()
comp.train_dqn(graph, NUM_ACTIONS, 0.05)

results = (comp.run_many_comparisons(
    algorithms=algorithms, 
    initial_graph=graph, 
    num_comparisons=10, 
    num_actions=NUM_ACTIONS,
    cascade_prob=0.05, 
    gamma=0.8, 
    timesteps=30, 
    timestep_interval=5))

plot_trials(
    results, 
    output_dir="results", 
    plot_cumulative_for=("reward",),  # tuple of metrics you want to also plot cumulatively
    file_prefix="comparison")         # prefix for output filed