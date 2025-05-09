import networkx as nx
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim  as ns
from networkvis import NetworkVis as nv
from comparisons import Comparisons 
from plotting import plot_trials

import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)  # Shows the installed CUDA version (if any)

# graph = ns.build_graph_from_edgelist("../graphs/irvine_reindexed.txt", value_low=1, value_high=2)
# use gpickle to save and load graph
with open("test.gpickle", "rb") as f:
    graph = pickle.load(f)

# Verify the loaded graph
print(graph.nodes(data=True))  # Outputs nodes and their attributes
print(graph.edges())  # Outputs edges

print(graph)
pos = nx.spring_layout(graph)  # Positioning of nodes
algorithms = ['graph', 'dqn', 'hillclimb', 'none']
#ALL PREVIOUS EXPERIMENTS RAN WITH 30 ACTIONS
NUM_ACTIONS = 50
NUM_COMPARISONS = 50
CASCADE_PROB = 0.05
GAMMA = 0.8
TIMESTEPS = 30
TIMESTEP_INTERVAL=5
comp = Comparisons()
comp.train_graph(graph, NUM_ACTIONS, CASCADE_PROB)
comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB)
# comp.train_whittle(graph, GAMMA)

metadata = {"algorithms": algorithms,
            "initial_graph": graph,
            "num_comparisons": NUM_COMPARISONS,
            "num_actions": NUM_ACTIONS,
            "cascade_prob": CASCADE_PROB,
            "gamma": GAMMA,
            "timesteps": TIMESTEPS}

results = (comp.run_many_comparisons(
    algorithms=algorithms, 
    initial_graph=graph, 
    num_comparisons=NUM_COMPARISONS, 
    num_actions=NUM_ACTIONS,
    cascade_prob=CASCADE_PROB, 
    gamma=GAMMA, 
    timesteps=TIMESTEPS, 
    timestep_interval=TIMESTEP_INTERVAL))

plot_trials(
    results, 
    output_dir="results", 
    plot_cumulative_for=("reward",),  # tuple of metrics you want to also plot cumulatively
    file_prefix="comparison",
    metadata=metadata)         # prefix for output filed

print(algorithms)
print(f"num comparisons: {NUM_COMPARISONS}")
print(f"num actions: {NUM_ACTIONS}")
print(f"cascade prob: {CASCADE_PROB}")
print(len(graph.nodes))
print(len(graph.edges))