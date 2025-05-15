#!/usr/bin/env python3
# Combined Irvine parameters with India-style execution approach

import torch
import os
import sys
import networkx as nx

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Current device index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
print()  # Separator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
from comparisons import Comparisons
from plotting import plot_trials

graph = ns.build_graph_from_edgelist(
    "graphs/irvine_reindexed.txt",
    value_low=1,
    value_high=2
)
pos = nx.spring_layout(graph)

algorithms = ['graph', 'dqn', 'hillclimb','whittle','none']
NUM_ACTIONS = 50
NUM_COMPARISONS = 50
CASCADE_PROB = 0.05
GAMMA = 0.8
TIMESTEPS = 30
TIMESTEP_INTERVAL = 5

comp = Comparisons(device=device)

print("Starting train_graph")
comp.train_graph(graph, NUM_ACTIONS, CASCADE_PROB, GAMMA)
print("Finished train_graph")

print("Starting train_whittle")
comp.train_whittle(graph, GAMMA)
print("Finished train_whittle")

print("Starting train_dqn")
comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB)
print("Finished train_dqn")

metadata = {
    "algorithms": algorithms,
    "initial_graph": graph,
    "num_comparisons": NUM_COMPARISONS,
    "num_actions": NUM_ACTIONS,
    "cascade_prob": CASCADE_PROB,
    "gamma": GAMMA,
    "timesteps": TIMESTEPS
}

results = comp.run_many_comparisons(
    algorithms=algorithms,
    initial_graph=graph,
    num_comparisons=NUM_COMPARISONS,
    num_actions=NUM_ACTIONS,
    cascade_prob=CASCADE_PROB,
    gamma=GAMMA,
    timesteps=TIMESTEPS,
    timestep_interval=TIMESTEP_INTERVAL,
    device=device
)

plot_trials(
    results,
    output_dir="results",
    plot_cumulative_for=("reward",),  # tuple of metrics to plot cumulatively
    file_prefix="comparison",
    metadata=metadata
)

print("Algorithms:", algorithms)
print(f"Num comparisons: {NUM_COMPARISONS}")
print(f"Num actions: {NUM_ACTIONS}")
print(f"Cascade prob: {CASCADE_PROB}")
print(f"Gamma: {GAMMA}")
print("Num nodes:", len(graph.nodes))
print("Num edges:", len(graph.edges))