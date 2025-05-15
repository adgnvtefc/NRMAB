#!/usr/bin/env python3
# India-style comparison script for DQN vs. Tabular DQN

import torch
import sys
import os
# 1. Check CUDA availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Current device index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
print()

# 2. Dependencies and path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials

# 3. Initialize a random graph (India style uses on-the-fly generation)
G = ns.init_random_graph(
    num_nodes=10,
    num_edges=20,
    value_low=1,
    value_high=2
)
print(f"Initialized random graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

# 4. Parameters (retain your original settings)
algorithms = ['dqn', 'tabular', 'graph']
NUM_COMPARISONS = 10
NUM_ACTIONS = 2
CASCADE_PROB = 0.05
GAMMA = 0.8
TIMESTEPS = 30
TIMESTEP_INTERVAL = 5

# 5. Initialize Comparisons object
comp = Comparisons(device=device)
comp.train_graph(G, NUM_ACTIONS, CASCADE_PROB, GAMMA)
comp.train_dqn(G, NUM_ACTIONS, CASCADE_PROB)
comp.train_tabular(G, NUM_ACTIONS, GAMMA)

# 6. Execute comparisons
print("Starting run_many_comparisons...")
results = comp.run_many_comparisons(
    algorithms=algorithms,
    initial_graph=G,
    num_comparisons=NUM_COMPARISONS,
    num_actions=NUM_ACTIONS,
    cascade_prob=CASCADE_PROB,
    gamma=GAMMA,
    timesteps=TIMESTEPS,
    timestep_interval=TIMESTEP_INTERVAL,
    device=device
)
print("Completed comparisons.")

# 7. Plot results and save to results/
print("Plotting results...")
plot_trials(
    results,
    output_dir="results",
    plot_cumulative_for=("reward",),  # tuple of metrics to plot cumulatively
    file_prefix="comparison"
)
print("Plots saved to results/ with prefix 'comparison'.")
