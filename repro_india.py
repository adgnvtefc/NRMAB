#!/usr/bin/env python3
"""Reproduction driver: same protocol as india_real_data_trial.py, but writes
each seed's results to real_data_trials/results_repro/ to keep the paper's
original history CSVs untouched. Run with:  python repro_india.py [num_seeds]
"""
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

import networkx as nx
from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 19
algorithms = ['dqn', 'cdsqn', 'hillclimb', 'whittle', 'random', 'none']
NUM_ACTIONS = 10
NUM_COMPARISONS = 50
CASCADE_PROB = 0.05
GAMMA = 0.8
TIMESTEPS = 30
TIMESTEP_INTERVAL = 5

for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    graph = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=2)
    comp = Comparisons(device=device)
    comp.train_whittle(graph, GAMMA)
    comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB)
    comp.train_cdsqn(graph, NUM_ACTIONS, CASCADE_PROB)
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
        output_dir="real_data_trials/results_repro",
        plot_cumulative_for=("reward",),
        file_prefix="comparison"
    )
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE")
