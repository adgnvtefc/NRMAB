#!/usr/bin/env python3
"""Pilot in the HARD2 regime: sparse spontaneous activation, fast decay,
moderate cascade — a setting where interventions and set-selection actually
matter (none ~280 vs 1-step hillclimb ~445 plateau reward).

Node prob ranges: aa (0.85,0.95), ap (0.2,0.5), pa (0.7,0.95), pp (0.0,0.02)
Cascade 0.10, node values 1-10, k=10. Improved training for both models.
Results -> real_data_trials/results_hard/history/.
"""
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
algorithms = ['dqn', 'cdsqn', 'hillclimb', 'whittle', 'random', 'none']
HARD = {"aa": (0.85, 0.95), "ap": (0.2, 0.5), "pa": (0.7, 0.95), "pp": (0.0, 0.02)}
NUM_ACTIONS = 10
NUM_COMPARISONS = 50
CASCADE_PROB = 0.10
GAMMA = 0.8
TIMESTEPS = 30

TRAIN = dict(
    num_epochs=5,
    step_per_epoch=1000,
    gamma=GAMMA,
    update_per_step=0.25,
    eps_decay_steps=3500,
    eps_final=0.05,
    max_episode_steps=50,
)

for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    graph = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=10, prob_ranges=HARD)
    comp = Comparisons(device=device)
    comp.train_whittle(graph, GAMMA)
    comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB, lr=3e-4, **TRAIN)
    comp.train_cdsqn(graph, NUM_ACTIONS, CASCADE_PROB, learning_rate=3e-4, **TRAIN)
    results = comp.run_many_comparisons(
        algorithms=algorithms,
        initial_graph=graph,
        num_comparisons=NUM_COMPARISONS,
        num_actions=NUM_ACTIONS,
        cascade_prob=CASCADE_PROB,
        gamma=GAMMA,
        timesteps=TIMESTEPS,
        timestep_interval=5,
        device=device
    )
    plot_trials(
        results,
        output_dir="real_data_trials/results_hard",
        plot_cumulative_for=("reward",),
        file_prefix="comparison"
    )
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
