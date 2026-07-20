#!/usr/bin/env python3
"""Pilot: does CDSQN show a real advantage over DQN once both are actually trained?

Fixes applied to BOTH models (identical budgets, fair comparison):
  - epsilon decays over the real training length (was: stuck at ~0.97)
  - episodes truncated at 50 steps so training sees fresh starts (was: never reset)
  - update_per_step 0.25 (was: 0.1 -> only ~150 gradient steps)
  - gamma 0.8 to match the paper's MDP objective (was: 0.99 -> Q targets ~230)
  - lr 3e-4 for both (was: DQN 1e-5, CDSQN 1e-4)
  - 5 epochs x 1000 steps = 5000 env steps (was: 1500)

Evaluation protocol is unchanged from the paper (50 runs x 30 timesteps).
Results go to real_data_trials/results_improved/history/.
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
NUM_ACTIONS = 10
NUM_COMPARISONS = 50
CASCADE_PROB = 0.05
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
    graph = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=2)
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
        output_dir="real_data_trials/results_improved",
        plot_cumulative_for=("reward",),
        file_prefix="comparison"
    )
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
