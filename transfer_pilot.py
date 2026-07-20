#!/usr/bin/env python3
"""ZERO-SHOT TRANSFER: train on India graph with drawn attributes (HARD2),
evaluate on the SAME topology with freshly drawn attributes. DQN sees only the
202-bit state (its per-index memorization goes stale); CDSQN reads node
features (values/probs/structure) and should transfer.
Whittle stays stale (trained on G_train) = "needs per-instance retraining".
hillclimb uses the true test graph = skyline.
Results -> real_data_trials/results_transfer/history/.
"""
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
algorithms = ['dqn', 'cdsqn', 'hillclimb', 'whittle', 'random', 'none']
HARD = {"aa": (0.85, 0.95), "ap": (0.2, 0.5), "pa": (0.7, 0.95), "pp": (0.0, 0.02)}
NUM_ACTIONS = 10
NUM_COMPARISONS = 50
CASCADE_PROB = 0.10
GAMMA = 0.8
TIMESTEPS = 30

TRAIN = dict(
    num_epochs=5, step_per_epoch=1000, gamma=GAMMA, update_per_step=0.25,
    eps_decay_steps=3500, eps_final=0.05, max_episode_steps=50,
    guided_exploration=True,
)

for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    g_train = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=10, prob_ranges=HARD)
    comp = Comparisons(device=device)
    comp.train_whittle(g_train, GAMMA)  # deliberately stale at eval
    comp.train_dqn(g_train, NUM_ACTIONS, CASCADE_PROB, lr=3e-4, **TRAIN)
    comp.train_cdsqn(g_train, NUM_ACTIONS, CASCADE_PROB, learning_rate=3e-4, **TRAIN)

    # fresh attributes, same topology — models are NOT retrained
    g_test = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=10, prob_ranges=HARD)
    results = comp.run_many_comparisons(
        algorithms=algorithms,
        initial_graph=g_test,
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
        output_dir="real_data_trials/results_transfer",
        plot_cumulative_for=("reward",),
        file_prefix="comparison"
    )
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
