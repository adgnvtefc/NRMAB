#!/usr/bin/env python3
"""THE structural-advantage experiment: train a nodewise+dueling SQN on the
202-node India graph, evaluate ZERO-SHOT on the 1,899-node Irvine graph.
A DQN cannot do this at all (input dim is tied to n). Baselines on Irvine:
retrained DQN (skyline), hillclimb (needs full model), whittle, random, none.
Same regime on both graphs: pp (0,0.005), cascade 0.02, values 1-10, k=20.
Results -> real_data_trials/results_cross_graph/history/.
"""
import sys, copy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials
from algorithms.cdsqn import train_cdsqn_agent

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
HARD = {"aa": (0.85, 0.95), "ap": (0.2, 0.5), "pa": (0.7, 0.95), "pp": (0.0, 0.005)}
K = 20
CASCADE = 0.02
GAMMA = 0.8

TRAIN = dict(num_epochs=5, step_per_epoch=1000, gamma=GAMMA, update_per_step=0.25,
             eps_decay_steps=3500, eps_final=0.05, max_episode_steps=50,
             guided_exploration=True, nodewise=True, dueling=True)

for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    g_train = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=10, prob_ranges=HARD)
    cfg = {"graph": copy.deepcopy(g_train), "num_nodes": len(g_train.nodes),
           "cascade_prob": CASCADE, "stop_percent": 0.90,
           "reward_function": "normal", "reward_scale": 100.0}
    print("Training SQN on India (n=202)...", flush=True)
    sqn, _ = train_cdsqn_agent(cfg, K, learning_rate=3e-4, **TRAIN)

    g_test = ns.build_graph_from_edgelist("graphs/Irvine.txt", value_low=1, value_high=10, prob_ranges=HARD)
    comp = Comparisons(device=device)
    comp.train_whittle(g_test, GAMMA)  # baselines get the real test graph
    print("Training reference DQN on Irvine (n=1899)...", flush=True)
    comp.train_dqn(g_test, K, CASCADE, lr=3e-4,
                   **{k: v for k, v in TRAIN.items() if k not in ('nodewise', 'dueling')})
    comp.models['cdsqn'] = sqn.to(device)  # zero-shot transfer, no retraining

    results = comp.run_many_comparisons(
        algorithms=['cdsqn', 'dqn', 'hillclimb', 'whittle', 'random', 'none'],
        initial_graph=g_test, num_comparisons=30, num_actions=K,
        cascade_prob=CASCADE, gamma=GAMMA, timesteps=30, timestep_interval=5,
        device=device)
    plot_trials(results, output_dir="real_data_trials/results_cross_graph",
                plot_cumulative_for=("reward",), file_prefix="comparison")
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
