#!/usr/bin/env python3
"""HARD2 regime: SQN trained via the paper's meta-MDP (multi-Bellman) vs
standard-trained DQN. The meta-trained model occupies the 'cdsqn' slot so the
unchanged evaluation pipeline applies.
Results -> real_data_trials/results_hard_meta/history/.
"""
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials
from algorithms.sqn_meta import train_sqn_meta

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
algorithms = ['dqn', 'cdsqn', 'hillclimb', 'whittle', 'random', 'none']
HARD = {"aa": (0.85, 0.95), "ap": (0.2, 0.5), "pa": (0.7, 0.95), "pp": (0.0, 0.02)}
NUM_ACTIONS = 10
NUM_COMPARISONS = 50
CASCADE_PROB = 0.10
GAMMA = 0.8
TIMESTEPS = 30

DQN_TRAIN = dict(
    num_epochs=5, step_per_epoch=1000, gamma=GAMMA, update_per_step=0.25,
    eps_decay_steps=3500, eps_final=0.05, max_episode_steps=50,
    guided_exploration=True,
)

for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    graph = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=10, prob_ranges=HARD)
    comp = Comparisons(device=device)
    comp.train_whittle(graph, GAMMA)
    comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB, lr=3e-4, **DQN_TRAIN)
    import copy as _copy
    cfg = {"graph": _copy.deepcopy(graph), "num_nodes": len(graph.nodes),
           "cascade_prob": CASCADE_PROB, "stop_percent": 0.90,
           "reward_function": "normal", "reward_scale": 100.0}
    print("Training SQN via meta-MDP...", flush=True)
    comp.models['cdsqn'] = train_sqn_meta(cfg, NUM_ACTIONS, graph_steps=8000, guided_exploration=True,
                                          gamma=GAMMA, device=device)
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
        output_dir="real_data_trials/results_hard_meta",
        plot_cumulative_for=("reward",),
        file_prefix="comparison"
    )
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
