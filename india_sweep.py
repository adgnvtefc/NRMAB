#!/usr/bin/env python3
"""Full SQN recipe (nodewise + dueling + sum-heads + stable distillation +
guided TD) on the PAPER'S ORIGINAL India setting (cascade 0.05, values 1-2,
k=10, eval 50 runs x 30 steps) vs improved-trained DQN.
Results -> real_data_trials/results_india_sqnv2/history/.
"""
import sys, copy, random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

from networkSim import NetworkSim as ns
from comparisons import Comparisons
from plotting import plot_trials
from algorithms.cdsqn import CDSQN, train_cdsqn_agent
from algorithms.cdsqn_env import convert_nx_to_pyg
from algorithms.hillClimb import HillClimb

NUM_SEEDS = int(sys.argv[1])
CFG_NAME, TD_LR, HID, UPS = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
TRAIN_DIR = 'real_data_trials/results_india_sweep_' + CFG_NAME
K = 10
CASCADE = 0.05
GAMMA = 0.8
TRAIN = dict(num_epochs=5, step_per_epoch=1000, gamma=GAMMA, update_per_step=UPS,
             eps_decay_steps=3500, eps_final=0.05, max_episode_steps=50,
             guided_exploration=True)


def pretrain_distill(graph, n_states=200, epochs=300, batch=8, lr=1e-3):
    N = len(graph.nodes())
    model = CDSQN(N, 10, HID, HID, nodewise=True, dueling=True).to(device)
    bfs = HillClimb._bfs_scores(graph)
    xs, ys = [], []
    g = copy.deepcopy(graph)
    for i in range(n_states):
        xs.append(convert_nx_to_pyg(g).x.numpy())
        sc = []
        for n in g.nodes():
            o = g.nodes[n]['obj']
            v = o.getValue() + bfs[n]
            if o.isActive():
                v -= 10
            sc.append(v)
        sc = np.array(sc, dtype=np.float32)
        lo, hi = sc.min(), sc.max()
        ys.append((sc - lo) / max(hi - lo, 1e-6) * 3.0)
        picks = random.sample(range(N), K)
        objs = [g.nodes[n]['obj'] for n in picks]
        ns.passive_state_transition_without_neighbors(g, exempt_nodes=objs)
        ns.active_state_transition(objs)
        ns.independent_cascade_allNodes(g, CASCADE)
        ns.rearm_nodes(g)
        if (i + 1) % 50 == 0:
            g = copy.deepcopy(graph)
    xs = np.stack(xs); ys = np.stack(ys)
    ei = torch.tensor(convert_nx_to_pyg(copy.deepcopy(graph)).edge_index.numpy(),
                      dtype=torch.long, device=device)
    eye = torch.eye(N, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = np.random.permutation(n_states)
        for j in range(0, n_states, batch):
            idx = perm[j:j + batch]
            B = len(idx)
            x = torch.tensor(xs[idx].reshape(-1, 10), dtype=torch.float32, device=device)
            bidx = torch.arange(B, device=device).repeat_interleave(N)
            eib = torch.cat([ei + i * N for i in range(B)], dim=1)
            w1, w2, w3 = model.get_weights(x, eib, bidx)
            g_single = model.compute_q(w1, w2, w3, eye.unsqueeze(0).expand(B, N, N))
            y = torch.tensor(ys[idx], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(g_single, y)
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  distill done, final loss {loss.item():.4f}", flush=True)
    return model


for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    graph = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=2)
    comp = Comparisons(device=device)
    comp.train_whittle(graph, GAMMA)
    comp.train_dqn(graph, K, CASCADE, lr=3e-4, **TRAIN)
    model = pretrain_distill(graph)
    cfg = {"graph": copy.deepcopy(graph), "num_nodes": len(graph.nodes),
           "cascade_prob": CASCADE, "stop_percent": 0.90,
           "reward_function": "normal", "reward_scale": 100.0}
    model, _ = train_cdsqn_agent(cfg, K, learning_rate=TD_LR, model=model, **TRAIN)
    comp.models['cdsqn'] = model.to(device)
    results = comp.run_many_comparisons(
        algorithms=['cdsqn', 'dqn', 'hillclimb', 'whittle', 'random', 'none'],
        initial_graph=graph, num_comparisons=50, num_actions=K,
        cascade_prob=CASCADE, gamma=GAMMA, timesteps=30, timestep_interval=5,
        device=device)
    plot_trials(results, output_dir=TRAIN_DIR,
                plot_cumulative_for=("reward",), file_prefix="comparison")
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
