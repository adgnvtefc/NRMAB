#!/usr/bin/env python3
"""Distillation-pretrained Dueling-SQN on Irvine: supervise G's singleton
values against the hillclimb score vector (direct within-state comparative
signal that TD never provides), then TD-fine-tune with guided exploration.
Results -> real_data_trials/results_irvine_final/history/.
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

NUM_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
algorithms = ['dqn', 'cdsqn', 'hillclimb', 'whittle', 'random', 'none']
HARD = {"aa": (0.85, 0.95), "ap": (0.2, 0.5), "pa": (0.7, 0.95), "pp": (0.0, 0.005)}
NUM_ACTIONS = 20
NUM_COMPARISONS = 30
CASCADE_PROB = 0.02
GAMMA = 0.8
TIMESTEPS = 30

TRAIN = dict(num_epochs=5, step_per_epoch=1000, gamma=GAMMA, update_per_step=0.25,
             eps_decay_steps=3500, eps_final=0.05, max_episode_steps=50,
             guided_exploration=True)


def hillclimb_scores(graph, bfs):
    s = []
    for n in graph.nodes():
        o = graph.nodes[n]['obj']
        v = o.getValue() + bfs[n]
        if o.isActive():
            v -= 10
        s.append(v)
    return np.array(s, dtype=np.float32)


def pretrain_distill(graph, n_states=200, epochs=60, batch=8, lr=1e-3):
    N = len(graph.nodes())
    model = CDSQN(N, 10, 64, 64, nodewise=True, dueling=True, linear_head=True).to(device)
    bfs = HillClimb._bfs_scores(graph)
    # collect states from random rollouts + score targets
    xs, ys = [], []
    g = copy.deepcopy(graph)
    for i in range(n_states):
        xs.append(convert_nx_to_pyg(g).x.numpy())
        sc = hillclimb_scores(g, bfs)
        lo, hi = sc.min(), sc.max()
        ys.append((sc - lo) / max(hi - lo, 1e-6) * 3.0)  # monotone rescale to [0,3]
        picks = random.sample(range(N), NUM_ACTIONS)
        objs = [g.nodes[n]['obj'] for n in picks]
        ns.passive_state_transition_without_neighbors(g, exempt_nodes=objs)
        ns.active_state_transition(objs)
        ns.independent_cascade_allNodes(g, CASCADE_PROB)
        ns.rearm_nodes(g)
        if (i + 1) % 50 == 0:
            g = copy.deepcopy(graph)  # fresh episode
    xs = np.stack(xs); ys = np.stack(ys)
    ei = torch.tensor(convert_nx_to_pyg(copy.deepcopy(graph)).edge_index.numpy(),
                      dtype=torch.long, device=device)
    eye = torch.eye(N, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = np.random.permutation(n_states)
        tot = 0.0
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
            tot += loss.item() * B
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                from scipy.stats import spearmanr
                rho = spearmanr(g_single[0].cpu().numpy(), ys[idx][0]).statistic
            print(f"  distill ep {ep+1}: loss {tot/n_states:.4f}, spearman {rho:.3f}", flush=True)
    return model




def pretrain_distill_dqn(graph, n_states=200, epochs=300, batch=8, lr=1e-3):
    from algorithms.deepq import QNet
    N = len(graph.nodes())
    model = QNet(N, N).to(device)
    bfs = HillClimb._bfs_scores(graph)
    xs, ys = [], []
    g = copy.deepcopy(graph)
    for i in range(n_states):
        xs.append(np.array([1.0 if g.nodes[n]['obj'].isActive() else 0.0 for n in g.nodes()], dtype=np.float32))
        sc = hillclimb_scores(g, bfs)
        lo, hi = sc.min(), sc.max()
        ys.append((sc - lo) / max(hi - lo, 1e-6) * 3.0)
        picks = random.sample(range(N), NUM_ACTIONS)
        objs = [g.nodes[n]['obj'] for n in picks]
        ns.passive_state_transition_without_neighbors(g, exempt_nodes=objs)
        ns.active_state_transition(objs)
        ns.independent_cascade_allNodes(g, CASCADE_PROB)
        ns.rearm_nodes(g)
        if (i + 1) % 50 == 0:
            g = copy.deepcopy(graph)
    xs = np.stack(xs); ys = np.stack(ys)
    eye = torch.eye(N, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = np.random.permutation(n_states)
        for j in range(0, n_states, batch):
            idx = perm[j:j + batch]
            loss_t = 0
            opt.zero_grad()
            for ii in idx:
                s = torch.tensor(xs[ii], device=device).unsqueeze(0).expand(N, N)
                q = model(s, eye).squeeze(-1)
                y = torch.tensor(ys[ii], device=device)
                l = torch.nn.functional.mse_loss(q, y)
                l.backward()
                loss_t += l.item()
            opt.step()
    print(f"  dqn distill done, loss {loss_t/len(idx):.4f}", flush=True)
    return model

for seed_i in range(NUM_SEEDS):
    print(f"=== SEED {seed_i + 1}/{NUM_SEEDS} ===", flush=True)
    graph = ns.build_graph_from_edgelist("graphs/Irvine.txt", value_low=1, value_high=10, prob_ranges=HARD)
    comp = Comparisons(device=device)
    comp.train_whittle(graph, GAMMA)
    dqn_model = pretrain_distill_dqn(graph)
    comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB, lr=1e-4, model=dqn_model, **TRAIN)
    print("Distillation pretraining...", flush=True)
    model = pretrain_distill(graph, epochs=300, lr=1e-3)
    cfg = {"graph": copy.deepcopy(graph), "num_nodes": len(graph.nodes),
           "cascade_prob": CASCADE_PROB, "stop_percent": 0.90,
           "reward_function": "normal", "reward_scale": 100.0}
    print("TD fine-tuning...", flush=True)
    model, _ = train_cdsqn_agent(cfg, NUM_ACTIONS, learning_rate=1e-4, model=model, **TRAIN)
    comp.models['cdsqn'] = model.to(device)
    results = comp.run_many_comparisons(
        algorithms=algorithms, initial_graph=graph, num_comparisons=NUM_COMPARISONS,
        num_actions=NUM_ACTIONS, cascade_prob=CASCADE_PROB, gamma=GAMMA,
        timesteps=TIMESTEPS, timestep_interval=5, device=device)
    plot_trials(results, output_dir="real_data_trials/results_irvine_final",
                plot_cumulative_for=("reward",), file_prefix="comparison")
    print(f"=== SEED {seed_i + 1} DONE ===", flush=True)

print("ALL SEEDS COMPLETE", flush=True)
