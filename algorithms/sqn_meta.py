"""Meta-MDP training for the (CD)SQN — a direct implementation of the paper's
multi-Bellman operator (Definition 3.3): each meta-step picks ONE node given the
current partial action set; after k picks the graph transitions. This trains
Q(s, partial set + candidate) exactly on the distribution the greedy
hill-climbing evaluation queries, unlike the standard trainer which only ever
sees full k-sets.

Uses the same CDSQN model class, so `select_action_cdsqn` works unchanged for
evaluation. gamma_tilde = gamma ** (1/k), rewards are the paper's scaled
marginal rewards R~ so the k-step discounted sum telescopes to R(s,A) + gamma V(s').
"""
import copy
import numpy as np
import torch

from networkSim import NetworkSim as ns
from algorithms.cdsqn import CDSQN
from algorithms.cdsqn_env import convert_nx_to_pyg


class MetaReplay:
    def __init__(self, capacity, n_nodes, n_feat):
        self.cap = capacity
        self.x = np.zeros((capacity, n_nodes, n_feat), dtype=np.float32)
        self.xn = np.zeros((capacity, n_nodes, n_feat), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, x, a, r, xn, d):
        i = self.idx
        self.x[i], self.act[i], self.rew[i], self.xn[i], self.done[i] = x, a, r, xn, d
        self.idx = (i + 1) % self.cap
        self.full = self.full or self.idx == 0

    def sample(self, n):
        hi = self.cap if self.full else self.idx
        j = np.random.randint(0, hi, size=n)
        return self.x[j], self.act[j], self.rew[j], self.xn[j], self.done[j]

    def __len__(self):
        return self.cap if self.full else self.idx


def _obs_features(graph, partial_set, t_graph):
    """Node feature matrix [N, 10]; feature 6 marks the current partial set."""
    return convert_nx_to_pyg(graph, last_action_indices=partial_set,
                             current_step=t_graph).x.numpy()


def _q_all(model, x_np, edge_index, device):
    """Q values of adding each single node, given features x_np [B, N, F]
    (partial set encoded in feature 6). Returns [B, N]."""
    B, N, F = x_np.shape
    x_flat = torch.tensor(x_np.reshape(-1, F), dtype=torch.float32, device=device)
    batch_idx = torch.arange(B, device=device).repeat_interleave(N)
    ei = torch.cat([edge_index + i * N for i in range(B)], dim=1)
    w1, w2, w3 = model.get_weights(x_flat, ei, batch_idx)
    base = torch.tensor(x_np[:, :, 6], dtype=torch.float32, device=device)  # current set
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
    cands = torch.clamp(base.unsqueeze(1) + eye, 0, 1)  # [B, N, N]
    return model.compute_q(w1, w2, w3, cands)  # [B, N]


def train_sqn_meta(config, num_actions, graph_steps=5000, gamma=0.8,
                   lr=3e-4, batch_size=64, update_every=4, tau=0.005,
                   eps_decay_frac=0.7, eps_final=0.05, episode_len=50,
                   hidden_dim=64, buffer_cap=30000, device=None,
                   guided_exploration=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = num_actions
    g_t = gamma ** (1.0 / k)  # gamma tilde
    scale = config.get('reward_scale', 100.0)

    graph0 = config['graph']
    N = len(graph0.nodes())
    node_order = list(graph0.nodes())

    model = CDSQN(N, 10, hidden_dim, hidden_dim).to(device)
    target = copy.deepcopy(model)
    target.eval()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ei_np = convert_nx_to_pyg(copy.deepcopy(graph0)).edge_index.numpy()
    edge_index = torch.tensor(ei_np, dtype=torch.long, device=device)

    guide = None
    if guided_exploration:
        from algorithms.hillClimb import HillClimb
        guide = HillClimb.static_scores(graph0)

    buf = MetaReplay(buffer_cap, N, 10)
    total_meta = graph_steps * k
    eps_decay_steps = eps_decay_frac * total_meta

    graph = copy.deepcopy(graph0)
    ep_t = 0
    partial = []
    meta_step = 0
    losses = []

    obs = _obs_features(graph, partial, ep_t)
    for gs in range(graph_steps):
        for t in range(k):
            eps = max(eps_final, 1 - meta_step / eps_decay_steps)
            active = obs[:, 0] > 0.5
            in_set = obs[:, 6] > 0.5
            valid = ~(active | in_set)
            if not valid.any():
                break
            if np.random.random() < eps:
                if guide is not None and np.random.random() < 0.5:
                    masked = np.where(valid, guide, -np.inf)
                    a = int(np.argmax(masked))
                else:
                    a = int(np.random.choice(np.flatnonzero(valid)))
            else:
                with torch.no_grad():
                    q = _q_all(model, obs[None], edge_index, device)[0]
                q[torch.tensor(~valid, device=device)] = -float('inf')
                a = int(torch.argmax(q))

            node = node_order[a]
            marginal = graph.nodes[node]['obj'].getValue()
            state_val = sum(graph.nodes[n]['obj'].getValue() for n in graph.nodes()
                            if graph.nodes[n]['obj'].isActive()) if t == 0 else 0.0
            r = (state_val + marginal) / scale * (g_t ** (-t))

            partial.append(node)
            if t == k - 1:
                # apply the full set: transition the graph
                objs = [graph.nodes[n]['obj'] for n in partial]
                ns.passive_state_transition_without_neighbors(graph, exempt_nodes=objs)
                ns.active_state_transition(objs)
                ns.independent_cascade_allNodes(graph, config['cascade_prob'])
                ns.rearm_nodes(graph)
                partial = []
                ep_t += 1
                done = float(ep_t >= episode_len)
            else:
                done = 0.0

            obs_next = _obs_features(graph, partial, ep_t)
            buf.push(obs, a, r, obs_next, done)
            obs = obs_next
            meta_step += 1

            if len(buf) >= batch_size and meta_step % update_every == 0:
                bx, ba, br, bxn, bd = buf.sample(batch_size)
                bx_t = torch.tensor(bx, device=device)
                bxn_t = torch.tensor(bxn, device=device)
                q_all = _q_all(model, bx, edge_index, device)
                q_sa = q_all[torch.arange(batch_size), torch.tensor(ba, device=device)]
                with torch.no_grad():
                    qn = _q_all(target, bxn, edge_index, device)
                    invalid_n = (bxn_t[:, :, 0] > 0.5) | (bxn_t[:, :, 6] > 0.5)
                    qn = qn.masked_fill(invalid_n, -float('inf'))
                    qn_max = qn.max(dim=1).values
                    qn_max[~torch.isfinite(qn_max)] = 0.0
                    y = torch.tensor(br, device=device) + g_t * (1 - torch.tensor(bd, device=device)) * qn_max
                loss = torch.nn.functional.mse_loss(q_sa, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if ep_t >= episode_len:
            graph = copy.deepcopy(graph0)
            partial = []
            ep_t = 0
            obs = _obs_features(graph, partial, ep_t)
        if (gs + 1) % 500 == 0:
            recent = np.mean(losses[-100:]) if losses else float('nan')
            print(f"  sqn-meta: graph step {gs+1}/{graph_steps}, eps={max(eps_final, 1 - meta_step / eps_decay_steps):.2f}, loss={recent:.4f}", flush=True)

    return model
