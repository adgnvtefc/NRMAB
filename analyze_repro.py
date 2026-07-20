#!/usr/bin/env python3
"""Final reproduction analysis: compare the paper's 20 seeds (results/history)
against fresh locally-trained seeds (results_repro/history), and run the
significance tests both ways (paper's t=30 snapshot vs plateau mean t>=15).
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats

ALGOS = ["cdsqn", "dqn", "whittle", "hillclimb", "random", "none"]

def load(dirpath):
    return [pd.read_csv(f) for f in sorted(glob.glob(dirpath + "/comparison_metrics_*.csv"))]

def per_seed(dfs, a, mode):
    col = f"{a}_reward_mean"
    if mode == "t30":
        return np.array([d[col][d.timestep == 30].values[0] for d in dfs])
    return np.array([d[col][d.timestep >= 15].mean() for d in dfs])

paper = load("real_data_trials/results/history")
repro = load("real_data_trials/results_repro/history")
print(f"paper seeds: {len(paper)}   repro seeds: {len(repro)}\n")

for mode, desc in [("plateau", "reward, plateau mean t>=15"), ("t30", "reward at t=30 (paper's test)")]:
    print(f"=== {desc} ===")
    print(f"{'algo':10s} {'paper':>8s} {'repro':>8s}   {'KS p':>7s}")
    for a in ALGOS:
        pv, rv = per_seed(paper, a, mode), per_seed(repro, a, mode)
        ks = stats.ks_2samp(pv, rv).pvalue  # same distribution?
        print(f"{a:10s} {pv.mean():8.2f} {rv.mean():8.2f}   {ks:7.3f}")
    for name, dfs in [("paper", paper), ("repro", repro)]:
        base = per_seed(dfs, "cdsqn", mode)
        line = [f"{name} paired-t cdsqn vs:"]
        for a in ["dqn", "whittle", "hillclimb", "none"]:
            p = stats.ttest_rel(base, per_seed(dfs, a, mode)).pvalue
            line.append(f"{a}={p:.4g}")
        print("  " + "  ".join(line))
    print()
