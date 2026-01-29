#!/usr/bin/env python3
import time
import torch
import numpy as np
import pandas as pd
from networkSim import NetworkSim as ns
from comparisons import Comparisons

def benchmark():
    device = torch.device("cpu")
    comp = Comparisons(device=device)
    
    nodes_range = list(range(2, 12))
    num_actions = 2
    cascade_prob = 0.05
    gamma = 0.8
    epochs = 3
    
    results = []
    
    for n in nodes_range:
        print(f"\nBenchmarking size: {n} nodes...")
        G = ns.init_random_graph(num_nodes=n, num_edges=max(n, n*(n-1)//4), value_low=1, value_high=2)
        
        # Benchmark Tabular
        start = time.time()
        comp.train_tabular(G, num_actions, gamma)
        tabular_time = time.time() - start
        
        # Benchmark DQN
        start = time.time()
        comp.train_dqn(G, num_actions, cascade_prob, num_epochs=epochs)
        dqn_time_per_epoch = (time.time() - start) / epochs
        
        # Benchmark CDSQN
        start = time.time()
        comp.train_cdsqn(G, num_actions, cascade_prob, num_epochs=epochs)
        cdsqn_time_per_epoch = (time.time() - start) / epochs
        
        results.append({
            'nodes': n,
            'tabular': tabular_time,
            'dqn': dqn_time_per_epoch,
            'cdsqn': cdsqn_time_per_epoch
        })
        
        print(f"  Tabular: {tabular_time:.4f}s total")
        print(f"  DQN:     {dqn_time_per_epoch:.4f}s/epoch")
        print(f"  CDSQN:   {cdsqn_time_per_epoch:.4f}s/epoch")

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df.to_string(index=False))
    
    # Print in a format easy to copy-paste into plot_ccost.py
    print("\nCopy-paste arrays for plot_ccost.py:")
    print(f"nodes = {df['nodes'].tolist()}")
    print(f"tab_time = {[round(x, 4) for x in df['tabular'].tolist()]}")
    print(f"dqn_time = {[round(x, 4) for x in df['dqn'].tolist()]}")
    print(f"cdsqn_time = {[round(x, 4) for x in df['cdsqn'].tolist()]}")

if __name__ == "__main__":
    benchmark()
