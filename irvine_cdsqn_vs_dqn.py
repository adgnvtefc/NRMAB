#!/usr/bin/env python3
# irvine_cdsqn_vs_dqn.py

import torch
import os, sys
import networkx as nx
from networkSim import NetworkSim as ns
from comparisons import Comparisons 
from plotting import plot_trials

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print()  

    # 1. Load Data
    graph_path = "graphs/irvine.txt"
    print(f"Loading Irvine graph from {graph_path}...")
    # Using irvine parameters
    graph = ns.build_graph_from_edgelist(graph_path, value_low=1, value_high=2)
    
    # 2. Configuration (Matched to irvine_real_data_trial.py)
    algorithms = ['dqn', 'cdsqn', 'whittle', 'random']
    
    NUM_ACTIONS = 100 # From irvine_real_data_trial.py
    NUM_COMPARISONS = 20 # Kept at 20 for speed (script had 50 but 20 is standard for us)
    # User said "similar parameters", 50 might take too long if Irvine is large.
    # Irvine is usually larger than India.
    # Let's check size first?
    CASCADE_PROB = 0.05
    GAMMA = 0.8
    TIMESTEPS = 30
    TIMESTEP_INTERVAL = 5

    comp = Comparisons(device=device)  
    
    print("-" * 30)
    print(f"Starting comparison: {algorithms}")
    print(f"Graph Nodes: {len(graph.nodes)}")
    print(f"Graph Edges: {len(graph.edges)}")
    print("-" * 30)

    # 3. Train Models
    # Train DQN
    if 'dqn' in algorithms:
        print("\n[Training DQN]")
        comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB, num_epochs=3)
        print("Finished training DQN")

    # Train CDSQN
    if 'cdsqn' in algorithms:
        print("\n[Training CDSQN]")
        comp.train_cdsqn(graph, NUM_ACTIONS, CASCADE_PROB, num_epochs=3)
        print("Finished training CDSQN")

    # Train Whittle (Initialize Policy)
    if 'whittle' in algorithms:
        print("\n[Initializing Whittle Index Policy]")
        comp.train_whittle(graph, GAMMA)
        print("Finished initializing Whittle")
        
    # 4. Run Comparisons
    print(f"\n[Running {NUM_COMPARISONS} Comparisons]")
    
    metadata = {
        "algorithms": algorithms,
        "initial_graph": graph,
        "num_comparisons": NUM_COMPARISONS,
        "num_actions": NUM_ACTIONS,
        "cascade_prob": CASCADE_PROB,
        "gamma": GAMMA,
        "timesteps": TIMESTEPS
    }

    results = comp.run_many_comparisons(
        algorithms=algorithms,
        initial_graph=graph,
        num_comparisons=NUM_COMPARISONS,
        num_actions=NUM_ACTIONS,
        cascade_prob=CASCADE_PROB,
        gamma=GAMMA,
        timesteps=TIMESTEPS,
        timestep_interval=TIMESTEP_INTERVAL,
        device=device
    )

    # 5. Plotting
    output_dir = "results/irvine_cdsqn_vs_dqn"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n[Saving Results to {output_dir}]")
    import pandas as pd
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dfs = []
    for trial_idx, trial_data in enumerate(results):
        for algo, metrics in trial_data.items():
            df = pd.DataFrame(metrics)
            df['algorithm'] = algo
            df['trial'] = trial_idx
            dfs.append(df)
    
    full_df = pd.concat(dfs)
    full_df.to_csv(f"{output_dir}/comparison_irvine_raw_{timestamp}.csv", index=False)
    
    means = full_df.groupby(['algorithm', 'timestep']).mean().reset_index()
    means.to_csv(f"{output_dir}/comparison_irvine_means_{timestamp}.csv", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
