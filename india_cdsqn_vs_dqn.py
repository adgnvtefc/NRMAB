#!/usr/bin/env python3
# india_cdsqn_vs_dqn.py

import torch
import os, sys
import networkx as nx
from networkSim import NetworkSim as ns
from comparisons import Comparisons 
from plotting import plot_trials

# Ensure we can import modules from slightly deep structure if needed, 
# though usually running from root is fine.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Current device index:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
    print()  

    # 1. Load Data
    print("Loading India graph...")
    graph = ns.build_graph_from_edgelist("graphs/India.txt", value_low=1, value_high=2)
    # pos = nx.spring_layout(graph) # Not strictly needed unless visualizing graph structure itself

    # 2. Configuration
    algorithms = ['dqn', 'cdsqn']
    
    # Matching style of india_real_data_trial.py but possibly adjusting for runtime
    NUM_ACTIONS = 10
    NUM_COMPARISONS = 20 # Slightly reduced from 50 for faster turnaround, but statistically significant
    CASCADE_PROB = 0.05
    GAMMA = 0.95
    TIMESTEPS = 20
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
        comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB)
        print("Finished training DQN")

    # Train CDSQN
    if 'cdsqn' in algorithms:
        print("\n[Training CDSQN]")
        comp.train_cdsqn(graph, NUM_ACTIONS, CASCADE_PROB)
        print("Finished training CDSQN")

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
    output_dir = "results/india_cdsqn_vs_dqn"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n[Saving Results to {output_dir}]")
    plot_trials(
        results,
        output_dir=output_dir,
        plot_cumulative_for=("reward",),
        file_prefix="comparison_india",
        metadata=metadata
    )
    
    # Save a simple summary text
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(f"Algorithms: {algorithms}\n")
        f.write(f"Comparisons: {NUM_COMPARISONS}\n")
        f.write(f"Timesteps: {TIMESTEPS}\n")
        f.write(f"Num Actions: {NUM_ACTIONS}\n")
        # Could add average rewards here if extracted from results

    print("Done!")

if __name__ == "__main__":
    main()
