#!/usr/bin/env python3
# tune_cdsqn.py

import torch
import numpy as np
import networkx as nx
from networkSim import NetworkSim as ns
import copy
import itertools
from algorithms.cdsqn import train_cdsqn_agent

def run_tuning():
    print("CUDA available:", torch.cuda.is_available())
    
    # 1. Load Graph
    graph_path = "graphs/India.txt"
    print(f"Loading India graph from {graph_path}...")
    graph = ns.build_graph_from_edgelist(graph_path, value_low=1, value_high=2)
    print(f"Graph Nodes: {len(graph.nodes)}")
    print(f"Graph Edges: {len(graph.edges)}")
    
    # 2. Parameters
    NUM_ACTIONS = 10
    CASCADE_PROB = 0.05
    NUM_EPOCHS = 5 # Intermediate epochs to check stability
    
    # Tuning Grid
    learning_rates = [1e-4, 1e-5]
    hidden_dims = [64, 128]
    reward_scales = [100.0, 200.0]
    
    combinations = list(itertools.product(learning_rates, hidden_dims, reward_scales))
    print(f"\nTotal Tuning Combinations: {len(combinations)}")
    
    results = []
    
    for i, (lr, hidden, scale) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing Config: LR={lr}, Hidden={hidden}, Scale={scale}")
        
        config = {
            "graph": copy.deepcopy(graph),
            "num_nodes": len(graph.nodes),
            "cascade_prob": CASCADE_PROB,
            "stop_percent": 0.90,
            "reward_function": "normal",
            "reward_scale": scale
        }
        
        try:
            # We don't have easy access to final loss/reward from train_cdsqn_agent directly
            # without modifying it to return metrics.
            # But the agent training prints logs.
            # Ideally we'd modify train_cdsqn_agent to return the final logs, 
            # but for now we observe if it crashes or explodes.
            # Actually, `train_cdsqn_agent` returns `model, policy`.
            # We can capture stdout or just rely on the training output.
            
            # To assess "quality", we should technically run a small validation eval.
            
            model, policy = train_cdsqn_agent(
                config, 
                num_actions=NUM_ACTIONS, 
                num_epochs=NUM_EPOCHS, 
                step_per_epoch=500, # Smaller steps for tuning speed
                hidden_dim=hidden,
                learning_rate=lr
            )
            
            print(f"--> [SUCCESS] Config {i+1} finished.")
            results.append({
                "config": (lr, hidden, scale),
                "status": "Success"
            })
            
            # Optional: Run a quick eval? 
            # Not strictly necessary if we just check for loss explosion via logs.
            
        except Exception as e:
            print(f"--> [FAILED] Config {i+1} Crashed: {e}")
            results.append({
                "config": (lr, hidden, scale),
                "status": f"Failed: {e}"
            })

    print("\n--- Tuning Summary ---")
    for res in results:
        print(f"Config {res['config']} : {res['status']}")

if __name__ == "__main__":
    run_tuning()
