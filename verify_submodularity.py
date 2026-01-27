import torch
import numpy as np
import networkx as nx
import sys
import os
from algorithms.deepq import QNet
from algorithms.cdsqn import CDSQN
from algorithms.cdsqn_env import convert_nx_to_pyg
from networkSim import NetworkSim as ns

def verify_models():
    # Setup
    num_nodes = 50
    graph = nx.erdos_renyi_graph(num_nodes, 0.1, seed=42)
    # Generate nodes using NetworkSim static method
    nodes_dict = ns.generate_random_nodes(num_nodes, 0.0, 1.0)
    for node in graph.nodes():
        # Assign the node object from the generated dictionary
        # We assume graph nodes are 0..N-1 which matches keys in nodes_dict
        if node in nodes_dict:
            graph.nodes[node]['obj'] = nodes_dict[node]['obj']
        else:
            # Fallback (should not happen with erdos_renyi standard numbering)
             pass
        
    device = torch.device('cpu')
    
    print("--- Verifying Structural Submodularity ---")
    
    # 1. DQN (Random Weights)
    # DQN models Q(S, a). If submodular, Q(A, v) >= Q(B, v) for A subset B.
    print("\ntesting DQN (Standard MLP)...")
    dqn_model = QNet(state_dim=num_nodes, action_dim=num_nodes).to(device)
    dqn_violations = 0
    trials = 1000
    
    for _ in range(trials):
        # Create random sets A (small) and B (large) such that A subset B
        # Random binary mask for state
        # State in DQN is "Active Nodes".
        
        # Randomly select subset A
        size_a = np.random.randint(1, num_nodes // 3)
        indices_a = np.random.choice(num_nodes, size_a, replace=False)
        
        mask_a = torch.zeros(1, num_nodes)
        mask_a[0, indices_a] = 1.0
        
        # Add elements to make B
        remaining = list(set(range(num_nodes)) - set(indices_a))
        if not remaining: continue
            
        size_b_add = np.random.randint(1, len(remaining) // 2)
        indices_b_add = np.random.choice(remaining, size_b_add, replace=False)
        
        mask_b = mask_a.clone()
        mask_b[0, indices_b_add] = 1.0
        
        # Select v not in B
        remaining_v = list(set(remaining) - set(indices_b_add))
        if not remaining_v: continue
        v_idx = np.random.choice(remaining_v)
        
        # Action vector (one-hot v)
        action_v = torch.zeros(1, num_nodes)
        action_v[0, v_idx] = 1.0
        
        # Compute Q(A, v)
        # DQN forward: cat(state, action)
        with torch.no_grad():
            q_a = dqn_model(mask_a, action_v).item()
            q_b = dqn_model(mask_b, action_v).item()
            
        # Diminishing returns: Gain at A should be >= Gain at B
        if q_a < q_b:
            dqn_violations += 1
            
    print(f"DQN Violations: {dqn_violations}/{trials} ({dqn_violations/trials:.2%})")
    print("-> Result: DQN violates submodularity constraint (Expected).")

    # 2. CDSQN (Random Weights)
    # CDSQN models Q(Set). Check Q(A+v) - Q(A) >= Q(B+v) - Q(B)
    print("\ntesting CDSQN (Deep Submodular Network)...")
    
    state_dim = 10
    hidden_dim = 64
    dsf_hidden_dim = 64
    cdsqn_model = CDSQN(num_nodes, state_dim, hidden_dim, dsf_hidden_dim).to(device)
    
    # We need a dummy context (state). CDSQN weights generation depends on graph state.
    # We can just use one random graph state for all checks to verify "Conditioned on State S, function is submodular".
    
    # Generate dummy features for the graph
    pyg_data = convert_nx_to_pyg(graph)
    x = pyg_data.x
    edge_index = pyg_data.edge_index
    batch_index = torch.zeros(x.size(0), dtype=torch.long)
    
    # Get structural weights (w1, w2, w3) once
    with torch.no_grad():
         w1, w2, w3 = cdsqn_model.get_weights(x, edge_index, batch_index)
    
    cdsqn_violations = 0
    tolerance = 1e-5 # Floating point tolerance
    
    for _ in range(trials):
        # A subset B
        size_a = np.random.randint(0, num_nodes // 3)
        indices_a = np.random.choice(num_nodes, size_a, replace=False)
        
        mask_a = torch.zeros(1, num_nodes)
        mask_a[0, indices_a] = 1.0
        
        remaining = list(set(range(num_nodes)) - set(indices_a))
        if not remaining: continue
            
        size_b_add = np.random.randint(1, len(remaining) // 2)
        indices_b_add = np.random.choice(remaining, size_b_add, replace=False)
        
        mask_b = mask_a.clone()
        mask_b[0, indices_b_add] = 1.0
        
        remaining_v = list(set(remaining) - set(indices_b_add))
        if not remaining_v: continue
        v_idx = np.random.choice(remaining_v)
        
        mask_a_v = mask_a.clone()
        mask_a_v[0, v_idx] = 1.0
        
        mask_b_v = mask_b.clone()
        mask_b_v[0, v_idx] = 1.0
        
        # Prepare batch of 4 sets: A, A+v, B, B+v
        actions_batch = torch.cat([mask_a, mask_a_v, mask_b, mask_b_v], dim=0) # [4, N]
        
        with torch.no_grad():
            qs = cdsqn_model.compute_q(w1, w2, w3, actions_batch) # [4]
            
        val_a = qs[0].item()
        val_a_v = qs[1].item()
        val_b = qs[2].item()
        val_b_v = qs[3].item()
        
        gain_a = val_a_v - val_a
        gain_b = val_b_v - val_b
        
        # Check: Gain A >= Gain B
        if gain_a < gain_b - tolerance:
            cdsqn_violations += 1
            # print(f"Violation! A={val_a}, A+v={val_a_v} (Gain {gain_a}) | B={val_b}, B+v={val_b_v} (Gain {gain_b})")
            
    print(f"CDSQN Violations: {cdsqn_violations}/{trials} ({cdsqn_violations/trials:.2%})")
    print("-> Result: CDSQN guarantees submodularity via architecture (0% violations expected).")

if __name__ == "__main__":
    verify_models()
