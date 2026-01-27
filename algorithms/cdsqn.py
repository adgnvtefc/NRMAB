import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch as PyGBatch
import numpy as np
import time
from typing import Dict, Iterator, Tuple, List, Optional, Any
from dataclasses import dataclass
import os

from tianshou.policy import BasePolicy
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from algorithms.cdsqn_env import CDSQNEnv

# Model Classes
class SafeSqrt(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return torch.sqrt(torch.clamp(x, min=0) + self.epsilon)

class SafeLog(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        return torch.log(1 + torch.clamp(x, min=0))

class HyperGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
    def forward(self, x, edge_index, batch_index=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        h_g = global_mean_pool(x, batch_index)
        return h_g

class CDSQN(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, hidden_dim, dsf_hidden_dim, num_heads=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.dsf_hidden_dim = dsf_hidden_dim
        self.num_heads = num_heads
        
        self.context_net = HyperGNN(node_feat_dim, hidden_dim, hidden_dim)
        
        self.w1_gen = nn.Linear(hidden_dim, num_heads * num_nodes * dsf_hidden_dim)
        self.w2_gen = nn.Linear(hidden_dim, num_heads * dsf_hidden_dim * dsf_hidden_dim)
        self.w3_gen = nn.Linear(hidden_dim, num_heads * dsf_hidden_dim * 1)
        
        self.act1 = SafeSqrt()
        self.act2 = SafeLog()

        self._initialize_generators()

    def _initialize_generators(self):
        nn.init.constant_(self.w1_gen.bias, 0.1)
        nn.init.constant_(self.w2_gen.bias, 0.1)
        nn.init.constant_(self.w3_gen.bias, 0.1)

    def get_weights(self, x, edge_index, batch_index):
        h_g = self.context_net(x, edge_index, batch_index)
        batch_size = h_g.size(0)
        
        w1_raw = self.w1_gen(h_g).view(batch_size, self.num_heads, self.num_nodes, self.dsf_hidden_dim)
        w2_raw = self.w2_gen(h_g).view(batch_size, self.num_heads, self.dsf_hidden_dim, self.dsf_hidden_dim)
        w3_raw = self.w3_gen(h_g).view(batch_size, self.num_heads, self.dsf_hidden_dim, 1)
        
        w1 = F.softplus(w1_raw)
        w2 = F.softplus(w2_raw)
        w3 = F.softplus(w3_raw)
        
        return w1, w2, w3
        
    def compute_q(self, w1, w2, w3, actions):
        actions = actions.float()
        
        if actions.dim() == 2:
            # [Batch, Num_Nodes]
            actions = actions.view(actions.size(0), 1, 1, actions.size(1))
            h1 = torch.matmul(actions, w1)
            h1 = self.act1(h1)
            h2 = torch.matmul(h1, w2)
            h2 = self.act2(h2)
            out_heads = torch.matmul(h2, w3)
            out_heads = out_heads.view(out_heads.size(0), -1)
            
        elif actions.dim() == 3:
            # [Batch, Candidates, Num_Nodes]
            B, C, N = actions.shape
            actions = actions.view(B, C, 1, 1, N)
            w1_exp = w1.unsqueeze(1)
            w2_exp = w2.unsqueeze(1)
            w3_exp = w3.unsqueeze(1)
            
            h1 = torch.matmul(actions, w1_exp)
            h1 = self.act1(h1)
            h2 = torch.matmul(h1, w2_exp)
            h2 = self.act2(h2)
            out_heads = torch.matmul(h2, w3_exp)
            out_heads = out_heads.view(B, C, -1)
            
        else:
            raise ValueError(f"Unsupported action shape: {actions.shape}")

        q_values, _ = torch.min(out_heads, dim=-1)
        return q_values

    def forward(self, x, edge_index, batch_index, actions):
        w1, w2, w3 = self.get_weights(x, edge_index, batch_index)
        return self.compute_q(w1, w2, w3, actions)

@dataclass
class TrainStepResult:
    loss: float

    def get_loss_stats_dict(self) -> Dict[str, float]:
        return {'loss': self.loss}

    def keys(self) -> Tuple[str, ...]:
        return ("loss",)

    def __getitem__(self, key: str) -> float:
        if key == "loss":
            return self.loss
        raise KeyError(f"TrainStepResult does not contain key: {key}")

    def __setitem__(self, key: str, value: float) -> None:
        if key == "loss":
            self.loss = value
        else:
            raise KeyError(f"TrainStepResult does not contain key: {key}")
    
    def items(self) -> Iterator[Tuple[str, float]]:
        for k in self.keys():
            yield k, self[k]

# Policy
class CDSQNPolicy(BasePolicy):
    def __init__(self, model, optim, action_dim, num_nodes, k=5, gamma=0.99, epsilon=1.0):
        super().__init__(action_space=gym.spaces.MultiBinary(action_dim))
        self.model = model
        self.optim = optim
        self.k = k
        self.num_nodes = num_nodes
        self._gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def _prepare_batch_data(self, batch_obs):
        """
        Convert Tianshou Batch obs (dict with stacked arrays) to PyG Batch data.
        batch_obs.x: [B, N, F]
        batch_obs.edge_index: [B, 2, E]
        """
        device = next(self.model.parameters()).device
        
        # If batch_obs is a numpy array (sometimes happens with single envs?), check first
        # But CDSQNEnv returns dict, so it should be Batch or dict
        
        x = batch_obs.x
        edge_index = batch_obs.edge_index
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            
        B, N, F = x.shape
        # Flatten x: [B*N, F]
        x_flat = x.view(-1, F)
        
        # Prepare edge_index and batch_index
        # edge_index is [B, 2, E]
        # We need to offset nodes for each graph i by i*N
        batch_assignment = []
        edge_indices_list = []
        
        for i in range(B):
            batch_assignment.append(torch.full((N,), i, dtype=torch.long, device=device))
            # Current edges: [2, E]
            edges = edge_index[i]
            offset = i * N
            edge_indices_list.append(edges + offset)
            
        batch_idx = torch.cat(batch_assignment)
        if len(edge_indices_list) > 0:
            full_edge_index = torch.cat(edge_indices_list, dim=1)
        else:
            full_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
        return x_flat, full_edge_index, batch_idx, B

    def forward(self, batch, state=None):
        # Action Selection (Greedy Hill Climbing)
        # obs is a Batch
        x_flat, edge_index, batch_idx, B = self._prepare_batch_data(batch.obs)
        
        # Get Weights (Context)
        # We compute weights once per state
        w1, w2, w3 = self.model.get_weights(x_flat, edge_index, batch_idx)
        
        device = x_flat.device
        final_actions = torch.zeros(B, self.num_nodes, device=device)
        
        # Current selected set for each batch item (initially all 0)
        current_actions = torch.zeros(B, self.num_nodes, device=device)
        
        # State Availability mask?
        # The node features contain "isActive" at index 0 (based on cdsqn_env.py)
        # x feature 0 is isActive.
        # batch.obs.x: [B, N, F]
        # available: x[:, :, 0] == 0
        node_active = batch.obs.x[:, :, 0] # numpy?
        if isinstance(node_active, np.ndarray):
            node_active = torch.tensor(node_active, device=device)
        
        # Hill Climbing Loop
        for step in range(self.k):
            # Identify candidates: Not active AND Not already selected
            # candidate_mask: [B, N] -> 1 if candidate, 0 otherwise
            candidate_mask = (node_active == 0) & (current_actions == 0)
            
            # If no candidates for a batch item, we can't add more.
            # We process all B in parallel.
            
            # Construct candidate actions
            # For each B, we want to evaluate Q(current + {v}) for all v where mask is 1
            # To vectorize: Create candidates [B, N, N]
            # Base action repeated: [B, N, N]
            base_actions = current_actions.unsqueeze(1).repeat(1, self.num_nodes, 1) # [B, N, N]
            
            # Add one-hot for each node
            eye = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(B, 1, 1) # [B, N, N]
            
            candidate_actions = base_actions + eye
            # Clamp to 1 (although shouldn't be >1 if logic is right)
            candidate_actions = torch.clamp(candidate_actions, 0, 1)
            
            # Compute Q
            # w1: [B, Heads, N, H]
            # actions: [B, N, N] (Candidates=N)
            q_values = self.model.compute_q(w1, w2, w3, candidate_actions) # [B, N]
            
            # Mask unavailable actions
            # Set Q=-inf where mask is 0
            q_values = q_values.masked_fill(~candidate_mask.bool(), -float('inf'))
            
            # Select Best
            best_q, best_idx = torch.max(q_values, dim=1) # [B]
            
            # Exploration
            if np.random.random() < self.epsilon:
                # Random selection among valid candidates
                # Naive: pick random index where mask is 1
                for i in range(B):
                    indices = candidate_mask[i].nonzero().squeeze()
                    if indices.numel() > 0:
                        if indices.numel() == 1:
                            choice = indices.item()
                        else:
                            choice = indices[torch.randint(0, len(indices), (1,))].item()
                        best_idx[i] = choice
                    else:
                         # No operation if no candidates
                        pass
            
            # Update current_actions
            # check if candidates existed
            has_candidates = candidate_mask.sum(dim=1) > 0
            
            rows = torch.arange(B, device=device)
            # Only update for those with candidates
            valid_rows = rows[has_candidates]
            valid_cols = best_idx[valid_rows]
            
            current_actions[valid_rows, valid_cols] = 1.0
            
        final_actions = current_actions
        return Batch(act=final_actions)

    def learn(self, batch, **kwargs):
        # batch has obs, act, rew, obs_next, done
        device = next(self.model.parameters()).device
        
        rewards = torch.tensor(batch.rew, dtype=torch.float32, device=device).view(-1)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device).view(-1)
        actions = torch.tensor(batch.act, dtype=torch.float32, device=device)
        
        # Current Q
        x_flat, edge_index, batch_idx, B = self._prepare_batch_data(batch.obs)
        w1, w2, w3 = self.model.get_weights(x_flat, edge_index, batch_idx)
        current_q = self.model.compute_q(w1, w2, w3, actions) # [B]
        
        # Target Q
        with torch.no_grad():
            x_next_flat, edge_index_next, batch_idx_next, B_next = self._prepare_batch_data(batch.obs_next)
            w1_next, w2_next, w3_next = self.model.get_weights(x_next_flat, edge_index_next, batch_idx_next)
            
            # Greedy maximization on Next State
            next_actions = torch.zeros(B_next, self.num_nodes, device=device)
            node_active_next = batch.obs_next.x[:, :, 0]
            if isinstance(node_active_next, np.ndarray):
                node_active_next = torch.tensor(node_active_next, device=device)
                
            for step in range(self.k):
                candidate_mask = (node_active_next == 0) & (next_actions == 0)
                
                base_actions = next_actions.unsqueeze(1).repeat(1, self.num_nodes, 1)
                eye = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(B_next, 1, 1)
                candidate_actions = torch.clamp(base_actions + eye, 0, 1)
                
                q_vals = self.model.compute_q(w1_next, w2_next, w3_next, candidate_actions)
                q_vals = q_vals.masked_fill(~candidate_mask.bool(), -float('inf'))
                
                best_q, best_idx = torch.max(q_vals, dim=1)
                
                has_candidates = candidate_mask.sum(dim=1) > 0
                rows = torch.arange(B_next, device=device)
                valid_rows = rows[has_candidates]
                valid_cols = best_idx[valid_rows]
                next_actions[valid_rows, valid_cols] = 1.0
            
            # Evaluate Max Q
            target_max_q = self.model.compute_q(w1_next, w2_next, w3_next, next_actions)
            target_q = rewards + self._gamma * (1 - dones) * target_max_q

        loss = F.mse_loss(current_q, target_q)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return TrainStepResult(loss=loss.item())

log_path = os.path.join('logs', 'cdsqn')
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

def train_cdsqn_agent(config, num_actions, num_epochs=3, step_per_epoch=1000):
    start_time = time.perf_counter()
    
    def get_env():
        return CDSQNEnv(config)
        
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(1)])
    
    def stop_fn(mean_rewards):
        return False
        
    def train_fn(epoch, env_step):
        epsilon = max(0.1, 1 - env_step / 50000)
        policy.epsilon = epsilon
        
    def test_fn(epoch, env_step):
        pass
        
    state_dim = 10 # 10 features per node
    hidden_dim = 64
    dsf_hidden_dim = 64
    
    model = CDSQN(config['num_nodes'], state_dim, hidden_dim, dsf_hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    policy = CDSQNPolicy(model, optimizer, action_dim=config['num_nodes'], num_nodes=config['num_nodes'], k=num_actions, gamma=0.99)
    
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=20000, buffer_num=10))
    
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=50,
        episode_per_test=0,
        batch_size=16,
        update_per_step=0.1,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        logger=logger
    ).run()
    
    end_time = time.perf_counter()
    train_cdsqn_agent.time = end_time - start_time
    return model, policy

train_cdsqn_agent.time = 0
def get_train_cdsqn_agent_time():
    return train_cdsqn_agent.time

select_action_cdsqn_stats = {'time': 0.0, 'calls': 0}

def select_action_cdsqn(graph, model, num_actions):
    """
    Inference interface for comparison scripts.
    """
    start_time = time.perf_counter()
    select_action_cdsqn_stats['calls'] += 1
    
    # We need to construct a "Batch" for the Policy
    # Create temp env or manually construct PyG data
    # Let's manually construct to avoid env overhead logic if possible, 
    # but CDSQNEnv logic (convert_nx_to_pyg) is needed.
    from algorithms.cdsqn_env import convert_nx_to_pyg
    
    # Convert graph to PyG
    # No last_action or step info easily available here? 
    # Usually in comparisons, we act step-by-step.
    # If the comparison script passes just the graph, we assume it's the current state.
    pyg_data = convert_nx_to_pyg(graph)
    
    # Make a Tianshou Batch
    # obs must be a Batch with x and edge_index
    obs = Batch(x=pyg_data.x.numpy()[None, ...], edge_index=pyg_data.edge_index.numpy()[None, ...])
    batch = Batch(obs=obs, info={})
    
    # Policy forward needs a dummy dummy model wrapper?
    # No, we can just instantiate a dummy policy or use the model directly if we replicate the logic.
    # But better to use the Policy class to ensure consistency.
    
    # We don't have the Policy object passed in, only Model.
    # We can reconstruct a temporary Policy wrapper or just replicate the greedy loop.
    # Replicating greedy loop is safer here to avoid Optimizer requirements.
    
    # Prepare Inputs
    device = next(model.parameters()).device
    x = torch.tensor(obs.x, dtype=torch.float32, device=device) # [1, N, F]
    edge_index = torch.tensor(obs.edge_index, dtype=torch.long, device=device) # [1, 2, E]
    
    # Prepare batch data (simplify since B=1)
    B, N, F = x.shape
    x_flat = x.view(-1, F)
    batched_edge_index = edge_index[0] # No offset needed for 1 graph
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)
    
    w1, w2, w3 = model.get_weights(x_flat, batched_edge_index, batch_idx)
    
    current_actions = torch.zeros(B, N, device=device)
    node_active = x[:, :, 0] # [1, N]
    
    for step in range(num_actions):
        candidate_mask = (node_active == 0) & (current_actions == 0)
        
        base_actions = current_actions.unsqueeze(1).repeat(1, N, 1) # [1, N, N]
        eye = torch.eye(N, device=device).unsqueeze(0) # [1, N, N]
        candidate_actions = torch.clamp(base_actions + eye, 0, 1)
        
        q_vals = model.compute_q(w1, w2, w3, candidate_actions)
        q_vals = q_vals.masked_fill(~candidate_mask.bool(), -float('inf'))
        
        best_q, best_idx = torch.max(q_vals, dim=1)
        
        if candidate_mask.sum() > 0:
            current_actions[0, best_idx[0]] = 1.0
            
    # Convert selected action to nodes
    selected_indices = current_actions[0].nonzero().squeeze().tolist()
    if isinstance(selected_indices, int):
        selected_indices = [selected_indices]
    elif len(selected_indices) == 0:  # Handle case where no indices are selected
        selected_indices = []

    selected_nodes = [graph.nodes[i]['obj'] for i in selected_indices]
    
    end_time = time.perf_counter()
    select_action_cdsqn_stats['time'] += (end_time - start_time)
    
    return selected_nodes

def get_cdsqn_total_time():
    return select_action_cdsqn_stats['time']

def get_cdsqn_times_called():
    return select_action_cdsqn_stats['calls']
