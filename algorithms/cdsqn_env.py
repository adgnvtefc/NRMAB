import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import copy
from networkSim import NetworkSim as ns

def convert_nx_to_pyg(G: nx.Graph, last_action_indices=None, current_step=None) -> Data:
    """
    Convert a NetworkX graph to a PyG Data object with enhanced node features.
    """
    total_nodes = G.number_of_nodes()
    total_active = sum(1 for i in G.nodes() if G.nodes[i]['obj'].isActive())
    fraction_active = float(total_active) / total_nodes if total_nodes > 0 else 0.0
    for n in G.nodes():
        node_obj = G.nodes[n]['obj']
        was_acted_on = 1.0 if (last_action_indices is not None and n in last_action_indices) else 0.0
        active_neighbor_count = sum(1 for nbr in G.neighbors(n) if G.nodes[nbr]['obj'].isActive())
        normalized_step = 0.0 if current_step is None else min(current_step / 50.0, 1.0)
        G.nodes[n]['x'] = torch.tensor([
            float(node_obj.isActive()),
            node_obj.getValue(),
            node_obj.active_activation_active,
            node_obj.active_activation_passive,
            node_obj.passive_activation_active,
            node_obj.passive_activation_passive,
            was_acted_on,
            float(active_neighbor_count),
            fraction_active,
            normalized_step,
        ], dtype=torch.float)
    data = from_networkx(G)
    return data

class CDSQNEnv(gym.Env):
    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config
        self.num_nodes = config['num_nodes']
        self.cascade_prob = config['cascade_prob']
        self.stop_percent = config['stop_percent']
        self.stop_percent = config['stop_percent']
        self.gamma = 0.8 # Hardcoded to match deepq_env.py for consistent reward definition

        self.original_graph = config['graph']
        self.latest_step_reward = 0
        self.last_action_indices = None

        # Change: MultiBinary action space for selecting sets of nodes
        self.action_space = spaces.MultiBinary(self.num_nodes)
        
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 10), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.num_nodes - 1, shape=(2, self.original_graph.number_of_edges() * 2), dtype=np.int64)
        })

        self.graph = None
        self.current_step = 0
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.graph = copy.deepcopy(self.original_graph)
        self.last_action_indices = None
        self.current_step = 0
        self.latest_step_reward = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pyg_data = convert_nx_to_pyg(self.graph, self.last_action_indices, self.current_step)
        return {
            "x": pyg_data.x.numpy(),
            "edge_index": pyg_data.edge_index.numpy()
        }

    def step(self, action):
        self.current_step += 1
        
        # Change: Handle MultiBinary action (numpy array of 0s and 1s)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        action_indices = np.argwhere(action).flatten()

        # Align reward calculation with standard RL (immediate reward only)
        # Using ns.reward_function calculates the value of the current state (sum of active node values)
        reward = ns.reward_function(self.graph, action_indices) / 100.0
        self.latest_step_reward = reward

        # State transitions
        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        self.last_action_indices = action_indices
        
        # Check termination
        active_count = sum(1 for i in self.graph.nodes() if self.graph.nodes[i]['obj'].isActive())
        terminated = (
            sum([self.graph.nodes[i]['obj'].isActive() for i in self.graph.nodes()]) >= self.stop_percent * self.num_nodes
        )
        
        # Only truncated if max steps reached (handled by TimeLimit wrapper usually, but adding explicit check for consistency)
        truncated = False 

        return self._get_obs(), reward, terminated, truncated, {"reward": reward}


# Register the environment
from gymnasium.envs.registration import register

register(
    id='NetworkInfluence-CDSQN-v0',
    entry_point='algorithms.cdsqn_env:CDSQNEnv',
    max_episode_steps=50,
)
