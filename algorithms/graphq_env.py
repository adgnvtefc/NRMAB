import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import copy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns

def convert_nx_to_pyg(G: nx.Graph, last_action_indices=None, current_step=None) -> Data:
    """
    Convert a NetworkX graph to a PyG Data object with enhanced node features.
    If last_action_indices is provided, mark an indicator feature for those nodes as 1.
    Each node's feature vector is extended from 7 to 10 dimensions by adding:
    - Active neighbor count
    - Global fraction of nodes currently active
    - Current timestep (normalized)
    """
    total_nodes = G.number_of_nodes()
    total_active = sum(1 for i in G.nodes() if G.nodes[i]['obj'].isActive())
    fraction_active = float(total_active) / total_nodes if total_nodes > 0 else 0.0
    for n in G.nodes():
        node_obj = G.nodes[n]['obj']
        # Mark if this node was acted on in the last action
        was_acted_on = 1.0 if (last_action_indices is not None and n in last_action_indices) else 0.0
        # Compute active neighbor count
        active_neighbor_count = sum(1 for nbr in G.neighbors(n) if G.nodes[nbr]['obj'].isActive())
        # Determine normalized timestep feature
        normalized_step = 0.0 if current_step is None else min(current_step / 50.0, 1.0)
        # Create each node's feature vector (10-dim: original 7 + 3 new features)
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

class GraphEnv(gym.Env):
    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config
        self.num_nodes = config['num_nodes']
        self.cascade_prob = config['cascade_prob']
        self.stop_percent = config['stop_percent']
        self.reward_function = config['reward_function']
        self.gamma = config.get('gamma', 0.8)

        self.original_graph = config['graph']
        self.latest_step_reward = 0

        self.last_action_indices = None

        self.action_space = spaces.Discrete(n=self.num_nodes, start=0)
        # Observation space: node features (10 per node) and edge indices
        self.observation_space = spaces.Dict({
            "x": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_nodes, 10),
                dtype=np.float32
            ),
            "edge_index": spaces.Box(
                low=0,
                high=self.num_nodes - 1,
                shape=(2, self.original_graph.number_of_edges() * 2),
                dtype=np.int64
            )
        })

        self.graph = None
        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # Reset the graph to its original state and clear any previous state
        self.graph = copy.deepcopy(self.original_graph)
        self.state = np.zeros(self.num_nodes, dtype=np.int8)
        self.last_action_indices = None
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        # Convert the current graph state to observation, including current timestep
        pyg_data = convert_nx_to_pyg(self.graph, self.last_action_indices, self.current_step)
        return {
            "x": pyg_data.x.numpy(),
            "edge_index": pyg_data.edge_index.numpy()
        }

    def _get_info(self):
        return {"reward": self.latest_step_reward}

    def step(self, action):
        # Increment timestep at the start of each step
        self.current_step += 1
        action_indices = [action]

        # 1. Measure total active value before actions
        prev_active_value = sum(
            self.graph.nodes[i]['obj'].getValue()
            for i in self.graph.nodes()
            if self.graph.nodes[i]['obj'].isActive()
        )

        # 2. Apply state transitions
        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        # 3. Measure total active value after transitions
        new_active_value = sum(
            self.graph.nodes[i]['obj'].getValue()
            for i in self.graph.nodes()
            if self.graph.nodes[i]['obj'].isActive()
        )

        # 4. Calculate reward contribution (delta in active value)
        raw_reward = new_active_value - prev_active_value

        # 5. Clip and normalize the reward
        reward_clip = 100.0  # clip threshold
        clipped_reward = float(np.clip(raw_reward, -reward_clip, reward_clip))
        normalized_reward = clipped_reward / reward_clip
        self.latest_step_reward = normalized_reward

        # 6. Prepare next state observation
        self.state = np.array([
            int(self.graph.nodes[i]['obj'].isActive())
            for i in self.graph.nodes()
        ], dtype=np.int8)
        self.last_action_indices = action_indices

        # 7. Check termination condition
        terminated = (
            sum(self.state) >= self.stop_percent * self.num_nodes
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, normalized_reward, terminated, False, info
