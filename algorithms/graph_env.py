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

def convert_nx_to_pyg(G: nx.Graph, last_action_indices=None) -> Data:
    """
    Convert a NetworkX graph to a PyG Data object.
    If last_action_indices is provided, we set the 7th feature=1 for those nodes.
    """
    for n in G.nodes():
        node_obj = G.nodes[n]['obj']

        # Mark if acted on
        was_acted_on = 1.0 if (last_action_indices is not None and n in last_action_indices) else 0.0

        # Create each node's feature vector (7-dim)
        G.nodes[n]['x'] = torch.tensor([
            float(node_obj.isActive()),
            node_obj.getValue(),
            node_obj.active_activation_active,
            node_obj.active_activation_passive,
            node_obj.passive_activation_active,
            node_obj.passive_activation_passive,
            was_acted_on,
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
        self.gamma = config.get('gamma', 0.99)

        self.original_graph = config['graph']
        self.latest_step_reward = 0

        self.last_action_indices = None

        self.action_space = spaces.Discrete(n=self.num_nodes, start=0)

        self.observation_space = spaces.Dict({
            "x": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_nodes, 7),
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

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.graph = copy.deepcopy(self.original_graph)
        self.state = np.zeros(self.num_nodes, dtype=np.int8)

        self.last_action_indices = None

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        pyg_data = convert_nx_to_pyg(self.graph, self.last_action_indices)
        return {
            "x": pyg_data.x.numpy(),
            "edge_index": pyg_data.edge_index.numpy()
        }

    def _get_info(self):
        return {"reward": self.latest_step_reward}

    def step(self, action):
        action_indices = [action]

        reward = ns.action_value_function(
            self.graph,
            action,
            num_actions=1,
            cascade_prob=self.cascade_prob,
            gamma=self.gamma,
            horizon=1,
            num_samples=1
        )
        # Removed the line that decayed rewards over time
        self.latest_step_reward = reward

        # Update graph state
        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        self.state = np.array([
            int(self.graph.nodes[i]['obj'].isActive())
            for i in self.graph.nodes()
        ])

        self.last_action_indices = action_indices

        terminated = (
            sum(self.state) >= self.stop_percent * self.num_nodes
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info
