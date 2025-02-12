import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
import networkx as nx
import random

class NetworkInfluenceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config, render_mode=None):
        super(NetworkInfluenceEnv, self).__init__()
        self.config = config
        # Define action and observation space
        self.num_nodes = config['num_nodes']
        self.num_actions = config['num_nodes']
        self.cascade_prob = config['cascade_prob']
        self.stop_percent = config['stop_percent']
        self.reward_function = config['reward_function']
        self.gamma = 0.8  # Discount factor for the enhanced reward function

        self.observation_space = spaces.MultiBinary(self.num_nodes)
        self.action_space = spaces.MultiBinary(self.num_nodes)

        self.latest_step_reward = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize your network here
        self.original_graph = config['graph']
        self.pos = nx.spring_layout(self.original_graph)

        self.graph = config['graph']
        self.state = np.zeros(self.num_nodes, dtype=np.int8)

    def _get_obs(self):
        # Return the state as a NumPy array
        return self.state.copy()

    def _get_info(self):
        return {"reward": self.latest_step_reward}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        # Reset the network to the initial state
        self.graph = copy.deepcopy(self.original_graph)
        self.state = np.zeros(self.num_nodes, dtype=np.int8)
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        action_indices = np.argwhere(action).flatten()

        #significantly slows down with any parameter other than num_samples=1 (abt 2 min per epoch at =1, 20 min at =3, for... reasons ig)
        reward = ns.action_value_function(self.graph, action, num_actions=1, cascade_prob=self.cascade_prob, gamma=self.gamma, horizon=1, num_samples=1)
        self.latest_step_reward = reward

        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        self.state = np.array([
            int(self.graph.nodes[i]['obj'].isActive()) for i in self.graph.nodes()
        ])

        terminated = (
            sum([self.graph.nodes[i]['obj'].isActive() for i in self.graph.nodes()]) >= self.stop_percent * self.num_nodes
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        if self.render_mode == "human":
            nv.render(self.graph, self.pos)

# Register the environment
from gymnasium.envs.registration import register

register(
    id='NetworkInfluence-v0',
    entry_point='network_env:NetworkInfluenceEnv',
    max_episode_steps=300,
)