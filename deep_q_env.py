import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from networkSim import NetworkSim as ns
from networkvis import NetworkVis as nv
import networkx as nx


class NetworkInfluenceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}


    def __init__(self, config, render_mode=None):
        super(NetworkInfluenceEnv, self).__init__()
        self.config = config
        # Define action and observation space
        # For example, action space could be a MultiBinary or MultiDiscrete space
        self.num_nodes = config['num_nodes']
        self.num_actions = config['num_nodes']
        self.cascade_prob = config['cascade_prob']

        self.stop_percent = config['stop_percent']

        self.observation_space = spaces.MultiBinary(self.num_nodes)
        self.action_space = spaces.MultiBinary(self.num_nodes)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize your network here
        self.original_graph = config['graph']
        self.pos = nx.spring_layout(self.original_graph)

        self.graph = config['graph']
        self.state = self.observation_space.sample(mask=np.zeros(self.num_nodes, dtype=np.int8))

    def _get_obs(self):
        # Return the state as a NumPy array
        return self.state.copy()
    def _get_info(self):
       return {"reward": ns.reward_function(graph=self.graph, seed=None, cascade_reward=self.cascade_prob)}
#### **2. Implement Required Methods**

#reset(): Resets the environment to an initial state and returns the initial observation.

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
        cur_reward = ns.reward_function(graph=self.graph, seed=None)

        action_indices = np.argwhere(action).flatten()

        ns.passive_state_transition_without_neighbors(self.graph, action_indices)
        ns.active_state_transition_graph_indices(self.graph, action_indices)
        ns.independent_cascade_allNodes(self.graph, self.cascade_prob)
        ns.rearm_nodes(self.graph)

        self.state = np.array([int(self.graph.nodes[i]['obj'].isActive()) for i in self.graph.nodes()])
        action_reward = ns.reward_function(graph=self.graph, seed=None, cascade_reward=self.cascade_prob)

        reward = action_reward - cur_reward
        terminated = (sum([self.graph.nodes[i]['obj'].isActive() for i in self.graph.nodes()]) >= self.stop_percent * self.num_nodes)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
          self._render_frame()
        
        return observation, reward, terminated, False, info

    def _render_frame(self):
       if self.render_mode == "human":
          nv.render(self.graph, self.pos)

from gymnasium.envs.registration import register

register(
    id='NetworkInfluence-v0',
    entry_point='deep_q_env:NetworkInfluenceEnv',
    max_episode_steps=300,
)
