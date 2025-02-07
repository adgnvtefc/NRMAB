import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from networkSim import NetworkSim as ns

class NetworkEnv(gym.Env):
    def __init__(self, graph, k=2):
        super(NetworkEnv, self).__init__()
        self.original_graph = graph  # Store the original graph
        self.graph = copy.deepcopy(graph)
        self.k = k  # Number of nodes to activate at each timestep
        self.num_nodes = len(self.graph.nodes())
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.reset()
    
    def reset(self):
        # Reset the graph to its initial state with all nodes inactive
        self.graph = copy.deepcopy(self.original_graph)
        for node in self.graph.nodes():
            self.graph.nodes[node]['obj'].deactivate()
        return self._get_observation()
    
    def step(self, action_logits):
        # Ensure that exactly k nodes are selected
        action_logits = action_logits.flatten()
        if len(action_logits) != self.num_nodes:
            raise ValueError("Action length must be equal to number of nodes")
        
        # Select k nodes with the highest logits
        node_indices = np.argpartition(-action_logits, self.k)[:self.k]
        exempt_nodes = [self.graph.nodes[i]['obj'] for i in node_indices]
        
        # Apply active state transition probabilistically
        changed_nodes_active = ns.active_state_transition(exempt_nodes)
        
        # Apply passive state transition to remaining nodes
        remaining_nodes = set(self.graph.nodes()) - set(node_indices)
        remaining_node_objs = [self.graph.nodes[i]['obj'] for i in remaining_nodes]
        ns.passive_state_transition_without_neighbors(
            self.graph, exempt_nodes=exempt_nodes)
        
        # Cascades
        ns.independent_cascade_allNodes(self.graph, edge_weight=0.1)
        ns.rearm_nodes(self.graph)
        
        # Compute reward using the function from tabularbellman
        reward = self._compute_reward(set(node_indices))
        
        # Check if 50% or more nodes are active
        active_nodes_count = sum([self.graph.nodes[i]['obj'].isActive() for i in self.graph.nodes()])
        if active_nodes_count >= 0.5 * self.num_nodes:
            done = True
        else:
            done = False
        
        # Optional: Include diagnostic information
        info = {
            'active_nodes': active_nodes_count,
            'total_nodes': self.num_nodes,
            'activated_nodes_indices': list(node_indices)
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _get_observation(self):
        state = np.array([int(self.graph.nodes[i]['obj'].isActive()) for i in self.graph.nodes()], dtype=np.float32)
        return state
    
    def _compute_reward(self, seed):
        # Use the reward function from tabularbellman
        reward = ns.reward_function(self.graph, seed)
        return reward
    
    def render(self):
        pass
    
    def close(self):
        pass
