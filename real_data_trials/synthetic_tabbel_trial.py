import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim  as ns
from networkvis import NetworkVis as nv
from comparisons import Comparisons 
from plotting import plot_trials



algorithms = ['dqn', 'tabular', 'whittle', 'hillclimb', 'none']
#ALL PREVIOUS EXPERIMENTS RAN WITH 30 ACTIONS
NUM_ACTIONS = 2
NUM_COMPARISONS = 50
CASCADE_PROB = 0.05
GAMMA = 0.8
TIMESTEPS = 30
TIMESTEP_INTERVAL=5
graph = ns.init_random_graph(10,30,1,2)
pos = nx.spring_layout(graph)  # Positioning of nodes

comp = Comparisons()
comp.train_tabular(graph, NUM_ACTIONS, GAMMA)
print("whit")
comp.train_whittle(graph, GAMMA)
print("tle")
comp.train_dqn(graph, NUM_ACTIONS, CASCADE_PROB)


metadata = {"algorithms": algorithms,
            "initial_graph": graph,
            "num_comparisons": NUM_COMPARISONS,
            "num_actions": NUM_ACTIONS,
            "cascade_prob": CASCADE_PROB,
            "gamma": GAMMA,
            "timesteps": TIMESTEPS}

results = (comp.run_many_comparisons(
    algorithms=algorithms, 
    initial_graph=graph, 
    num_comparisons=NUM_COMPARISONS, 
    num_actions=NUM_ACTIONS,
    cascade_prob=CASCADE_PROB, 
    gamma=GAMMA, 
    timesteps=TIMESTEPS, 
    timestep_interval=TIMESTEP_INTERVAL))

plot_trials(
    results, 
    output_dir="results", 
    plot_cumulative_for=("reward",),  # tuple of metrics you want to also plot cumulatively
    file_prefix="comparison",
    metadata=metadata)         # prefix for output filed

print(algorithms)
print(f"num comparisons: {NUM_COMPARISONS}")
print(f"num actions: {NUM_ACTIONS}")
print(f"cascade prob: {CASCADE_PROB}")
print(len(graph.nodes))
print(len(graph.edges))


# Training Q-table: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [02:06<00:00,  2.38iteration/s]
# whit
# tle
# Training DQN agent with normal reward function...
# Epoch #1:   5%|####                                                                            | 50/1000 [00:01<00:21, 44.76it/s, env_step=50, gradient_step=5, len=3, n/ep=9, n/st=50, rew=65.31]C:\Users\zhang\Documents\Academics\Reseach\.venv\Lib\site-packages\tianshou\data\buffer\base.py:237: RuntimeWarning: invalid value encountered in scalar multiply
#   return ptr, self._ep_rew * 0.0, 0, self._ep_idx
# C:\Users\zhang\Documents\Academics\Reseach\.venv\Lib\site-packages\numpy\core\_methods.py:173: RuntimeWarning: invalid value encountered in subtract
#   x = asanyarray(arr - arrmean)
# Epoch #1: 1001it [00:18, 53.10it/s, env_step=1000, gradient_step=100, len=4, n/ep=14, n/st=50, rew=92.47]
# Epoch #1: test_reward: 111.089487 ± 73.563359, best_reward: 120.024282 ± 35.380152 in #0
# Epoch #2: 1001it [00:17, 55.67it/s, env_step=2000, gradient_step=200, len=4, n/ep=12, n/st=50, rew=94.81]                                                                                          
# Epoch #2: test_reward: 82.926893 ± 47.150978, best_reward: 120.024282 ± 35.380152 in #0
# Epoch #3: 1001it [00:17, 56.77it/s, env_step=3000, gradient_step=300, len=4, n/ep=13, n/st=50, rew=102.71]
# Epoch #3: test_reward: -inf ± nan, best_reward: 120.024282 ± 35.380152 in #0
# dqn
# 1.6231760988011956
# 1500