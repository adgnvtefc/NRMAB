#running example comparison between DQN and Tabular DQN
#EXAMPLES RESULTS IN RESULTS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networkSim import NetworkSim as ns
from comparisons import Comparisons 
from plotting import plot_trials

G = ns.init_random_graph(10, 20, 1, 2)
algorithms = ['dqn', 'tabular']

results = (Comparisons.run_many_comparisons(
    algorithms=algorithms, 
    initial_graph=G, 
    num_comparisons=10, 
    num_actions=2, 
    cascade_prob=0.05, 
    gamma=0.8, 
    timesteps=30, 
    timestep_interval=5))

plot_trials(
    results, 
    output_dir="results", 
    plot_cumulative_for=("reward",),  # tuple of metrics you want to also plot cumulatively
    file_prefix="comparison")         # prefix for output filed