#running example comparison between DQN and Tabular DQN
from networkSim import NetworkSim as ns
from comparisons import Comparisons 

G = ns.init_random_graph(10, 20, 1, 2)
algorithms = ['dqn', 'tabular']

print(Comparisons.run_many_comparisons(
    algorithms=algorithms, 
    initial_graph=G, 
    num_comparisons=10, 
    num_actions=2, 
    cascade_prob=0.05, 
    gamma=0.8, 
    timesteps=30, 
    timestep_interval=5))
