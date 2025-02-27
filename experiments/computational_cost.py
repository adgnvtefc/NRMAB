#module for comparing the computational cost between using DQN and Tabular Q-Learning
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.deepq import train_dqn_agent, get_train_dqn_agent_time
from algorithms.tabularbellman import TabularBellman 
from networkSim import NetworkSim as ns
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

MIN_NODES = 2
MAX_NODES = 12
NUM_ACTIONS = 2
NUM_TRIALS = 1
CASCADE_PROB = 0.05
times = []
times_dqn = []

# for i in range(MIN_NODES, MAX_NODES):
#     total_time_for_epoch = 0
#     total_time_for_epoch_dqn = 0
#     for j in range(NUM_TRIALS):
#         graph = ns.init_random_graph(i, i, 1, 2)
#         tabular_bellman = TabularBellman(graph, num_actions=1, gamma=0.9, alpha=0.1)

#         start_time = time.time()

#         # Train the Q-table
#         #tabular_bellman.update_q_table(num_episodes=10, steps_per_episode=100, epsilon=0.1)
#         tabular_bellman.fill_q_table()

#         end_time = time.time()

#         total_time_for_epoch += (end_time - start_time)
#         #print(f"training time for graph with {i} nodes: {end_time - start_time}")

#         config_normal = {
#                 "graph": copy.deepcopy(graph),
#                 "num_nodes": len(graph.nodes),
#                 "cascade_prob": CASCADE_PROB,
#                 #arbitrary
#                 "stop_percent": 0.8,
#                 "reward_function": "normal"
#             }
#         model_normal, policy_normal = train_dqn_agent(config_normal, NUM_ACTIONS, num_epochs=3)
#         total_time_for_epoch_dqn += get_train_dqn_agent_time()
    
#     total_time_for_epoch /= NUM_TRIALS
#     total_time_for_epoch_dqn /= NUM_TRIALS
#     times.append(total_time_for_epoch)
#     times_dqn.append(total_time_for_epoch_dqn)
#     print(f"tabular average training time for graph with {i} nodes: {total_time_for_epoch}")    
#     print(f"dqn average training time for graph with {i} nodes: {total_time_for_epoch_dqn}")

# print(times)
# print(times_dqn)

# a_runtimes = np.array([0.201269793510437, 0.28529298305511475, 0.3605933666229248, 0.46743853092193605, 0.554813289642334, 0.6899858951568604, 0.8337456703186035, 0.9393076658248901, 1.1117680549621582, 1.3469007015228271, 1.4961987257003784, 1.7052064418792725, 1.9220740795135498, 2.21234986782074, 2.2144230127334597, 2.4953343152999876, 2.8278039932250976, 3.163846325874329])
# b_runtimes = np.array([4.933292880048976, 5.859606259944849, 6.464010970061645, 7.492647859989665, 7.842072710068896, 8.315672670048661, 9.57213875001762, 10.327240709983744, 11.230684849969112, 12.239974410040304, 12.938589599984699, 13.87210137997754, 14.49313642999623, 15.946637309971265, 16.25062983003445, 17.6153388700448, 18.556819780007935, 19.783559059980327])

a_runtimes = [0.015635251998901367, 0.006460905075073242, 0.0306243896484375, 0.06658530235290527, 0.18877530097961426, 0.5346224308013916, 1.4552600383758545, 4.5019142627716064, 11.475944519042969, 27.638933420181274]
b_runtimes = [9.014686099952087, 5.452420099871233, 6.1317813999485224, 7.145758500089869, 7.880650899838656, 8.134764599846676, 9.090941099915653, 10.239565199939534, 11.32968660001643, 11.974735700059682] 
nodes = np.arange(2, 12)  # Number of nodes from 2 to 19
plt.figure(figsize=(10, 6))
plt.plot(nodes, a_runtimes, marker='o', linestyle='-', label="Tabular Q Learning")
plt.plot(nodes, b_runtimes, marker='s', linestyle='-', label="Deep Q Learning")

plt.xlabel("Number of Nodes")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison of Tabular Q Learning and Deep Q Learning")
plt.legend()
plt.grid(True)
plt.show()

from scipy.optimize import curve_fit

# Define fitting functions
def poly_fit(n, a, b, c):
    return a * n**2 + b * n + c

def exp_fit(n, a, b):
    return a * np.exp(b * n)

# Fit polynomial and exponential models
poly_params_a, _ = curve_fit(poly_fit, nodes, a_runtimes)
exp_params_a, _ = curve_fit(exp_fit, nodes, a_runtimes, maxfev=5000)

poly_params_b, _ = curve_fit(poly_fit, nodes, b_runtimes)
exp_params_b, _ = curve_fit(exp_fit, nodes, b_runtimes, maxfev=5000)

# Generate fitted values
nodes_fine = np.linspace(2, 19, 100)
poly_fit_a_vals = poly_fit(nodes_fine, *poly_params_a)
exp_fit_a_vals = exp_fit(nodes_fine, *exp_params_a)

poly_fit_b_vals = poly_fit(nodes_fine, *poly_params_b)
exp_fit_b_vals = exp_fit(nodes_fine, *exp_params_b)

# Plot the results with equations as labels

# Plot the results with whole number x-axis labels

plt.figure(figsize=(10, 6))

# Plot actual data
plt.scatter(nodes, a_runtimes, label="Tabular Q Learning", color="blue", marker="o")
plt.scatter(nodes, b_runtimes, label="Deep Q Learning", color="red", marker="s")

# Plot fitted curves
plt.plot(nodes, poly_fit(nodes, *poly_params_a), label=r"Tabular - Polynomial Fit: $0.00597n^2 + 0.0451n + 0.0861$", linestyle="dashed", color="blue")
plt.plot(nodes, exp_fit(nodes, *exp_params_a), label=r"Tabular - Exponential Fit: $0.3133e^{0.1236n}$", linestyle="dotted", color="blue")

plt.plot(nodes, poly_fit(nodes, *poly_params_b), label=r"DQN - Polynomial Fit: $0.0104n^2 + 0.6386n + 3.7403$", linestyle="dashed", color="red")
plt.plot(nodes, exp_fit(nodes, *exp_params_b), label=r"DQN - Exponential Fit: $5.2515e^{0.0715n}$", linestyle="dotted", color="red")

plt.xlabel("Number of Nodes")
plt.ylabel("Runtime (seconds)")
plt.title("Fitting Algorithm Growth with Equations")

# Ensure whole numbers on x-axis
plt.xticks(nodes)  

plt.legend()
plt.grid(True)
plt.show()

# Print out fitted parameters for interpretation
poly_params_a, exp_params_a, poly_params_b, exp_params_b
