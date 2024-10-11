# NetworkSimVisualization

Work in progress

Requirements in `requirement.txt`

SEE Changelog for week-to-week updates

`networkSim.py` contains network simulation class built on top of networkx

`simpleNode.py` contains node class used in the above

`tabularbellman.py` contains code for implementing a tabular version of Bellman's equation

`networkvis.py` bundles together a lot (but not all) of the code needed for visiualizing the simulation in terminal

To see visualization, see `qlearningCascadeNetwork.py` or `RmabCascadeNetwork.py`

The below is written by ChatGPT and should be mostly correct:

# Network Simulation with Reinforcement Learning

This project implements a network simulation using reinforcement learning techniques to determine optimal actions for node activation. The simulation models how nodes in a network can be activated or deactivated based on probabilities, and how actions taken at each timestep can affect the future state of the network. The goal is to determine an optimal strategy for activating nodes to maximize rewards in a dynamic network environment.

## Project Structure

The project is composed of several key components, each of which plays a role in simulating the network, managing the Q-table, or visualizing the process:

- **`networkSim.py`**: This module defines the `NetworkSim` class, which provides functionality to create a random network graph, perform state transitions, simulate cascades, and calculate rewards.
- **`simpleNode.py`**: Defines the `SimpleNode` class that represents individual nodes in the graph. Nodes have properties such as activation and deactivation chances, and can be activated or deactivated based on different conditions.
- **`tabularbellman.py`**: Implements the `TabularBellman` class, which uses a Q-table to learn the optimal actions for activating nodes. The class applies Bellman's equation using Monte Carlo sampling to estimate the future rewards of different actions.
- **`networkvis.py`**: Provides visualization tools using `matplotlib` and `networkx` to visually represent the network graph, its nodes, and edges at different timesteps of the simulation.

## Key Concepts

### Reinforcement Learning and Bellman's Equation

The Q-learning approach used in this project is based on Bellman's equation, which provides a way to iteratively update the Q-values for different states and actions. The Q-table is updated over multiple iterations using Monte Carlo sampling to approximate expected future values for each state-action pair. The Q-value update formula used is:

\[
Q(s,a) = (1 - \alpha) Q(s,a) + \alpha \Big( R(s,a) + \gamma \cdot \mathbb{E}_{s',u}[V(s')] \Big)
\]

Where:
- \( \alpha \) is the learning rate.
- \( \gamma \) is the discount factor.
- \( R(s, a) \) is the reward received for taking action \( a \) in state \( s \).
- The expectation \( \mathbb{E}_{s',u}[V(s')] \) is estimated using Monte Carlo sampling.

### Network Simulation

The network graph is represented using the `networkx` library, with nodes represented by instances of the `SimpleNode` class. Nodes can transition between active and inactive states based on passive activation/deactivation probabilities or as a result of explicit actions taken by the agent. The `NetworkSim` class handles the simulation logic, including cascading activations based on connected nodes.

### Visualization

The network is visualized using `matplotlib`, which helps in visualizing the activation and deactivation process of nodes over time. The visualization includes color coding for nodes and edges, indicating their current state (active/inactive).

## Usage

### Prerequisites

- Python 3.7+
- Required libraries: see requirements.txt

### Running the Simulation

1. **Initialize the Network**: Create a random graph using the `NetworkSim` class, specifying the number of nodes, edges, and activation/deactivation probabilities.

   ```python
   G = ns.init_random_graph(10, 30, PASSIVE_ACTIVATION_CHANCE, PASSIVE_DEACTIVATION_CHANCE, ACTIVE_ACTIVATION_CHANCE, ACTIVE_DEACTIVATION_CHANCE)
   ```

2. **Train the Q-table**: Use the `TabularBellman` class to train the Q-table for optimal actions over multiple iterations:

   ```python
   tab = tb(G, num_actions=2)
   tab.update_q_table(num_iterations=10)
   ```

3. **Visualize the Process**: Use `networkvis` to visualize the graph and show the state of the network over time:

   ```python
   pos = nx.spring_layout(G)  # Get positions for the nodes
   timestep = 0
   while True:
       plt.clf()  # Clear the previous plot
       seeded_nodes, utility = tab.get_best_action(G)
       nv.do_things(G, pos, seeded_nodes, node_obj_to_id, timestep)
       timestep += 1
   ```

### Parameters

- **Passive Activation/Deactivation Chance**: These parameters determine the likelihood that a node will change its state during each passive transition.
- **Active Activation/Deactivation Chance**: These parameters control the chances of nodes being activated or deactivated as a direct action taken by the agent.
- **`num_iterations`**: Number of iterations to update the Q-table during training.
- **`num_samples`**: Number of samples used to approximate the expected value in the Q-learning update.

## Contact

If you have questions, feel free to reach out via GitHub issues or email.
