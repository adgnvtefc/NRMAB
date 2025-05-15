# NetworkSimVisualization

Work in progress

Requirements in `requirement.txt`

SEE Changelog for week-to-week updates

# Network Simulation with Reinforcement Learning

This project implements a network simulation using reinforcement learning techniques to determine optimal actions for node activation. The simulation models how nodes in a network can be activated or deactivated based on probabilities, and how actions taken at each timestep can affect the future state of the network. The goal is to determine an optimal strategy for activating nodes to maximize rewards in a dynamic network environment.

## Project Structure

The project is composed of several key components. The key network simulations, built on top of the `networkx` library, functions as follows:
- `networkSim.py` defines crucial functions in creating, facilitating state changes, and collecting data from the network.
- `networkvis.py` contains helper functions in visualizing the network using the `matplotlib` library
- `simpleNode.py` defines the custome node data type that is used in our networks
- `comparisons.py` streamlines performance comparison of multiple algorithms
- `plotting.py` streamlines visualization of data collected from `comparisons.py`
- `real_data_trial.py` is an example comparing DQN and Hill Climbing algorithms using the aforementioned modules.

Algorithms can be found in the `algorithms` folder. Implemented algorithms include:
- DQN with Hill Climbing: uses a DQN to predict Q(s,a) for each individual action in a state, then uses greedy hill-climbing to select the top k actions with the highest predicted Q values. Found in `deepq.py` with `deep_q_env.py` supporting.
- Naive Hill Climbing (K step look ahead): uses Greedy Hill Climbing with either Bellman's equation (only works for small graphs) or probability-informed deterministic action value calculation to select top k actions. Found in `hillClimb.py`.
- Tabular Bellman: uses tabular Q learning to select top k actions. Only works for small graph sizes. Found in `tabularbellman.py`.
- Whittle Index: calculates the Whittle Index used in traditional network-blind RMABs. Often used as baseline comparison. Found in `whittle.py`. (NOTE: temporarily dysfunctional after refactoring due to probably multithreading shenanigans)
New algorithms may be added in time. `comparisons.py` should be updated concurrently as new algorithms are introduced. 

## Data

Real-world data used to create graph structures is in `graphs`. While networks are created from that data, individual node properties and cascade probabilities are independently defined.

## Examples

For a visual representation of a graph and algorithm, see `visualized_simulations`. Some modules prefaced with `OLD` may be outdated. Modules not prefaced as such may still be outdated. Hopefully they work as intended.

Example comparison can be found in `example_comparison_usage` (NOTE: outdated) and `real_data_trial.py`. Results to the real data trial can be found in `results` folders.

## Old

`comparisons_old` contains the old code for comparing algorithms. In case you ever wondered why `comparisons.py` is necessary, look into that folder. It is not pretty. You have been warned.
