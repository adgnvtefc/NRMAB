## Introduction
This is the source code used to run the experiments in *Networked Restless Multi-Arm Bandits with
Reinforcement Learning*. 

## Usage Instructions

1. Create and activate a virtual environment
2. Install core dependencies using pip install -r requirements.txt
3. To replicate experimental results, run `python india_real_data_trial.py`, `python dqn_v_tabular.py`, or `python plot_ccost.py` respectively.

**Note that due to stochasticity in graph generation, experimental results may not be able to be exactly replicated. However, approximate results should uphold experimental validity.**

## Experiments

Below are the three scripts used to produce the data for figures 2, 3, and 4 in that order:

| Script | What it shows | 
|--------|---------------|
| **`india_real_data_trial.py`** | **Real‑world India contact graph**<br>Compares 5 strategies (GNN, DQN, 1-step lookahead, Whittle index, and a “no‑action” baseline) on the same static network. Outputs PDF reward curves and CSV logs in `real_data_trials/`.| 
| **`dqn_v_tabular.py`** | **Synthetic toy graphs** (10‑node random) to contrast learning vs. exhaustive lookup.<br>Runs **DQN**, **Tabular Bellman**, and **GNN‑DQN** for 10 independent seeds, plots reward trajectories to `results/`.|
| **`plot_ccost.py`** | **Computational cost sweep**: renders the time‑per‑epoch of Tabular, DQN, and GNN agents as a function of graph size (2 → 11 nodes). Produces `*.pdf` files under `real_data_trials/c_cost/`. | 

## NRMAB Algorithms in `/algorithms`

| Algorithm | File | Highlights |
|-----------|------|------------|
| **Deep Q‑Network** | `algorithms/deepq.py` | vanilla feed‑forward Q‑net with ε‑greedy K‑step action construction, trained with [Tianshou] collectors and TensorBoard logging.|
| **Graph Neural Netowrk** | `algorithms/graphq.py` | deeper GCN backbone, Double DQN + Prioritised Replay; operates on PyTorch‑Geometric graphs. |
| **1-step lookahead** | `algorithms/hillClimb.py` | fast greedy scorer plus an exact Bellman variant for small horizons. |
| **Tabular Bellman** | `algorithms/tabularbellman.py` | exhaustive Q‑table with enumerated binary node states; useful for ground‑truth on toy graphs.|
| **Whittle Index policy** | `algorithms/whittle.py` | single‑threaded Whittle‑index calculator for restless‑bandit style node models. |

The environments live in:

* **`algorithms/deepq_env.py`** – binary multi‑action space (`MultiBinary`) where an agent seeds *k* nodes per step. :contentReference[oaicite:5]{index=5}  
* **`algorithms/graphq_env.py`** – single‑action space with rich 10‑dim node features exported as a PyG `Data` object. :contentReference[oaicite:6]{index=6}  

## Graphs

`graphs/India.txt` contains the raw graph edge‑list (whitespace‑separated `src dst` pairs) used in `india_real_data_trial.py`.

| File | Description |
|------|-------------|
| **India.txt** | 202‑node, 692‑edge undirected call‑graph of a contact network in an Indian village. Used by `india_real_data_trial.py`.|

## Helper Utilities

These modules sit behind the RL agents and experiments in this repository.  
They generate graphs, run large‑scale comparisons, and produce publication‑quality plots.

| Module | Purpose | 
|--------|---------|
| **`comparisons.py`** | Orchestrates **training and evaluation** across multiple algorithms (DQN, GNN, Whittle, 1-step lookahead, etc.). Handles model caching, per‑timestep simulation, and aggregates results for plotting.|
| **`networkSim.py`** | Low‑level **graph simulator**: creates random graphs, executes node‑state transitions, independent‑cascade spread, and Monte‑Carlo look‑ahead value functions.|
| **`plotting.py`** | **Plotting + history tracker** with an “academic” Matplotlib/Seaborn style. Saves per‑run CSVs, then plots mean ± STD and cumulative metrics across all historical runs.| `plot_trials`, `aggregate_history` |
| **`simpleNode.py`** | 

## Results
The full suite of results can be found in `\results`
