# Networked Restless Multi-Arm Bandits with Reinforcement Learning

## Abstract
Restless Multi-Armed Bandits (RMABs) are a powerful framework for sequential decision-making, widely applied in resource allocation and intervention optimization challenges in public health. However, traditional RMABs assume independence among arms, limiting their ability to account for interactions between individuals that can be common and significant in a real-world environment. 

This repository implements **Networked RMAB**, a novel framework that integrates the RMAB model with the independent cascade model to capture interactions between arms in networked environments. We introduce **Contextual Deep Submodular Q-Networks (CDSQN)**, a specialized Deep Q-Learning architecture that guarantees submodularity of the learned Q-function, enabling efficient $1-1/e$ approximate action selection via a greedy hill-climbing strategy. We prove that the Bellman operator under this hill-climbing strategy remains a $\gamma$-contraction, ensuring convergence.

Experimental results on real-world graph data demonstrate that our Q-learning approach outperforms both $k$-step look-ahead and network-blind approaches, highlighting the importance of capturing and leveraging network effects where they exist.

---

## Usage Instructions

1. **Setup**: Run `reinstall.sh` to create the virtual environment (`venv`) and install all dependencies.
   ```bash
   bash reinstall.sh
   source venv/bin/activate
   ```

2. **Replicate Experiments**:
   - **Real-World Comparison (Figure 2)**: Run the India contact graph simulation.
     ```bash
     python india_real_data_trial.py
     ```
     *Compares CDSQN, DQN, Whittle Index, Hill-Climbing, and Random baselines.*

   - **Optimality Verification (Figure 3)**: Compare against optimal Tabular Q-Learning on small graphs.
     ```bash
     python dqn_v_tabular.py
     ```

   - **Computational Cost (Figure 4)**: Benchmark training time vs. graph size.
     ```bash
     python benchmark_costs.py
     python plot_ccost.py
     ```

3. **Results**: Output plots and data are saved to the `real_data_trials/results`, `real_data_trials/dvt`, and `real_data_trials/c_cost` directories.

---

## Source Code Description

### Experiments
| Script | Description |
|--------|-------------|
| **`india_real_data_trial.py`** | Runs the main comparison on the 202-node India contact network. Evaluates strategies (CDSQN, DQN, Whittle, HillClimb, Random) over multiple seeds. |
| **`dqn_v_tabular.py`** | Validates CDSQN and DQN performance against the optimal Tabular Bellman solution on a small (n=10) graph to prove near-optimality. |
| **`benchmark_costs.py`** | Collects runtime data for Tabular, DQN, and CDSQN across varying node counts. |
| **`plot_ccost.py`** | Visualizes the computational cost benchmark data. |

### Algorithms (`/algorithms`)
| Algorithm | File | Highlights |
|-----------|------|------------|
| **Contextual Deep Submodular Q-Net** | `cdsqn.py` | Hypernetwork-based architecture enforcing submodularity via concave activations and positive weights, and using greedy hill-climbing for action selection. |
| **Deep Q-Network** | `deepq.py` | Standard MLP-based DQN with greedy hill-climbing action selection. |
| **Whittle Index** | `whittle.py` | Classic RMAB index policy, unaware of network effects. |
| **1-Step Lookahead** | `hillClimb.py` | Greedy heuristic optimizing immediate gain. |
| **Tabular Bellman** | `tabularbellman.py` | Exact value iteration for small state spaces (ground truth). |

### Environments
- **`algorithms/cdsqn_env.py`**: Environment wrapper for CDSQN, handling graph state representations.
- **`algorithms/deepq_env.py`**: Standard binary multi-action space environment.

### Data
- **`graphs/India.txt`**: 202-node, 692-edge undirected contact network from an Indian village, used for the main experimental validation.

---

## Key Results
- **Performance**: CDSQN achieves the highest cumulative reward on the India contact network, outperforming both standard DQN and the Whittle Index.
- **Optimality**: On small graphs, CDSQN matches the performance of the optimal Tabular Q-Learning policy.
- **Scalability**: CDSQN's computational cost grows linearly with graph size, whereas the optimal Tabular solution scales exponentially.