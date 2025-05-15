import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Map for formal display names (if needed)
formal_name = {
    'graph': 'GNN',
    'dqn': 'DQN',
    'hillclimb': 'k-step lookahead',
    'whittle': 'Whittle',
    'none': 'None'
}

def aggregate_history(
    output_dir="real_data_trials/results",
    plot_cumulative_for=None,
    file_prefix="comparison"
):
    """
    Load all historical CSVs, compute mean ± std of the difference against the 'none' baseline,
    and plot both instantaneous and cumulative difference metrics.

    Args:
        output_dir (str): Base output directory containing 'history' subfolder.
        plot_cumulative_for (list or tuple, optional): List of metric names to also plot cumulative difference.
            If None, cumulative difference is plotted for all parsed metrics.
        file_prefix (str): Prefix for metric CSV files (default 'comparison').
    """
    history_dir = os.path.join(output_dir, 'history')
    pattern = os.path.join(history_dir, f"{file_prefix}_metrics_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print("No historical CSVs found in", history_dir)
        return

    # Read all CSVs
    dfs = [pd.read_csv(f) for f in csv_files]

    # Extract timesteps and extend with zero
    timesteps = dfs[0]['timestep'].values
    timesteps_ext = np.concatenate(([0], timesteps))

    # Filter for mean columns and parse algos/metrics
    cols = [c for c in dfs[0].columns if c != 'timestep' and c.endswith('_mean')]
    algos = sorted({c.split('_')[0] for c in cols})
    metrics = sorted({'_'.join(c.split('_')[1:-1]) for c in cols})

    # Default plot_cumulative_for to all metrics
    if plot_cumulative_for is None:
        plot_cumulative_for = metrics

    # Build data stack for each column
    data_stack = {
        col: np.stack([df[col].values for df in dfs], axis=0)
        for col in cols
    }

    # Plot instantaneous difference mean ± std for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        baseline_key = f"none_{metric}_mean"
        if baseline_key not in data_stack:
            continue
        baseline = data_stack[baseline_key]

        for algo in algos:
            if algo == 'none':
                continue
            key = f"{algo}_{metric}_mean"
            if key not in data_stack:
                continue

            arr = data_stack[key] - baseline
            mu = arr.mean(axis=0)
            sigma = arr.std(axis=0)

            mu_ext = np.concatenate(([0.], mu))
            sigma_ext = np.concatenate(([0.], sigma))

            label = formal_name.get(algo, algo)
            plt.plot(timesteps_ext, mu_ext, marker='o', label=label)
            plt.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.2
            )

        plt.title(f"Difference (±STD) of {metric.replace('_', ' ')} vs. Timestep")
        plt.xlabel("Timestep")
        plt.ylabel(f"Δ {metric.replace('_', ' ')}")
        plt.grid(True)
        plt.legend(title="Algorithm")
        outpath = os.path.join(output_dir, f"{file_prefix}_{metric}_history_diff.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"Saved difference mean±std for {metric} to {outpath}")

    # Plot cumulative difference for specified metrics
    for metric in plot_cumulative_for:
        if metric not in metrics:
            continue

        plt.figure(figsize=(10, 6))
        baseline_key = f"none_{metric}_mean"
        if baseline_key not in data_stack:
            continue
        baseline = data_stack[baseline_key]

        for algo in algos:
            if algo == 'none':
                continue
            key = f"{algo}_{metric}_mean"
            if key not in data_stack:
                continue

            diffs = data_stack[key] - baseline
            cum = np.cumsum(diffs, axis=1)
            mu_cum = cum.mean(axis=0)
            sigma_cum = cum.std(axis=0)

            mu_ext = np.concatenate(([0.], mu_cum))
            sigma_ext = np.concatenate(([0.], sigma_cum))

            label = formal_name.get(algo, algo)
            plt.plot(timesteps_ext, mu_ext, marker='s', label=label)
            plt.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.2
            )

        plt.title(f"Cumulative Difference (±STD) of {metric.replace('_', ' ')} vs. Timestep")
        plt.xlabel("Timestep")
        plt.ylabel(f"Cumulative Δ {metric.replace('_', ' ')}")
        plt.grid(True)
        plt.legend(title="Algorithm")
        outpath = os.path.join(output_dir, f"{file_prefix}_{metric}_history_cumulative_diff.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"Saved cumulative difference for {metric} to {outpath}")


if __name__ == '__main__':
    aggregate_history()