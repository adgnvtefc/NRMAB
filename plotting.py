#module used for plotting the results of comparisons
#TEMP, WILL FIX LATER

# plotting.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trials(
    trials, 
    output_dir="results", 
    plot_cumulative_for=("reward",),  # tuple of metrics you want to also plot cumulatively
    file_prefix="comparison"         # prefix for output files
):
    """
    Given a list of 'trials' (each an output of Comparisons.run_comparisons),
    this function computes the mean over trials for each algorithm and metric,
    saves it to CSV, and plots each metric (plus optional cumulative plots).

    :param trials: list of data_collection dicts, each for one simulation run.
    :param output_dir: directory to save CSV and plots.
    :param plot_cumulative_for: metrics for which to also produce cumulative plots.
    :param file_prefix: prefix string used in the output filenames.
    """
    
    if not trials:
        print("No trials provided. Exiting plot.")
        return

    # Make sure our output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Identify algorithms and metrics from the first trial ---
    first_trial = trials[0]
    algorithms = sorted(first_trial.keys())
    # We'll assume each algorithm has the same set of metrics:
    # e.g. ['timestep', 'cumulative_active_nodes', 'percent_activated', 'reward']
    # We'll exclude 'timestep' from the "metrics" we aggregate (since it's the x-axis).
    possible_keys = list(first_trial[algorithms[0]].keys())
    if 'timestep' in possible_keys:
        possible_keys.remove('timestep')
    metrics = sorted(possible_keys)

    # --- 2) Gather all timesteps (assume they are consistent) from the first trial for each algorithm ---
    # We'll assume all algorithms have the same length of timesteps in a single trial.
    # We'll also assume each trial has the same number of timesteps.
    T = len(first_trial[algorithms[0]]['timestep'])
    timesteps = np.array(first_trial[algorithms[0]]['timestep'])

    # --- 3) Create data structures to hold all values across trials ---
    # data_values[algo][metric] will be a np.array of shape (num_trials, T)
    data_values = {
        algo: {
            metric: np.zeros((len(trials), T), dtype=float)
            for metric in metrics
        }
        for algo in algorithms
    }

    # --- 4) Fill data_values from each trial ---
    for trial_idx, trial in enumerate(trials):
        for algo in algorithms:
            # For each metric, fill the row [trial_idx, :]
            for metric in metrics:
                data_values[algo][metric][trial_idx, :] = trial[algo][metric]

    # --- 5) Compute mean over trials (axis=0) for each metric ---
    # We'll store the results in a dictionary of shape [algo][metric][mean array].
    mean_data = {}
    for algo in algorithms:
        mean_data[algo] = {}
        for metric in metrics:
            mean_data[algo][metric] = data_values[algo][metric].mean(axis=0)  # shape (T,)

    # --- 6) Build a DataFrame to export to CSV ---
    # We'll have columns like: Timestep, <algo>_<metric>_mean, ...
    # The index from 1..T is the timestep for clarity.
    df_dict = {
        'timestep': timesteps
    }
    for algo in algorithms:
        for metric in metrics:
            col_name = f"{algo}_{metric}_mean"
            df_dict[col_name] = mean_data[algo][metric]
    df = pd.DataFrame(df_dict)

    # --- 7) Save the CSV ---
    csv_path = os.path.join(output_dir, f"{file_prefix}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved mean metrics to {csv_path}")

    # --- 8) Plot the mean metrics (one plot per metric) ---
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            y_vals = mean_data[algo][metric]
            plt.plot(timesteps, y_vals, label=f"{algo}", marker='o')
        plt.title(f"Mean {metric} vs. Timestep")
        plt.xlabel("Timestep")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        outpath = os.path.join(output_dir, f"{file_prefix}_{metric}_mean.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"Saved plot of mean {metric} to {outpath}")

        # --- 9) Plot cumulative if this metric is in the `plot_cumulative_for` ---
        if metric in plot_cumulative_for:
            plt.figure(figsize=(10, 6))
            for algo in algorithms:
                y_vals = mean_data[algo][metric]
                y_cum = np.cumsum(y_vals)
                plt.plot(timesteps, y_cum, label=f"{algo}", marker='s')
            plt.title(f"Cumulative {metric} vs. Timestep")
            plt.xlabel("Timestep")
            plt.ylabel(f"Cumulative {metric}")
            plt.grid(True)
            plt.legend()
            outpath = os.path.join(output_dir, f"{file_prefix}_{metric}_cumulative.png")
            plt.savefig(outpath, dpi=300)
            plt.close()
            print(f"Saved plot of cumulative {metric} to {outpath}")


def main():
    """
    Example usage. 
    Suppose you've already run something like:

        trials = Comparisons.run_many_comparisons(
            algorithms=["hillclimb", "dqn", "whittle", "tabular", "none", "random"],
            initial_graph=G,
            num_comparisons=50,
            num_actions=2,
            cascade_prob=0.2,
            gamma=0.9,
            timesteps=20,
            timestep_interval=1
        )

    Then you just call:

        plot_trials(trials, output_dir="results", plot_cumulative_for=("reward",))
    """
    pass

if __name__ == "__main__":
    main()
