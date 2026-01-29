import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Seaborn base style without grid
sns.set_style('white')
sns.set_context('paper', font_scale=1.0)
# Matplotlib rc settings for formal academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'axes.titlesize': 16,
    'axes.titlepad': 10,
    'axes.labelsize': 14,
    'axes.labelpad': 8,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.pad': 6,
    'ytick.major.pad': 6,
    'lines.linewidth': 1.0,
    'axes.grid': False,
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.88,
    'text.usetex': False
})

# Formal names for legend
formal_name = {
    'dqn': 'DQN',
    'cdsqn': 'CDSQN',
    'hillclimb': '1-step lookahead',
    'whittle': 'Whittle',
    'none': 'None',
    'tabular': 'Tabular'
}
# Fixed, distinctive colors for each algorithm
colors = {
    'dqn':       "#bbea37",
    'hillclimb': '#e377c2',
    'whittle':   '#2ca02c',
    'none':      '#9467bd',
    'tabular':   '#17becf',
    'cdsqn':     '#ff7f0e'
}
# Distinctive markers
markers = {
    'dqn':       '.',
    'hillclimb': 'd',
    'whittle':   '^',
    'none':      'v',
    'tabular':   '*',
    'cdsqn':     's'
}
# Legend order
legend_order = ['cdsqn', 'dqn', 'whittle', 'hillclimb', 'none', 'tabular', 'random']


def plot_trials(
    trials,
    output_dir="real_data_trials/results",
    plot_cumulative_for=("reward",),
    file_prefix="comparison",
    textwidth_inches=7.0,
    metadata=None
):
    """
    Plot trial metrics and history with academic style:
      - Saves current-run CSV to history
      - Plots historical mean±STD and cumulative
      - Formal fonts, padding, margins
      - Saves figures as PDF
    """
    if not trials:
        print("No trials provided. Exiting plot.")
        return

    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    history_dir = os.path.join(output_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Extract algorithms and timesteps
    first = trials[0]
    algos = sorted(first.keys())
    metrics = sorted(k for k in first[algos[0]].keys() if k != 'timestep')
    timesteps = np.array(first[algos[0]]['timestep'])
    timesteps_ext = np.concatenate(([0], timesteps))

    # Compute current-run means/stds
    mean_cur = {a: {m: np.mean([t[a][m] for t in trials], axis=0) for m in metrics} for a in algos}
    std_cur  = {a: {m: np.std([t[a][m] for t in trials], axis=0)  for m in metrics} for a in algos}

    # Save current-run CSV
    df = pd.DataFrame({'timestep': timesteps})
    for a in algos:
        for m in metrics:
            df[f"{a}_{m}_mean"] = mean_cur[a][m]
    csv_name = f"{file_prefix}_metrics_{run_id}.csv"
    df.to_csv(os.path.join(history_dir, csv_name), index=False)

    # Load all history files
    files = sorted(glob.glob(os.path.join(history_dir, f"{file_prefix}_metrics_*.csv")))
    if len(files) < 2:
        print("Need at least 2 history runs to plot history.")
        return
    dfs = [pd.read_csv(f) for f in files]
    cols = [c for c in dfs[0].columns if c.endswith('_mean')]
    print(cols)

    # Determine min length to handle shape mismatches
    min_len = min(len(df) for df in dfs)
    print(f"Truncating history to {min_len} timesteps due to length mismatch.")

    # Determine history algos and metrics
    hist_algos = [a for a in legend_order if any(c.startswith(a+'_') for c in cols)]
    print(f'hist{hist_algos}')
    hist_metrics = sorted({c[:-5].split('_',1)[1] for c in cols})
    
    # Truncate to min_len
    stack = {c: np.stack([df[c].values[:min_len] for df in dfs], axis=0) for c in cols}
    
    # Update timesteps for history plotting
    timesteps_hist = dfs[0]['timestep'].values[:min_len]
    timesteps_ext = np.concatenate(([0], timesteps_hist))

    # Style helper
    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    # Figure dimensions
    width, height = textwidth_inches, textwidth_inches * 0.6

    # Plot history and cumulative
    for m in hist_metrics:
        # History
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)
        finals = []
        for a in hist_algos:
            col = f"{a}_{m}_mean"
            data = stack[col]
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            mu_ext = np.concatenate(([0], mu))
            sigma_ext = np.concatenate(([0], sigma))
            finals.append(mu_ext[-1])
            ax.plot(
                timesteps_ext, mu_ext,
                label=formal_name[a],
                color=colors[a],
                marker=markers[a],
                markersize=3,
                linestyle='-'
            )
            ax.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.2,
                color=colors[a]
            )
        lo, hi = min(finals), max(finals)
        margin = 0.1*(hi-lo) if hi>lo else lo*0.1
        ax.set_ylim(lo-margin, hi+margin)
        ax.set_title(f"{m.replace('_',' ').title()} vs. Timestep")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(m.replace('_',' ').title())
        style_ax(ax)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(run_dir, f"{file_prefix}_{m}_history.pdf"), format='pdf', dpi=300)
        plt.close(fig)

        # Cumulative
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)
        finals = []
        for a in hist_algos:
            col = f"{a}_{m}_mean"
            cum = np.cumsum(stack[col], axis=1)
            mu = cum.mean(axis=0)
            sigma = cum.std(axis=0)
            mu_ext = np.concatenate(([0], mu))
            sigma_ext = np.concatenate(([0], sigma))
            finals.append(mu_ext[-1])
            ax.plot(
                timesteps_ext, mu_ext,
                label=formal_name[a],
                color=colors[a],
                marker=markers[a],
                markersize=3,
                linestyle='-'
            )
            ax.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.2,
                color=colors[a]
            )
        lo, hi = min(finals), max(finals)
        margin = 0.1*(hi-lo) if hi>lo else lo*0.1
        ax.set_ylim(lo-margin, hi+margin)
        ax.set_title(f"Cumulative {m.replace('_',' ').title()} vs. Timestep")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"Cumulative {m.replace('_',' ').title()}")
        style_ax(ax)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(run_dir, f"{file_prefix}_{m}_history_cumulative.pdf"), format='pdf', dpi=300)
        plt.close(fig)


def aggregate_history(
    output_dir="real_data_trials/results",
    plot_cumulative_for=("reward",),
    file_prefix="comparison",
    textwidth_inches=7.0,
    auto_scale=False
):
    """
    Aggregated history plots with identical style
    """
    history_dir = os.path.join(output_dir, 'history')
    files = sorted(glob.glob(os.path.join(history_dir, f"{file_prefix}_metrics_*.csv")))
    if len(files) < 1:
        print("No historical CSVs found.")
        return
    dfs = [pd.read_csv(f) for f in files]
    
    # Handle length mismatch by truncating to minimum common length
    min_len = min(len(df) for df in dfs)
    print(f"Truncating history to {min_len} timesteps due to length mismatch.")
    
    timesteps = dfs[0]['timestep'].values[:min_len]
    timesteps_ext = np.concatenate(([0], timesteps))
    cols = [c for c in dfs[0].columns if c.endswith('_mean')]
    hist_algos = [a for a in legend_order if any(c.startswith(a+'_') for c in cols)]
    hist_metrics = sorted({c[:-5].split('_',1)[1] for c in cols})
    
    # stack truncated
    stack = {c: np.stack([df[c].values[:min_len] for df in dfs], axis=0) for c in cols}

    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    # width, height = textwidth_inches, textwidth_inches * 0.6
    width, height = textwidth_inches, 7 * 0.6
    for m in hist_metrics:
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)
        finals=[]
        for a in hist_algos:
            col=f"{a}_{m}_mean"
            if col not in stack: continue
            mu=stack[col].mean(axis=0); sigma=stack[col].std(axis=0)
            mu_ext=np.concatenate(([0],mu)); sigma_ext=np.concatenate(([0],sigma))
            finals.append(mu_ext[-1])
            ax.plot(
                timesteps_ext, mu_ext,
                label=formal_name[a],
                color=colors[a],
                marker=markers[a],
                markersize=3,
                linestyle='-'
            )
            ax.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.2,
                color=colors[a]
            )
        if auto_scale and finals:
            lo, hi = min(finals), max(finals)
            margin = 0.1 * (hi - lo) if hi > lo else lo * 0.1
            ax.set_ylim(lo - margin, hi + margin)    
        ax.set_xlabel("Timestep")
        ax.set_ylabel(m.replace('_',' ').title())
        style_ax(ax)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(output_dir, f"{file_prefix}_{m}_history_mean_std.pdf"), format='pdf', dpi=300)
        plt.close(fig)

        # cumulative
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)
        finals=[]
        for a in hist_algos:
            col=f"{a}_{m}_mean"
            if col not in stack: continue
            cum=np.cumsum(stack[col], axis=1)
            mu=cum.mean(axis=0); sigma=cum.std(axis=0)
            mu_ext=np.concatenate(([0],mu)); sigma_ext=np.concatenate(([0],sigma))
            finals.append(mu_ext[-1])
            ax.plot(
                timesteps_ext, mu_ext,
                label=formal_name[a],
                color=colors[a],
                marker=markers[a],
                markersize=3,
                linestyle='-'
            )
            ax.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.2,
                color=colors[a]
            )
        if auto_scale and finals:
            lo, hi = min(finals), max(finals)
            margin = 0.1 * (hi - lo) if hi > lo else lo * 0.1
            ax.set_ylim(lo - margin, hi + margin)

        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"Cumulative {m.replace('_',' ').title()}")
        style_ax(ax)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(output_dir, f"{file_prefix}_{m}_history_cumulative.pdf"), format='pdf', dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    aggregate_history()