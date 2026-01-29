import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from plotting import formal_name, colors, markers, legend_order

sns.set_style('white')
sns.set_context('paper', font_scale=1.0)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.grid': False,
    'text.usetex': False
})

def make_zoomed_plot(
    output_dir="real_data_trials/results",
    file_prefix="comparison",
    target_metric="percent_activated" # Y-range 60-80 implies percent
):
    history_dir = os.path.join(output_dir, 'history')
    # Find latest run or all runs? plotting.py aggregates all history.
    files = sorted(glob.glob(os.path.join(history_dir, f"{file_prefix}_metrics_*.csv")))
    
    if not files:
        print(f"No files found in {history_dir}")
        return

    print(f"Found {len(files)} history files.")
    dfs = [pd.read_csv(f) for f in files]
    
    # Handle length mismatch
    min_len = min(len(df) for df in dfs)
    print(f"Truncating to {min_len} timesteps.")
    
    # We want X axis 0-10. So ensure we have at least 11 points (0..10)
    # The data starts at t=1 usually in the CSV, plotting adds 0.
    
    timesteps = dfs[0]['timestep'].values[:min_len]
    timesteps_ext = np.concatenate(([0], timesteps))
    
    cols = [c for c in dfs[0].columns if c.endswith('_mean')]
    
    # Filter for target metric? Or plot all and crop?
    # User said "zoom in on the graph", implying one specific graph or all.
    # Given the Y-range 60-80, I will prioritize graphs that fit or apply blindly.
    # Let's apply blindly but warn if empty.
    
    hist_algos = [a for a in legend_order if any(c.startswith(a+'_') for c in cols)]
    hist_metrics = sorted({c[:-5].split('_',1)[1] for c in cols})
    
    stack = {c: np.stack([df[c].values[:min_len] for df in dfs], axis=0) for c in cols}
    
    width, height = 7.0, 4.2
    
    for m in hist_metrics:
        print(f"Plotting {m}...")
        
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)
        
        finals = []
        for a in hist_algos:
            col = f"{a}_{m}_mean"
            if col not in stack: continue
            
            data = stack[col]
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            mu_ext = np.concatenate(([0], mu))
            sigma_ext = np.concatenate(([0], sigma))
            
            ax.plot(
                timesteps_ext, mu_ext,
                label=formal_name.get(a, a),
                color=colors.get(a, 'black'),
                marker=markers.get(a, 'o'),
                markersize=4,
                linestyle='-' if a != 'random' else '--'
            )
            ax.fill_between(
                timesteps_ext,
                mu_ext - sigma_ext,
                mu_ext + sigma_ext,
                alpha=0.15,
                color=colors.get(a, 'black')
            )
            
        ax.set_xlabel("Timestep")
        ax.set_title(f"{m.replace('_',' ').title()} (Zoomed)")
        ax.set_ylabel(m.replace('_',' ').title())
        
        # Apply Zoom
        ax.set_xlim(2, 15)
        ax.set_ylim(200, 250)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='lower right')
        
        filename = f"zoomed_{m}.pdf"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, format='pdf', dpi=300)
        print(f"Saved {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    make_zoomed_plot()
