import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

def analyze_and_plot_sem(
    output_dir="real_data_trials/results",
    file_prefix="comparison"
):
    history_dir = os.path.join(output_dir, 'history')
    files = sorted(glob.glob(os.path.join(history_dir, f"{file_prefix}_metrics_*.csv")))
    
    if not files:
        print(f"No files found in {history_dir}")
        return

    print(f"Found {len(files)} history files.")
    dfs = [pd.read_csv(f) for f in files]
    n_runs = len(dfs)
    
    min_len = min(len(df) for df in dfs)
    
    timesteps = dfs[0]['timestep'].values[:min_len]
    timesteps_ext = np.concatenate(([0], timesteps))
    
    cols = [c for c in dfs[0].columns if c.endswith('_mean')]
    hist_algos = [a for a in legend_order if any(c.startswith(a+'_') for c in cols)]
    hist_metrics = sorted({c[:-5].split('_',1)[1] for c in cols})
    
    stack = {c: np.stack([df[c].values[:min_len] for df in dfs], axis=0) for c in cols}
    
    # Perform t-test for final reward
    if 'reward' in hist_metrics and 'cdsqn' in hist_algos:
        m = 'reward'
        cdsqn_col = f"cdsqn_{m}_mean"
        cdsqn_final = stack[cdsqn_col][:, -1]
        
        best_baseline = None
        best_baseline_val = -float('inf')
        
        for a in hist_algos:
            if a == 'cdsqn': continue
            col = f"{a}_{m}_mean"
            if col not in stack: continue
            mean_val = stack[col][:, -1].mean()
            if mean_val > best_baseline_val:
                best_baseline_val = mean_val
                best_baseline = a
        
        print("\n" + "="*40)
        print("STATISTICAL SIGNIFICANCE TESTS (Final Reward)")
        print(f"Sample Size (n_runs): {n_runs}")
        
        for a in hist_algos:
            if a == 'cdsqn': continue
            col = f"{a}_{m}_mean"
            if col not in stack: continue
            
            baseline_final = stack[col][:, -1]
            t_stat_rel, p_val_rel = stats.ttest_rel(cdsqn_final, baseline_final)
            
            print(f"\n--- CDSQN vs. {formal_name.get(a, a)} ---")
            print(f"Mean {formal_name.get(a, a)}: {baseline_final.mean():.4f}")
            print(f"Difference: {cdsqn_final.mean() - baseline_final.mean():.4f}")
            print(f"P-value (Paired): {p_val_rel:.6f}")
            
        print("="*40 + "\n")

    width, height = 7.0, 4.2
    
    for m in hist_metrics:
        print(f"Plotting {m} (SEM)...")
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)
        
        for a in hist_algos:
            col = f"{a}_{m}_mean"
            if col not in stack: continue
            
            data = stack[col]
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            sem = sigma / np.sqrt(n_runs) # Standard Error of the Mean
            
            mu_ext = np.concatenate(([0], mu))
            sem_ext = np.concatenate(([0], sem))
            
            ax.plot(
                timesteps_ext, mu_ext,
                label=formal_name.get(a, a),
                color=colors.get(a, 'black'),
                marker=markers.get(a, 'o'),
                markersize=4,
                linestyle='-'
            )
            ax.fill_between(
                timesteps_ext,
                mu_ext - sem_ext,
                mu_ext + sem_ext,
                alpha=0.2,
                color=colors.get(a, 'black')
            )
            
        ax.set_xlabel("Timestep")
        ax.set_ylabel(m.replace('_',' ').title())
        ax.set_title(f"{m.replace('_',' ').title()} (SEM)")
        
        # Zoom parameters requested: X:0-10, Y:60-80
        ax.set_xlim(2, 15)
        ax.set_ylim(200, 250)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='lower right')
        
        save_path = os.path.join(output_dir, f"zoomed_sem_{m}.pdf")
        fig.savefig(save_path, format='pdf', dpi=300)
        print(f"Saved {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    analyze_and_plot_sem()
