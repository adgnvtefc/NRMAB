import os
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn and matplotlib style
sns.set_style('white')
sns.set_context('paper', font_scale=1.0)
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

# Formal names and styling maps
formal_name = {
    'cdsqn':   'CDSQN',
    'dqn':     'DQN',
    'tabular': 'Tabular'
}
colors = {
    'cdsqn':   '#ff7f0e',  # orange
    'dqn':     "#bbea37",  # lime
    'tabular': '#17becf'   # cyan
}
markers = {
    'cdsqn':   's',  # square
    'dqn':     '.',  # dot
    'tabular': '*'   # star
}
linestyles = {
    'cdsqn':   '-',
    'dqn':     '-',
    'tabular': '-'
}

# Legend order
legend_order = ['cdsqn', 'dqn', 'tabular']

def plot_compute_cost(
    nodes,
    tab_time,
    dqn_time,
    cdsqn_time,
    output_dir="real_data_trials/c_cost",
    file_name="computational_cost",
    textwidth_inches=7.0,
    auto_scale=True
):
    """
    Plot computational cost (seconds) vs. number of nodes:
      - Matches layout & styling of main plots
      - Fixed color, marker, and linestyle per algorithm
      - Removes grid, optional autoscale
      - Consistent font sizes and margins
      - Saves as PDF
    Args:
      auto_scale (bool): if False, disable y-axis autoscaling
    """
    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare figure
    width, height = textwidth_inches, textwidth_inches * 0.6
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.88)

    # Helper to style axes
    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    # Plot lines in legend_order: CDSQN, DQN, Tabular
    finals = []
    series = {
        'cdsqn': cdsqn_time,
        'dqn':   dqn_time,
        'tabular': tab_time
    }
    for key in legend_order:
        data = series[key]
        ax.plot(
            nodes, data,
            label=formal_name[key],
            color=colors[key],
            marker=markers[key],
            markersize=4,
            linestyle=linestyles[key]
        )
        finals.append(data[-1])

    # Optional autoscale Y-axis based on final values
    if auto_scale and finals:
        lo, hi = min(finals), max(finals)
        margin = 0.1 * (hi - lo) if hi > lo else lo * 0.1
        ax.set_ylim(lo - margin, hi + margin)

    # Labels and title
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Time (seconds)")

    # Style axes and legend
    style_ax(ax)
    ax.legend(frameon=False)

    ax.set_xticks(nodes)
    ax.set_xlim(nodes[0], nodes[-1])

    # Save as PDF
    out_path = os.path.join(output_dir, f"{file_name}.pdf")
    fig.savefig(out_path, format='pdf', dpi=300)
    plt.close(fig)

    print(f"Saved computational cost plot to {out_path}")

if __name__ == '__main__':
    nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    tab_time = [8.2575, 11.3764, 13.7647, 17.8291, 25.724, 32.3468, 48.4678, 57.7142, 51.7331, 59.4346]
    dqn_time = [0.7124, 0.161, 0.1861, 0.2, 0.2141, 0.2434, 0.2462, 0.2402, 0.2624, 0.2626]
    cdsqn_time = [0.6026, 0.6217, 0.6994, 0.7446, 0.7955, 0.9035, 1.0384, 0.8986, 1.0002, 1.0322]

    # call with auto_scale=True or False
    plot_compute_cost(nodes, tab_time, dqn_time, cdsqn_time, auto_scale=False)
