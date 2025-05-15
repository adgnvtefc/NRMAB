# plot_graphs_raw.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style and context for better aesthetics and larger fonts
sns.set(style="whitegrid", context="talk")  # 'talk' context increases font sizes suitable for presentations

# Read the combined CSV data
# Ensure that 'plots/tab_bell_dqn_comparison.csv' is in the same directory as this script
data = pd.read_csv('plots/tab_bell_dqn_comparison.csv')  # Update the path if necessary

# List of algorithms to plot
algorithms = ['Tab_Bell', 'DQN']
algorithm_display_names = {
    'Tab_Bell': 'Tabular-Q',
    'DQN': 'DQN'
}

# Define a color palette for consistency
colors = {
    'Tabular-Q': 'black',
    'DQN': 'green'
}

# Define markers for distinction
markers = {
    'Tabular-Q': 'o',  # Circle
    'DQN': 's'          # Square
}

# Create a single subplot for Mean Reward Comparison
plt.figure(figsize=(18, 10))  # Adjusted figure size for better visibility

# Plot Mean Reward per Timestep (Tabular-Q and DQN)
for algo in algorithms:
    display_name = algorithm_display_names[algo]
    plt.plot(
        data['Timestep'],
        data[f'{algo}_mean_reward'],
        label=f'{display_name} Mean Reward',
        marker=markers[display_name],
        color=colors[display_name],
        linewidth=3,        # Increased line width for better visibility
        markersize=10       # Increased marker size for better visibility
    )

# Set titles and labels with adjusted font sizes
plt.title('Mean Reward per Timestep (Tabular-Q vs DQN)', fontsize=24)  # Increased title font size
plt.xlabel('Timestep', fontsize=20)                                      # Increased X-axis label font size
plt.ylabel('Mean Reward', fontsize=20)                                   # Increased Y-axis label font size
plt.legend(title='Algorithms', fontsize=20, title_fontsize=22)           # Increased legend font sizes
plt.grid(True)

# Increase tick label font sizes
plt.tick_params(axis='both', which='major', labelsize=22)

# Adjust layout for better spacing
plt.tight_layout()

# Optional: Save the plot as an image file with higher resolution
# Uncomment the following line to save the figure
# plt.savefig('results/mean_reward_comparison_combined.png', dpi=300)

# Show the plot
plt.show()
