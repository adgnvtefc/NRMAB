# plot_difference_avg_cum_reward.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style and context for better aesthetics and larger fonts
sns.set(style="whitegrid", context="talk")  # 'talk' context increases font sizes suitable for presentations

# Read the combined CSV data
data = pd.read_csv('plots/many_comparisons_300nodes_300edges.csv')  # Update the path if necessary

# List of algorithms to compare against NoSelection
algorithms = ['HillClimb', 'DQN', 'Whittle']
baseline = 'NoSelection'

# Compute average cumulative rewards for each algorithm and the baseline
for algo in algorithms:
    avg_cum_col = f'{algo}_avg_cum_reward'
    cum_reward_col = f'{algo}_cum_reward'
    data[avg_cum_col] = data[cum_reward_col] / data['Timestep']

baseline_avg_cum_col = f'{baseline}_avg_cum_reward'
baseline_cum_reward_col = f'{baseline}_cum_reward'
data[baseline_avg_cum_col] = data[baseline_cum_reward_col] / data['Timestep']

# Calculate differences: ALG_avg_cum_reward - NoSelection_avg_cum_reward
for algo in algorithms:
    diff_col = f'{algo}_diff_avg_cum_reward'
    avg_cum_col = f'{algo}_avg_cum_reward'
    data[diff_col] = data[avg_cum_col] - data[baseline_avg_cum_col]

# Define colors and markers as per your specifications
colors = {
    'HillClimb': 'blue',
    'DQN': 'green',
    'Whittle': 'red'
}

markers = {
    'HillClimb': 'o',  # Circle
    'DQN': 's',        # Square
    'Whittle': '^'     # Triangle
}

# Create a single plot for Difference in Average Cumulative Rewards
plt.figure(figsize=(18, 10))  # Increased figure size for better visibility

# Plot the differences
for algo in algorithms:
    plt.plot(
        data['Timestep'],
        data[f'{algo}_diff_avg_cum_reward'],
        label=f'{algo} vs NoSelection',
        marker=markers[algo],
        color=colors[algo],
        linewidth=3,        # Increased line width for better visibility
        markersize=10       # Increased marker size for better visibility
    )

# Add horizontal dashed line at y=0 for reference
plt.axhline(0, color='black', linewidth=1, linestyle='--')

# Set titles and labels with adjusted font sizes
plt.title('Difference in Average Cumulative Reward per Timestep (ALG - NoSelection)', fontsize=24)
plt.xlabel('Timestep', fontsize=20)
plt.ylabel('Average Cumulative Reward Difference', fontsize=20)
plt.legend(title='Algorithms', fontsize=20, title_fontsize=22)
plt.grid(True)

# Set y-axis to start at 60
plt.ylim(bottom=60)

# Increase tick label font sizes
plt.tick_params(axis='both', which='major', labelsize=22)

# Adjust layout for better spacing
plt.tight_layout()

# Optional: Save the plot as an image file with higher resolution
# Uncomment the following line to save the figure
# plt.savefig('results/difference_avg_cum_reward_comparison.png', dpi=300)

# Show the plot
plt.show()
