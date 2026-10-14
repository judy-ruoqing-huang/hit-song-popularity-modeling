import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')
plot_output_file = os.path.join(results_file_path, 'anova_hierarchical_plot.png')

# Customizable plot title
plot_title = "ANOVA Test: Popularity Distribution Across Hierarchical Clusters"

# Load the hierarchical clustering results
hierarchical_df = pd.read_csv(hierarchical_results_file)

# Set up the Seaborn style for a clean look
sns.set(style="whitegrid")

# Create a box plot to show the distribution of Popularity by Cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Popularity', hue='Cluster', data=hierarchical_df, palette="Set2", legend=False)
plt.title(plot_title)
plt.xlabel("Cluster")
plt.ylabel("Popularity")

plt.tight_layout()

# Save the plot
plt.savefig(plot_output_file)
plt.show()

print(f"ANOVA test visualization saved to: {plot_output_file}")
