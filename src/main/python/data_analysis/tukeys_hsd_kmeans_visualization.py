import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# File path for Tukey's HSD results for K-means clusters
results_file_path = 'billboard_spotify_analysis/results'
tukey_kmeans_file = os.path.join(results_file_path, 'tukey_hsd_kmeans_results.csv')

# Load the Tukey's HSD results
tukey_kmeans_df = pd.read_csv(tukey_kmeans_file)

# Create a new column 'Comparison' to represent the cluster pair
tukey_kmeans_df['Comparison'] = tukey_kmeans_df['Group1'].astype(str) + ' vs ' + tukey_kmeans_df['Group2'].astype(str)

# Set up the Seaborn style for a clean look
sns.set(style="whitegrid")

# Create a point plot with error bars for the confidence intervals
plt.figure(figsize=(10, 6))
sns.pointplot(
    x='Mean Difference', 
    y='Comparison', 
    data=tukey_kmeans_df,
    join=False, 
    capsize=0.1, 
    color='blue'
)

# Add horizontal lines for confidence intervals
for index, row in tukey_kmeans_df.iterrows():
    plt.plot(
        [row['Lower CI'], row['Upper CI']], 
        [index, index], 
        color='gray', 
        linewidth=0.8
    )

# Add a vertical line at 0 for reference
plt.axvline(0, color='gray', linestyle='--')

# Set the plot title and labels
plt.title("Tukey's HSD Test: Mean Differences in Popularity Across K-means Clusters")
plt.xlabel("Mean Difference in Popularity")
plt.ylabel("Cluster Pair")

# Save the plot
plot_output_file = os.path.join(results_file_path, 'tukey_hsd_kmeans_plot.png')
plt.tight_layout()
plt.savefig(plot_output_file)
plt.show()

print(f"Tukey's HSD test visualization saved to: {plot_output_file}")
