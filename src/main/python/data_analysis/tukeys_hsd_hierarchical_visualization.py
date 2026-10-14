import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
tukey_hsd_file = os.path.join(results_file_path, 'tukey_hsd_hierarchical_results.csv')
plot_output_file = os.path.join(results_file_path, 'tukey_hsd_hierarchical_plot.png')

# Load the Tukey HSD results
tukey_df = pd.read_csv(tukey_hsd_file)

# Combine Group1 and Group2 into a single column for the plot
tukey_df['Cluster Pair'] = tukey_df['Group1'].astype(str) + " vs " + tukey_df['Group2'].astype(str)

# Set up the Seaborn style for clean visuals
sns.set(style="whitegrid")

# Create a point plot for the Mean Difference with error bars for confidence intervals
plt.figure(figsize=(10, 6))
sns.pointplot(
    x='Mean Difference', 
    y='Cluster Pair', 
    data=tukey_df, 
    join=False, 
    capsize=0.1, 
    color="blue",
    markers='o',
    linestyles='none'  # Ensure points are not connected
)

# Add error bars for confidence intervals
for index, row in tukey_df.iterrows():
    plt.errorbar(
        x=row['Mean Difference'], 
        y=index, 
        xerr=[[row['Mean Difference'] - row['Lower CI']], [row['Upper CI'] - row['Mean Difference']]], 
        fmt='o', 
        color='blue'
    )

# Add a vertical line at x=0 for reference
plt.axvline(0, color='gray', linestyle='--')

# Customize the title and labels
plt.title("Tukey's HSD Test: Mean Differences in Popularity Across Hierarchical Clusters")
plt.xlabel("Mean Difference in Popularity")
plt.ylabel("Cluster Pair")

# Save the plot
plt.tight_layout()
plt.savefig(plot_output_file)
plt.show()

print(f"Tukey's HSD visualization saved to: {plot_output_file}")
