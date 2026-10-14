import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# File path
results_file_path = 'billboard_spotify_analysis/results'
stats_file = os.path.join(results_file_path, 'clustering_methods_stats.csv')

# Load data
stats_df = pd.read_csv(stats_file)

# Set index for better heatmap organization (Clustering Method and Cluster)
stats_df.set_index(['Clustering Method', 'Cluster'], inplace=True)

# Select columns with means only for normalization
features_data = stats_df[['Popularity_Mean', 'Loudness_Mean', 'Danceability_Mean', 'Energy_Mean']]

# Normalize the data (0-1 range)
normalized_data = (features_data - features_data.min()) / (features_data.max() - features_data.min())

# Sort the DataFrame by 'Popularity_Mean' for ordering
normalized_data = normalized_data.sort_values(by='Popularity_Mean', ascending=False)

# Generate heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Normalized Mean Value'}, fmt=".2f")
plt.title("Heatmap of Normalized Mean Audio Features and Popularity by Clusters")
plt.xlabel("Popularity and Top 3 Audio Features in PC1")
plt.ylabel("Clusters")
plt.tight_layout()

# Save the plot
heatmap_output_file = os.path.join(results_file_path, 'clustering_methods_stats_heatmap_ordered_by_popularity.png')
plt.savefig(heatmap_output_file)
plt.show()

print(f"Heatmap ordered by Popularity Mean saved to: {heatmap_output_file}")
