import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define base directory paths
results_file_path = 'billboard_spotify_analysis/results'

# File paths for input data and output images
cluster_stats_file = os.path.join(results_file_path, 'kmeans_top_features_popularity_stats_by_cluster.csv')
kmeans_results_file = os.path.join(results_file_path, 'kmeans_clustering_results.csv')
top_features_plot_file = os.path.join(results_file_path, 'kmeans_top_features_stats_by_cluster.png')
popularity_plot_file = os.path.join(results_file_path, 'kmeans_popularity_distribution_by_cluster.png')
audio_features_plot_file = os.path.join(results_file_path, 'kmeans_audio_features_by_cluster.png')

# Titles for plots
top_features_title = 'Mean of Top Audio Features by Cluster'
popularity_title = 'Popularity Distribution by Cluster'

# Load the cluster statistics data
cluster_stats = pd.read_csv(cluster_stats_file)

# Dynamically identify the top audio features (excluding Popularity)
top_audio_features = [col.replace('_mean', '') for col in cluster_stats.columns if '_mean' in col and 'Popularity' not in col]

# Set up Seaborn style for clean visuals
sns.set(style="whitegrid")

# --- Visualization 1: Bar Plot for Mean of Top Audio Features by Cluster (Excluding Popularity) ---

# Extract mean values for the top audio features by cluster
audio_features_means = cluster_stats[['Cluster'] + [f'{feature}_mean' for feature in top_audio_features]]

# Melt the DataFrame for easy plotting with Seaborn
audio_features_means_melted = audio_features_means.melt(id_vars='Cluster', var_name='Feature', value_name='Mean')
audio_features_means_melted['Feature'] = audio_features_means_melted['Feature'].str.replace('_mean', '')  # Clean up feature names

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='Mean', hue='Feature', data=audio_features_means_melted, palette="Set2")
plt.title(top_features_title)
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.legend(title='Feature')
plt.tight_layout()

# Save the figure
plt.savefig(audio_features_plot_file)
plt.show()

# --- Visualization 2: Box Plot for Popularity by Cluster ---

# Load the K-means results with clusters
kmeans_df = pd.read_csv(kmeans_results_file)

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Popularity', hue='Cluster', data=kmeans_df, palette="Set3", legend=False)
plt.title(popularity_title)
plt.xlabel('Cluster')
plt.ylabel('Popularity')
plt.tight_layout()

# Save the figure
plt.savefig(popularity_plot_file)
plt.show()
