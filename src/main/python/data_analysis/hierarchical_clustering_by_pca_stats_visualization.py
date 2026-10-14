import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
import seaborn as sns

# File paths
results_file_path = 'billboard_spotify_analysis/results'
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')
cluster_stats_file = os.path.join(results_file_path, 'hierarchical_top_features_popularity_stats_by_cluster.csv')

# File paths
audio_features_plot_file = os.path.join(results_file_path, 'hierarchical_audio_features_by_cluster.png')
popularity_plot_file = os.path.join(results_file_path, 'hierarchical_popularity_distribution_by_cluster.png')

# Load the cluster statistics data
cluster_stats = pd.read_csv(cluster_stats_file)

# Identify top audio features
top_audio_features = [col.replace('_mean', '') for col in cluster_stats.columns if '_mean' in col and 'Popularity' not in col]

# --- Visualization 1: Bar Plot for Mean and Standard Deviation of Top Audio Features by Cluster ---
audio_features_means = cluster_stats[['Cluster'] + [f'{feature}_mean' for feature in top_audio_features]]

# Melt the DataFrame for easy plotting with Seaborn
audio_features_means_melted = pd.melt(audio_features_means, id_vars='Cluster', 
                                      var_name='Feature', value_name='Mean')
audio_features_means_melted['Feature'] = audio_features_means_melted['Feature'].str.replace('_mean', '')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='Mean', hue='Feature', data=audio_features_means_melted, palette="Set2")
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.legend(title='Feature')
plt.tight_layout()
plt.savefig(audio_features_plot_file)
plt.show()

# --- Visualization 2: Box Plot for Popularity by Cluster ---
popularity_stats = cluster_stats[['Cluster', 'Popularity_mean', 'Popularity_std']]

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Cluster', y='Popularity_mean', data=popularity_stats, palette="Set3", ci=None)
plt.errorbar(popularity_stats['Cluster'], popularity_stats['Popularity_mean'], 
             yerr=popularity_stats['Popularity_std'], fmt='none', color='gray', capsize=5)
plt.xlabel('Cluster')
plt.ylabel('Popularity')
plt.tight_layout()
plt.savefig(popularity_plot_file)
plt.show()
