import pandas as pd
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')

# File paths
loadings_file = os.path.join(results_file_path, 'pca_feature_loadings.csv')
cluster_stats_file = os.path.join(results_file_path, 'hierarchical_top_features_popularity_stats_by_cluster.csv')

# Load clustering results and PCA loadings
pca_df = pd.read_csv(hierarchical_results_file)
loadings_df = pd.read_csv(loadings_file)

# Identify the top 3 audio features contributing to PC1
top_features = loadings_df[['Feature', 'PC1']].set_index('Feature').abs().nlargest(3, 'PC1').index.tolist()

# Calculate mean and std for each of the top features and Popularity by cluster
cluster_stats = pca_df.groupby('Cluster')[top_features + ['Popularity']].agg(['mean', 'std'])

# Flatten column multi-index for readability
cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

# Save the statistics to a file
cluster_stats.to_csv(cluster_stats_file)
print(f"Top features and popularity statistics by cluster saved to: {cluster_stats_file}")
