import pandas as pd
import os

# Define the file paths
data_file_path = 'billboard_spotify_analysis/data'
results_file_path = 'billboard_spotify_analysis/results'

# File paths for the input and output files
loadings_file = os.path.join(results_file_path, 'pca_feature_loadings.csv')
kmeans_results_file = os.path.join(results_file_path, 'kmeans_clustering_results.csv')
stats_file = os.path.join(results_file_path, 'kmeans_top_features_popularity_stats_by_cluster.csv')

# Load PCA loadings and K-means clustering results (including 'Cluster' column)
loadings_df = pd.read_csv(loadings_file)
kmeans_df = pd.read_csv(kmeans_results_file)

# Identify the top 3 audio features contributing to PC1
top_features = loadings_df[['Feature', 'PC1']].set_index('Feature').abs().nlargest(3, 'PC1').index.tolist()

# Calculate mean and std for each of the top features and Popularity by cluster
try:
    cluster_stats = kmeans_df.groupby('Cluster')[top_features + ['Popularity']].agg(['mean', 'std'])
except KeyError as e:
    print(f"Error: {e}")
    print("Columns available in kmeans_df:", kmeans_df.columns)

# Flatten column multi-index for readability
cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

# Save the statistics to a CSV file
cluster_stats.to_csv(stats_file)
print(f"Statistics of top 3 audio features and popularity by cluster saved to: {stats_file}")

# Display the statistics
print("Statistics of Top 3 Audio Features and Popularity by Cluster:")
print(cluster_stats)
