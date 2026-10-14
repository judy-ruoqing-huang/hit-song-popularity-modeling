import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import os

# File paths
data_file_path = 'billboard_spotify_analysis/data'
results_file_path = 'billboard_spotify_analysis/results'
pca_results_file = os.path.join(results_file_path, 'pca_results.csv')
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')

# Load the PCA results
pca_df = pd.read_csv(pca_results_file)

# Select the PCA components for clustering
pca_components = pca_df[[f'PC{i+1}' for i in range(21)]]  # Using PC1 to PC21

# Perform hierarchical clustering with a chosen number of clusters
n_clusters = 3  # Modify based on optimal cluster determination
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
pca_df['Cluster'] = hierarchical.fit_predict(pca_components)

# Save clustering results to a file
pca_df.to_csv(hierarchical_results_file, index=False)
print(f"Hierarchical clustering results saved to: {hierarchical_results_file}")
