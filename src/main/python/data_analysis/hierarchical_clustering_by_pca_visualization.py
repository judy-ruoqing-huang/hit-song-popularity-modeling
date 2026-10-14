import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')
dendrogram_file = os.path.join(results_file_path, 'hierarchical_dendrogram.png')

# Load PCA components from the hierarchical clustering results file
pca_df = pd.read_csv(hierarchical_results_file)
pca_components = pca_df[[f'PC{i+1}' for i in range(21)]]  # Using PC1 to PC21

# Perform hierarchical clustering linkage for dendrogram
linkage_matrix = linkage(pca_components, method='ward')

# Plot and save the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.title("Dendrogram of Hierarchical Clustering on PCA")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(dendrogram_file)
plt.show()
print(f"Dendrogram saved to: {dendrogram_file}")
