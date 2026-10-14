import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set file paths (adjust paths as needed)
results_file_path = 'billboard_spotify_analysis/results'
pca_results_file = os.path.join(results_file_path, 'pca_results.csv')

# Define output file paths for the plots
elbow_plot_file = os.path.join(results_file_path, 'elbow_method_plot.png')
silhouette_plot_file = os.path.join(results_file_path, 'silhouette_analysis_plot.png')

# Load the PCA results data
data = pd.read_csv(pca_results_file)

# Select features for clustering (using PC1 and PC2)
clustering_features = data[['PC1', 'PC2']]

# Define the range of K values to test
K_range = range(2, 11)  # Start from 2 clusters since silhouette needs at least 2

# Elbow Method: Calculate Within-Cluster Sum of Squares (WCSS) for each K
wcss = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(clustering_features)
    wcss.append(kmeans.inertia_)

# Silhouette Analysis: Calculate silhouette score for each K
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_features)
    silhouette_avg = silhouette_score(clustering_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot and save the Elbow Method results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o', linestyle='-', color='b')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.grid(True, linestyle='--', alpha=0.7)

# Save Elbow plot
plt.savefig(elbow_plot_file, dpi=300)
print(f"Elbow method plot saved to {elbow_plot_file}")

# Plot and save the Silhouette Analysis results
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='g')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal K')
plt.grid(True, linestyle='--', alpha=0.7)

# Save Silhouette plot
plt.savefig(silhouette_plot_file, dpi=300)
print(f"Silhouette analysis plot saved to {silhouette_plot_file}")

plt.tight_layout()
plt.show()
