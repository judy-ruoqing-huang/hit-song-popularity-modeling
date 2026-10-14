import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set environment variable to avoid joblib warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust to the actual number of physical cores on your machine.

# Set file paths
results_file_path = 'billboard_spotify_analysis/results'
pca_results_file = os.path.join(results_file_path, 'pca_results.csv')
pca_loadings_file = os.path.join(results_file_path, 'pca_feature_loadings.csv')
explained_variance_file = os.path.join(results_file_path, 'pca_explained_variance_ratio.csv')

# Load data
data = pd.read_csv(pca_results_file)
pca_loadings = pd.read_csv(pca_loadings_file, index_col='Feature')
explained_variance_df = pd.read_csv(explained_variance_file)

# Retrieve explained variance ratio for PC1 and PC2
explained_variance_ratio = explained_variance_df['Explained Variance Ratio'].values[:2]  # First two PCs

# Select features for clustering (using PC1 and PC2)
clustering_features = data[['PC1', 'PC2']]

# Determine optimal K if needed, then apply K-means clustering
optimal_k = 3  # Set to optimal value found from prior analysis or the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Song_Cluster'] = kmeans.fit_predict(clustering_features)

# Identify the top 3 features for PC1 and PC2
top_features_pc1 = pca_loadings['PC1'].nlargest(3)
top_features_pc2 = pca_loadings['PC2'].nlargest(3)

# Plot the clusters
plt.figure(figsize=(12, 8))
cmap = sns.color_palette("pastel", optimal_k)

# Scatter plot of clusters
sns.scatterplot(data=data, x='PC1', y='PC2', hue='Song_Cluster', palette=cmap, s=50, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='X', label='Centroids')

# Label the axes with explained variance
plt.xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% variance explained)")
plt.ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% variance explained)")

# Add text box for top features of PC1 on the bottom right
pc1_text = '\n'.join([f"{feature}: {value:.2f}" for feature, value in top_features_pc1.items()])
plt.text(0.95, 0.05, f"Top PC1 Features:\n{pc1_text}", 
         transform=plt.gca().transAxes, ha='right', va='bottom', 
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'), fontsize=10)

# Add text box for top features of PC2 on the upper left
pc2_text = '\n'.join([f"{feature}: {value:.2f}" for feature, value in top_features_pc2.items()])
plt.text(0.05, 0.95, f"Top PC2 Features:\n{pc2_text}", 
         transform=plt.gca().transAxes, ha='left', va='top', 
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'), fontsize=10)

# Set y-axis limits to improve readability if needed
# plt.ylim(-10, 10)

# Title and legend
plt.title(f"K-means Clustering on Songs with Top Features (K={optimal_k})")
plt.legend(loc='best', title='Song Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show the plot
clustering_plot_file = os.path.join(results_file_path, 'kmeans_cluster_scatter_plot_with_top_features.png')
plt.savefig(clustering_plot_file, dpi=300)
plt.show()
print(f"K-means clustering plot saved to {clustering_plot_file}")
