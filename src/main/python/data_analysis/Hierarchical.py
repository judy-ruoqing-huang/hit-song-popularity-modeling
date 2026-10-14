import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Set the results path for output files
results_file_path = 'billboard_spotify_analysis/results'
os.makedirs(results_file_path, exist_ok=True)

# Load PCA feature loadings data
loadings_file = os.path.join(results_file_path, 'pca_feature_loadings.csv')
loadings_df = pd.read_csv(loadings_file)

# Extract features and PCA loadings for clustering
features = loadings_df['Feature']
loadings_data = loadings_df.drop(columns=['Feature'])

# Perform hierarchical clustering on features based on PCA loadings
linked = linkage(loadings_data, method='ward')

# Plot and save the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, labels=features.values, orientation='right', leaf_font_size=10, color_threshold=0.5)

# Set title and labels
plt.title('Hierarchical Clustering Dendrogram of Audio Features')
plt.xlabel('Distance')
plt.ylabel('Features')

# Save dendrogram as PNG file
dendrogram_file = os.path.join(results_file_path, 'hierarchical_dendrogram_features_pca.png')
plt.savefig(dendrogram_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Hierarchical clustering dendrogram saved to {dendrogram_file}")
