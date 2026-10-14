from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import os

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

# Define a range of linkage methods and cluster counts to test
linkage_methods = ['ward', 'complete', 'average', 'single']
cluster_counts = [2, 3, 4, 5]

best_score = -1
best_config = {}
results = []  # Store all results

# Iterate through linkage methods and cluster counts
for method in linkage_methods:
    Z = linkage(data[['PC1', 'PC2']], method=method)
    for k in cluster_counts:
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(data[['PC1', 'PC2']], labels)
        results.append((method, k, score))
        print(f"Method: {method}, Clusters: {k}, Silhouette Score: {score}")
        
        # Check if this configuration is the best
        if score > best_score:
            best_score = score
            best_config = {'method': method, 'clusters': k, 'score': score}

# Print the best configuration
print(f"Best configuration: {best_config}")

# Output all test results to a CSV file
results_df = pd.DataFrame(results, columns=['Linkage Method', 'Number of Clusters', 'Silhouette Score'])
results_file = os.path.join(results_file_path, 'all_hierarchical_clustering_test_results.csv')
results_df.to_csv(results_file, index=False)

# Output best configuration to a text file
with open(os.path.join(results_file_path, 'best_hierarchical_clustering_config.txt'), 'w') as f:
    f.write(f"Best configuration:\n")
    f.write(f"Linkage Method: {best_config['method']}\n")
    f.write(f"Number of Clusters: {best_config['clusters']}\n")
    f.write(f"Silhouette Score: {best_config['score']:.4f}\n")
