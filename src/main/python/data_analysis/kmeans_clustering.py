import pandas as pd
from sklearn.cluster import KMeans
import os

# Set the base data path for both raw and processed data
data_file_path = 'billboard_spotify_analysis/data'
results_file_path = 'billboard_spotify_analysis/results'

# File paths
pca_results_file = os.path.join(results_file_path, 'pca_results.csv')
output_file = os.path.join(results_file_path, 'kmeans_clustering_results.csv')

# Load PCA results
pca_df = pd.read_csv(pca_results_file)

# Check if the 'Cluster' column is already in the dataframe
if 'Cluster' not in pca_df.columns:
    # Define the optimal number of clusters
    n_clusters = 3

    # Select only the PCA components for clustering
    pca_components = pca_df[[f'PC{i+1}' for i in range(21)]]  # Using PC1 to PC21

    # Perform K-means clustering with k=3 and add the Cluster column
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pca_df['Cluster'] = kmeans.fit_predict(pca_components)

# Save the updated DataFrame with the Cluster column to a new CSV file
pca_df.to_csv(output_file, index=False)
print(f"K-means clustering results saved to: {output_file}")

# Display the first few rows to verify the output
print(pca_df.head())
