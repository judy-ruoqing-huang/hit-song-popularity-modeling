import pandas as pd
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
kmeans_file = os.path.join(results_file_path, 'kmeans_clustering_results.csv')
hierarchical_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')
output_file = os.path.join(results_file_path, 'clustering_methods_stats.csv')

# Function to format numbers consistently (e.g., scientific notation for small numbers)
def format_number(num):
    if pd.isna(num):  # Check for NaN
        return num
    elif abs(num) < 0.001:
        return f"{num:.2e}"  # Scientific notation for very small numbers
    else:
        return round(num, 4)  # Round to four decimal places

# Function to calculate mean and std for each cluster and format numbers
def calculate_cluster_stats(df, method_name):
    # Calculate mean and std for each feature by cluster
    stats_df = df.groupby('Cluster').agg(
        Popularity_Mean=('Popularity', 'mean'),
        Popularity_Std=('Popularity', 'std'),
        Loudness_Mean=('Loudness', 'mean'),
        Loudness_Std=('Loudness', 'std'),
        Danceability_Mean=('Danceability', 'mean'),
        Danceability_Std=('Danceability', 'std'),
        Energy_Mean=('Energy', 'mean'),
        Energy_Std=('Energy', 'std')
    ).reset_index()

    # Apply number formatting to each column
    for col in stats_df.columns[1:]:  # Skip 'Clustering Method'
        stats_df[col] = stats_df[col].apply(format_number)

    # Add a column for the clustering method
    stats_df.insert(0, 'Clustering Method', method_name)
    return stats_df

# Load data for each clustering method
kmeans_df = pd.read_csv(kmeans_file)
hierarchical_df = pd.read_csv(hierarchical_file)

# Calculate statistics for each method
kmeans_stats = calculate_cluster_stats(kmeans_df, 'K-means')
hierarchical_stats = calculate_cluster_stats(hierarchical_df, 'Hierarchical')

# Combine results
combined_stats = pd.concat([kmeans_stats, hierarchical_stats], ignore_index=True)

# Save to CSV
combined_stats.to_csv(output_file, index=False)
print(f"Clustering statistics saved to: {output_file}")
