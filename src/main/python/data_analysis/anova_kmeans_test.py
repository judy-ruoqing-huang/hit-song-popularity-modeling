import pandas as pd
from scipy.stats import f_oneway
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
kmeans_results_file = os.path.join(results_file_path, 'kmeans_clustering_results.csv')
anova_output_csv = os.path.join(results_file_path, 'anova_kmeans_results.csv')

# Load the K-means clustering results
kmeans_df = pd.read_csv(kmeans_results_file)

# Separate the Popularity scores by each K-means cluster
clusters = kmeans_df['Cluster'].unique()
popularity_by_cluster = [kmeans_df[kmeans_df['Cluster'] == cluster]['Popularity'] for cluster in clusters]

# Perform the ANOVA test
f_stat, p_value = f_oneway(*popularity_by_cluster)

# Prepare results for CSV output
anova_results_df = pd.DataFrame({
    'F-statistic': [f_stat],
    'P-value': [p_value],
    'Significant Difference': ['Yes' if p_value < 0.05 else 'No']
})

# Save the results to a CSV file
anova_results_df.to_csv(anova_output_csv, index=False)
print(f"ANOVA test results saved to CSV file: {anova_output_csv}")
