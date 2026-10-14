# File: tukey_hsd_kmeans_test.py
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
kmeans_results_file = os.path.join(results_file_path, 'kmeans_clustering_results.csv')
tukey_output_file = os.path.join(results_file_path, 'tukey_hsd_kmeans_results.csv')

# Load K-means clustering results
kmeans_df = pd.read_csv(kmeans_results_file)

# Perform Tukey's HSD test on Popularity by Cluster
tukey_results = pairwise_tukeyhsd(endog=kmeans_df['Popularity'], groups=kmeans_df['Cluster'], alpha=0.05)

# Convert the summary table to a DataFrame
tukey_summary = tukey_results.summary()
tukey_df = pd.DataFrame(data=tukey_summary.data[1:], columns=tukey_summary.data[0])

# Rename columns for consistency
tukey_df.columns = ["Group1", "Group2", "Mean Difference", "P-Value", "Lower CI", "Upper CI", "Reject"]

# Save results to CSV
tukey_df.to_csv(tukey_output_file, index=False)
print(f"Tukey's HSD test results saved to: {tukey_output_file}")
