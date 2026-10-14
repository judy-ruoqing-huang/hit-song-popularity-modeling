import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')
tukey_output_csv = os.path.join(results_file_path, 'tukey_hsd_hierarchical_results.csv')

# Load the hierarchical clustering results
hierarchical_df = pd.read_csv(hierarchical_results_file)

# Perform Tukey's HSD test for pairwise comparisons of popularity across clusters
tukey_results = pairwise_tukeyhsd(endog=hierarchical_df['Popularity'], groups=hierarchical_df['Cluster'], alpha=0.05)

# Extract Tukey HSD results and create a DataFrame
tukey_df = pd.DataFrame(data=tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])

# Rename columns for clarity
tukey_df.columns = ["Group1", "Group2", "Mean Difference", "P-Value", "Lower CI", "Upper CI", "Reject"]

# Save the Tukey test results to a CSV file
tukey_df.to_csv(tukey_output_csv, index=False)
print(f"Tukey's HSD test results saved to CSV: {tukey_output_csv}")
