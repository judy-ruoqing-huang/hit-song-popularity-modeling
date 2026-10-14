import pandas as pd
from scipy.stats import f_oneway
import os

# File paths
results_file_path = 'billboard_spotify_analysis/results'
hierarchical_results_file = os.path.join(results_file_path, 'hierarchical_clustering_results.csv')
anova_output_file = os.path.join(results_file_path, 'anova_hierarchical_results.csv')

# Load the hierarchical clustering results
hierarchical_df = pd.read_csv(hierarchical_results_file)

# Calculate ANOVA test statistics
clusters = hierarchical_df['Cluster'].unique()
popularity_by_cluster = [hierarchical_df[hierarchical_df['Cluster'] == cluster]['Popularity'] for cluster in clusters]
f_stat, p_value = f_oneway(*popularity_by_cluster)

# Save the ANOVA results to a CSV file
anova_results_df = pd.DataFrame({
    'F-statistic': [f_stat],
    'P-value': [p_value]
})
anova_results_df.to_csv(anova_output_file, index=False)

print(f"ANOVA test results saved to: {anova_output_file}")
