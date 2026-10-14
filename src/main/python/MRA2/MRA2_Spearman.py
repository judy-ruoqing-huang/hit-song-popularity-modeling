# Import necessary libraries
from scipy.stats import spearmanr
import pandas as pd

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
y = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Initialize an empty dictionary to store Spearman correlation values
spearman_corr = {}

# Calculate Spearman Rank Correlation for each feature
for feature in columns_to_visualize:
    # Calculate Spearman correlation and p-value
    corr, p_value = spearmanr(data[feature].dropna(), y.loc[data[feature].dropna().index])
    spearman_corr[feature] = corr

# Convert the results to a pandas Series for easy viewing
spearman_corr_series = pd.Series(spearman_corr)

# Output the Spearman correlations
print("Spearman Rank Correlation between Rank Category and Musical Features:\n")
print(spearman_corr_series)

# Output results to a file
output_file_path = 'BillboardExplain/data/spearman_rank_correlation_results.txt'
spearman_corr_series.to_string(output_file_path)

print(f"Results saved to: {output_file_path}")
