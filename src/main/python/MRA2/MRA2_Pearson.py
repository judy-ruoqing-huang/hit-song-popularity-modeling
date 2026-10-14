# Import necessary libraries
from scipy.stats import pearsonr
import pandas as pd

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
y = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Initialize dictionaries to store Pearson correlation values and p-values
pearson_corr = {}
p_values = {}

# Calculate Pearson correlation for each feature
for feature in columns_to_visualize:
    # Calculate Pearson correlation and p-value
    corr, p_value = pearsonr(data[feature].dropna(), y.loc[data[feature].dropna().index])
    pearson_corr[feature] = corr
    p_values[feature] = p_value

# Convert the results to pandas Series for easy viewing
pearson_corr_series = pd.Series(pearson_corr, name="Pearson Correlation")
p_values_series = pd.Series(p_values, name="p-value")

# Combine the Pearson Correlation and p-values into a single DataFrame
pearson_results = pd.DataFrame({'Pearson Correlation': pearson_corr_series, 'p-value': p_values_series})

# Output the Pearson correlations and p-values
print("Pearson Correlation and p-values between Rank Category and Musical Features:\n")
print(pearson_results)

# Output results to a file
output_file_path = 'BillboardExplain/data/pearson_correlation_with_p_values_results.txt'
pearson_results.to_string(output_file_path)

print(f"Results saved to: {output_file_path}")
