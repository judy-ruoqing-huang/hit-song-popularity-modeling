# Import necessary libraries
from scipy.stats import kendalltau
import pandas as pd

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
y = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Initialize dictionaries to store Kendall's Tau values and p-values
kendall_tau_corr = {}
p_values = {}

# Calculate Kendall's Tau for each feature
for feature in columns_to_visualize:
    # Calculate Kendall's Tau correlation and p-value
    tau, p_value = kendalltau(data[feature].dropna(), y.loc[data[feature].dropna().index])
    kendall_tau_corr[feature] = tau
    p_values[feature] = p_value

# Convert the results to pandas Series for easy viewing
kendall_tau_corr_series = pd.Series(kendall_tau_corr, name="Kendall's Tau")
p_values_series = pd.Series(p_values, name="p-value")

# Combine the Kendall's Tau and p-values into a single DataFrame
kendall_results = pd.DataFrame({'Kendall\'s Tau': kendall_tau_corr_series, 'p-value': p_values_series})

# Output the Kendall's Tau correlations and p-values
print("Kendall's Tau and p-values between Rank Category and Musical Features:\n")
print(kendall_results)

# Output results to a file
output_file_path = 'BillboardExplain/data/kendalls_tau_correlation_with_p_values_results.txt'
kendall_results.to_string(output_file_path)

print(f"Results saved to: {output_file_path}")
