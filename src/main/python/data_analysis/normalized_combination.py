import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Set the base file path for the results directory
results_file_path = 'billboard_spotify_analysis/results'

# Define the full paths for the correlation, regression, and ensemble results files
correlation_file = os.path.join(results_file_path, 'correlation_results.csv')
regression_file = os.path.join(results_file_path, 'regression_results.csv')
ensemble_file = os.path.join(results_file_path, 'ensemble_feature_importances.csv')

# Load the data from the three CSV files
correlation_data = pd.read_csv(correlation_file)
regression_data = pd.read_csv(regression_file)
ensemble_data = pd.read_csv(ensemble_file)

# Ensure the column names are meaningful in all datasets
# Correlation data columns: 'Feature', 'Pearson Correlation (r)', 'Spearman Correlation'
# Regression data columns: 'Feature', 'Lasso Coefficients', 'Elastic Net Coefficients', 'Linear Regression Coefficients'
# Ensemble data columns: 'Feature', 'Random Forest Importance', 'Gradient Boosting Importance'

# Merge the dataframes on the 'Feature' column
combined_data = pd.merge(correlation_data[['Feature', 'Pearson Correlation (r)', 'Spearman’s Rank Correlation Coefficient (ρ)']],
                         regression_data[['Feature', 'Lasso Coefficients', 'Elastic Net Coefficients', 'Linear Regression Coefficients']],
                         on='Feature')

combined_data = pd.merge(combined_data,
                         ensemble_data[['Feature', 'Random Forest Importance', 'Gradient Boosting Importance']],
                         on='Feature')

# Set the 'Feature' column as the index for the combined dataframe
combined_data.set_index('Feature', inplace=True)

# Normalize the data (since different metrics may have different scales, normalization ensures uniformity for clustering)
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(combined_data), columns=combined_data.columns, index=combined_data.index)

# Function to format numbers consistently (e.g., 0.0001 or scientific notation for small numbers)
def format_number(num):
    if abs(num) < 0.001:
        return f"{num:.2e}"  # Scientific notation for very small numbers
    else:
        return round(num, 4)  # Round to four decimal places for other numbers

# Apply the formatting function to the normalized data
formatted_data = normalized_data.applymap(format_number)

# Define the path for saving the formatted combined data
formatted_data_file = os.path.join(results_file_path, 'normalized_combined_data.csv')

# Save the formatted data to a CSV for further use
formatted_data.to_csv(formatted_data_file, index=True)

# Display the first few rows of the formatted data
print(formatted_data.head())
