# Import necessary libraries
from scipy.stats import f_oneway
import pandas as pd

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Define the musical features and rank category
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
rank_category = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Initialize dictionary to store ANOVA results
anova_results = {}

# Conduct one-way ANOVA for each feature
for feature in columns_to_visualize:
    # Perform ANOVA by grouping the feature by Rank Category Ordinal
    groups = [data[feature][rank_category == rank].dropna() for rank in rank_category.unique()]
    f_stat, p_value = f_oneway(*groups)
    
    # Store the results
    anova_results[feature] = {'F-statistic': f_stat, 'p-value': p_value}

# Convert the results to a pandas DataFrame for easy viewing
anova_results_df = pd.DataFrame(anova_results).T

# Output the ANOVA results
print("ANOVA Results (F-statistic and p-value) between Musical Features and Rank Category:\n")
print(anova_results_df)

# Output results to a file
output_file_path = 'BillboardExplain/data/anova_results.txt'
anova_results_df.to_string(output_file_path)

print(f"Results saved to: {output_file_path}")
