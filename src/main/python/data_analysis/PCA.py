import pandas as pd
from sklearn.decomposition import PCA
import os

# Function to format numbers consistently (e.g., 0.0001 or scientific notation for small numbers)
def format_number(num):
    if abs(num) < 0.001:
        return f"{num:.2e}"  # Scientific notation for very small numbers
    else:
        return round(num, 4)  # Round to four decimal places for other numbers

# Set the base data path for both raw and processed data
data_file_path = 'billboard_spotify_analysis/data'
results_file_path = 'billboard_spotify_analysis/results'

# Define processed data directory based on data_file_path
processed_data_file = os.path.join(data_file_path, 'processed', 'processed_billboard_spotify_data.csv')

# Load the processed dataset
data = pd.read_csv(processed_data_file)

# Select audio features including the one-hot encoded 'Key' and 'Mode' columns, excluding 'Key_0' and 'Mode_0'
audio_feature_columns = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 
                         'Speechiness', 'Valence', 'Loudness', 'Tempo', 
                         'Time Signature']
key_mode_columns = [col for col in data.columns if (col.startswith('Key_') or col.startswith('Mode_')) and col not in ['Key_0', 'Mode_0']]
audio_features = data[audio_feature_columns + key_mode_columns]

# Perform PCA without additional scaling
pca = PCA()
pca_components = pca.fit_transform(audio_features)

# Format PCA components
pca_components_formatted = [[format_number(value) for value in row] for row in pca_components]

# Create a DataFrame for PCA results and add original audio features
pca_df = pd.DataFrame(pca_components_formatted, columns=[f'PC{i+1}' for i in range(pca_components.shape[1])])
pca_df[audio_feature_columns + key_mode_columns] = data[audio_feature_columns + key_mode_columns]
pca_df['Popularity'] = data['Popularity'].values

# Save PCA results to a single file
pca_output_file = os.path.join(results_file_path, 'pca_results.csv')
pca_df.to_csv(pca_output_file, index=False)

# Explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
explained_variance_formatted = [format_number(var) for var in explained_variance]

# Principal component loadings (feature contributions to each component)
loadings = pca.components_.T
loading_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
loading_df.insert(0, 'Feature', audio_features.columns)  # Add the original feature names

# Apply formatting to each loading value
loading_df = loading_df.applymap(lambda x: format_number(x) if isinstance(x, (int, float)) else x)

# Save the loadings to match original feature names with each principal component
loadings_output_file = os.path.join(results_file_path, 'pca_feature_loadings.csv')
loading_df.to_csv(loadings_output_file, index=False)

# Save the explained variance with Feature labels
explained_variance_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(explained_variance_formatted))],
    'Explained Variance Ratio': explained_variance_formatted
})
variance_output_file = os.path.join(results_file_path, 'pca_explained_variance_ratio.csv')
explained_variance_df.to_csv(variance_output_file, index=False)

print("Explained Variance by Component saved to:", variance_output_file)
print("PCA results saved to:", pca_output_file)
print("Feature loadings saved to:", loadings_output_file)
