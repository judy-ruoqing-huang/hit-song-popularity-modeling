import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Function to format p-values according to best practices
def format_p_value(p_value):
    if p_value < 0.001:
        return f"{p_value:.2e}"  # Scientific notation for small p-values (e.g., 1.77E-16)
    else:
        return round(p_value, 3)  # Round to three decimal places for larger p-values

# Explanation of Key values based on Pitch Class notation
key_explanation = {
    'Key_0': 'C', 'Key_1': 'C♯/D♭', 'Key_2': 'D', 'Key_3': 'D♯/E♭', 'Key_4': 'E',
    'Key_5': 'F', 'Key_6': 'F♯/G♭', 'Key_7': 'G', 'Key_8': 'G♯/A♭', 'Key_9': 'A',
    'Key_10': 'A♯/B♭', 'Key_11': 'B'
}

# Explanation of Mode values
mode_explanation = {
    'Mode_0': 'Minor', 'Mode_1': 'Major'
}

# Set the base data path for both raw and processed data
data_file_path = 'billboard_spotify_analysis/data'

# Define raw and processed data directories based on data_file_path
raw_data_file = os.path.join(data_file_path, 'raw', 'billboard_spotify_raw_data.csv')
processed_data_file = os.path.join(data_file_path, 'processed', 'processed_billboard_spotify_data.csv')

# Load the raw dataset
# data = pd.read_csv(raw_data_file)
data = pd.read_csv(processed_data_file)

# Step 1: Handling Missing Values - Filling missing values with mean (or another strategy)
# data.fillna(data.mean(), inplace=True)

# Step 2: Normalize the numerical features, including 'Time Signature'
# numerical_features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 
#                       'Speechiness', 'Valence', 'Loudness', 'Tempo', 'Time Signature']
# scaler = StandardScaler()
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Step 3: Convert 'Key' and 'Mode' to categorical data type for one-hot encoding
# data['Key'] = data['Key'].astype(int).astype('category')
# data['Mode'] = data['Mode'].astype(int).astype('category')

# Step 4: Apply one-hot encoding to 'Key' and 'Mode'
# data = pd.get_dummies(data, columns=['Key', 'Mode'], drop_first=False)

# Step 5: Perform correlation analysis (Pearson and Spearman) and apply p-value formatting
features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 
            'Loudness', 'Tempo', 'Time Signature'] + \
           [col for col in data.columns if col.startswith('Key_')] + \
           [col for col in data.columns if col.startswith('Mode_')]

target = 'Popularity'

# Initialize dictionaries to store correlation results
pearson_results = {}
spearman_results = {}

# Calculate Pearson and Spearman correlations for each feature
for feature in features:
    pearson_corr, pearson_pval = pearsonr(data[feature], data[target])
    spearman_corr, spearman_pval = spearmanr(data[feature], data[target])
    
    # Store the results with formatted p-values
    pearson_results[feature] = (round(pearson_corr, 4), format_p_value(pearson_pval))
    spearman_results[feature] = (round(spearman_corr, 4), format_p_value(spearman_pval))

# Create a DataFrame to display the results with explanations for Key and Mode
correlation_df = pd.DataFrame({
    'Feature': features,
    'Pearson Correlation (r)': [pearson_results[feat][0] for feat in features],
    'Pearson p-value': [pearson_results[feat][1] for feat in features],
    'Spearman’s Rank Correlation Coefficient (ρ)': [spearman_results[feat][0] for feat in features],
    'Spearman p-value': [spearman_results[feat][1] for feat in features]
})

# Add explanations for Key in the format 'Key (X = Pitch)'
for key_col in key_explanation.keys():
    if key_col in correlation_df['Feature'].values:
        # Extract the numeric value of the key from the column name and format as 'Key (X = Pitch)'
        key_num = key_col.split('_')[1]
        correlation_df.loc[correlation_df['Feature'] == key_col, 'Feature'] = f"Key ({key_num} = {key_explanation[key_col]})"

# Add explanations for Mode in the format 'Mode (0 = Minor)' or 'Mode (1 = Major)'
for mode_col in mode_explanation.keys():
    if mode_col in correlation_df['Feature'].values:
        mode_num = mode_col.split('_')[1]
        correlation_df.loc[correlation_df['Feature'] == mode_col, 'Feature'] = f"Mode ({mode_num} = {mode_explanation[mode_col]})"

# Define the results file path
results_file_path = 'billboard_spotify_analysis/results'
correlation_file = os.path.join(results_file_path, 'correlation_results.csv')

# Ensure the results directory exists
os.makedirs(results_file_path, exist_ok=True)

# Save the results to a CSV file
correlation_df.to_csv(correlation_file, index=False)

# Display the first few rows of the correlation results with key and mode explanations
print(correlation_df)
