import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set the base data path for both raw and processed data
data_file_path = 'billboard_spotify_analysis/data'

# Define raw and processed data directories based on data_file_path
raw_data_file = os.path.join(data_file_path, 'raw', 'billboard_spotify_raw_data.csv')
processed_data_file = os.path.join(data_file_path, 'processed', 'processed_billboard_spotify_data.csv')

# Load the raw dataset
data = pd.read_csv(raw_data_file)

# Step 1: Handling Missing Values - Filling missing values with mean (or another strategy)
# Could consider using more advanced techniques (such as interpolation or KNN imputation) if the mean is not appropriate for all columns.
data.fillna(data.mean(), inplace=True)

# Step 2: Normalize the numerical features, including 'Time Signature'
numerical_features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 
                      'Speechiness', 'Valence', 'Loudness', 'Tempo', 'Time Signature']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Step 3: Convert 'Key' to a categorical data type
data['Key'] = data['Key'].astype(int)  # Ensure 'Key' is integer
data['Key'] = data['Key'].replace(-1, 'No Key Detected')  # Handle -1 as a category for missing key
data['Key'] = data['Key'].astype('category')  # Convert to categorical

# Step 4: Convert 'Mode' to a categorical data type
data['Mode'] = data['Mode'].astype(int)  # Ensure 'Mode' is integer (0 for Minor, 1 for Major)
data['Mode'] = data['Mode'].astype('category')  # Convert to categorical

# Step 5: Apply one-hot encoding to 'Key' and 'Mode'
# For tree-based ensemble methods (e.g., Random Forest, Gradient Boosting): Use drop_first=False to keep all categories.
# For regression analysis: Use drop_first=True to prevent multicollinearity
data = pd.get_dummies(data, columns=['Key', 'Mode'], drop_first=False)

# Save the processed dataset
data.to_csv(processed_data_file, index=False)

# Display the first few rows of the processed data
print(data.head())
