import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'BillboardExplain/data/music_dataset_with_rank_category.csv'
data = pd.read_csv(file_path)

# Encode Rank Category as an ordinal variable
data['Rank Category Ordinal'] = data['Rank Category'].map({
    'Top 20%': 1,
    'Next 20%': 2,
    'Middle 20%': 3,
    'Lower 20%': 4,
    'Bottom 20%': 5
})

# Normalize or standardize relevant musical features (Danceability, Energy, Loudness, Tempo, etc.)
features_to_normalize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
scaler = StandardScaler()

# Apply standardization to the selected features
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Save the processed dataset to a new CSV file
output_file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data.to_csv(output_file_path, index=False)
