import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pandas.plotting import scatter_matrix

# Set paths
data_file_path = 'billboard_spotify_analysis/data'
results_file_path = 'billboard_spotify_analysis/results'
correlation_file = os.path.join(results_file_path, 'correlation_results.csv')

# Load the correlation results
correlation_df = pd.read_csv(correlation_file)

# Rename the columns for simplicity
correlation_df.rename(columns={'Pearson Correlation (r)': 'Pearson', 'Spearman’s Rank Correlation Coefficient (ρ)': 'Spearman'}, inplace=True)

# Prepare the data for the heatmap
heatmap_data = correlation_df[['Pearson', 'Spearman']].set_index(correlation_df['Feature'])

# --- Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, fmt='.4g')
plt.title('Heatmap of Correlations Between Audio Features and Spotify Popularity')
plt.xticks(rotation=45)

# Save the heatmap as a file
heatmap_output_file = os.path.join(results_file_path, 'correlation_heatmap.png')
plt.tight_layout()
plt.savefig(heatmap_output_file)
plt.show()
print(f"Correlation Heatmap saved to {heatmap_output_file}")

# --- Bar Chart ---
plt.figure(figsize=(12, 6))

# Create the bar chart using a colorblind-friendly color scheme (Blue and Orange)
correlation_df.plot(x='Feature', y=['Pearson', 'Spearman'], 
                    kind='bar', figsize=(12, 6), color=['#0072B2', '#D55E00'])  # Blue and Orange
plt.title('Bar Chart of Correlation Between Audio Features and Spotify Popularity')
plt.xticks(rotation=90)
plt.ylabel('Correlation')
plt.tight_layout()

# Save the bar chart as a file
bar_chart_output_file = os.path.join(results_file_path, 'correlation_bar_chart.png')
plt.savefig(bar_chart_output_file)
plt.show()
print(f"Correlation Bar chart saved to {bar_chart_output_file}")

# --- Scatter Plot Matrix ---
# Load the original dataset (processed version)
processed_data_file = os.path.join(data_file_path, 'processed', 'processed_billboard_spotify_data.csv')
data = pd.read_csv(processed_data_file)

# Select a subset of features for scatter plot matrix (e.g., key audio features and Spotify popularity)
selected_features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo', 'Popularity']
scatter_matrix_data = data[selected_features]

# Create the scatter plot matrix
scatter_matrix_output_file = os.path.join(results_file_path, 'correlation_scatter_plot_matrix.png')
plt.figure(figsize=(12, 12))
scatter_matrix(scatter_matrix_data, alpha=0.2, figsize=(12, 12), diagonal='kde', marker='o', hist_kwds={'bins': 15}, s=40)
plt.suptitle('Correlation Scatter Plot Matrix of Audio Features and Spotify Popularity')

# Save the scatter plot matrix as a file
plt.savefig(scatter_matrix_output_file)
plt.show()
print(f"Correlation Scatter plot matrix saved to {scatter_matrix_output_file}")
