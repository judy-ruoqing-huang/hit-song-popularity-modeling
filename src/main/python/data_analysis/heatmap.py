import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the base file path for the results directory
results_file_path = 'billboard_spotify_analysis/results'

# Define the full path for the normalized combined data file
normalized_data_file = os.path.join(results_file_path, 'normalized_combined_data.csv')
plot_title = 'Cross-Method Heatmap of Audio Feature Influence on Popularity'

# Load the normalized combined data from the CSV file
normalized_data = pd.read_csv(normalized_data_file, index_col='Feature')

# 1. Abbreviate row labels for readability
normalized_data.index = [
    "Danceability", "Energy", "Instrum.", "Liveness", "Speechiness",
    "Valence", "Loudness", "Tempo", "Time Sig.",
    "Key 1 (C#/Db)", "Key 2 (D)", "Key 3 (D#/Eb)", "Key 4 (E)",
    "Key 5 (F)", "Key 6 (F#/Gb)", "Key 7 (G)", "Key 8 (G#/Ab)",
    "Key 9 (A)", "Key 10 (A#/Bb)", "Key 11 (B)", "Mode 1 (Major)"
]

# 2. Abbreviate column names for readability
normalized_data.columns = [
    "Pearson Corr.", "Spearman Corr.",  
    "Lasso Coeff.", "Elastic Net Coeff.", "Lin. Reg.",
    "RF Importance", "GB Importance"
]

# Set up the figure and customize the heatmap with optional enhancements
plt.figure(figsize=(12, 10))
sns.heatmap(
    normalized_data, 
    cmap="RdBu",                   # Diverging color map for clearer visual contrast
    annot=True,                    # Show values in each cell
    fmt=".2f",                     # Format for annotations (2 decimal places)
    annot_kws={"size": 7},         # Set annotation font size for clarity
    linewidths=1,                  # Thicker grid lines for better separation
    center=0,                      # Center color gradient at zero
    cbar_kws={"label": "Normalized Scale"}  # Label for the color bar
)

# Add labels and title
plt.title(plot_title, pad=20)
plt.xlabel("Analysis Methods", labelpad=10)
plt.ylabel("Audio Features", labelpad=10)

# Rotate x-axis labels for readability
plt.xticks(rotation=30, ha='right', fontsize=8)  # Rotate labels, adjust font size

# Save the heatmap to a file
heatmap_file = os.path.join(results_file_path, 'normalized_combined_data_heatmap.png')
plt.savefig(heatmap_file, bbox_inches='tight')  # Use tight layout to avoid cutoff
plt.show()  # Display the heatmap
