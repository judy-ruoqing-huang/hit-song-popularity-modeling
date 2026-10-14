import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set the results path for loading PCA results
results_file_path = 'billboard_spotify_analysis/results'

# Load explained variance ratios for the scree plot
explained_variance_file = os.path.join(results_file_path, 'pca_explained_variance_ratio.csv')
explained_variance_df = pd.read_csv(explained_variance_file)
explained_variance = explained_variance_df['Explained Variance Ratio'].values

# Set titles of the figures
scree_plot_title = 'PCA Scree Plot of Explained Variance of Principal Components for Audio Features'
biplot_title = 'PCA Biplot of Audio Features Contributions to Principal Components'

# Load PCA component scores for the biplot
pca_results_file = os.path.join(results_file_path, 'pca_results.csv')
pca_df = pd.read_csv(pca_results_file)

# Load feature loadings for biplot arrows
loadings_file = os.path.join(results_file_path, 'pca_feature_loadings.csv')
loadings_df = pd.read_csv(loadings_file)

# Calculate cumulative variance for the scree plot
cumulative_variance = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]

# Set up color palette similar to the heatmap's color scheme
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Scree Plot
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-', color=cmap(0.2), label="Explained Variance")
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color=cmap(0.8), label="Cumulative Variance")

# Adding annotations for each point
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    plt.text(i + 1, ev + 0.01, f"{ev:.2f}", ha='center', va='bottom', fontsize=8, color=cmap(0.2))
    plt.text(i + 1, cv + 0.01, f"{cv:.2f}", ha='center', va='bottom', fontsize=8, color=cmap(0.8))

# Customize appearance
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title(scree_plot_title)
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save Scree Plot
scree_plot_file = os.path.join(results_file_path, 'pca_explained_variance_scree_plot.png')
plt.savefig(scree_plot_file, dpi=300)
plt.close()
print(f"Scree Plot saved to {scree_plot_file}")

# Biplot function with enhanced offsets for central and Key/Mode features
def biplot(score, coeff, labels=None, explained_variance=[0, 0]):
    xs, ys = score[:, 0], score[:, 1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.figure(figsize=(10, 7))
    
    # Scatter plot of the PCA scores with reduced opacity and border
    plt.scatter(xs * scalex, ys * scaley, alpha=0.1, color=cmap(0.2), edgecolor='k', linewidth=0.5, s=20)

    # Plot arrows for each feature's loading with enhanced arrow styling
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color=cmap(0.8), alpha=0.9, lw=2.5, head_width=0.025)
        
        # Offset label positions for crowded features (central, Key, and Mode labels)
        if labels[i] in ["Liveness", "Tempo"] or "Key_" in labels[i] or "Mode_" in labels[i]:
            offset_x = coeff[i, 0] * 1.5  # Increased offset for clarity
            offset_y = coeff[i, 1] * 1.5
            plt.plot([coeff[i, 0], offset_x], [coeff[i, 1], offset_y], color='gray', linestyle='--', linewidth=0.7)
            plt.text(offset_x, offset_y, labels[i], color='green', ha='center', va='center', fontsize=9)
        else:
            # Bold the label if it's a key feature with strong loading
            fontweight = 'bold' if labels[i] in ["Energy", "Loudness", "Danceability"] else 'normal'
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='green', ha='center', va='center', fontsize=9, fontweight=fontweight)

    # Add labels and title with explained variance percentages
    plt.xlabel(f"PC1 ({explained_variance[0]:.1%} variance explained)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1%} variance explained)")
    plt.title(biplot_title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

# Prepare data for the biplot
pca_scores = pca_df.iloc[:, :-1].values  # Exclude 'Popularity' column
feature_loadings = loadings_df.iloc[:, 1:].values  # Exclude 'Feature' column
feature_names = loadings_df['Feature'].values

# Assuming explained variance ratios are available for PC1 and PC2
explained_variance_pc1_pc2 = [explained_variance[0], explained_variance[1]]  # Adjust if necessary

# Create the enhanced biplot
biplot(pca_scores[:, :2], feature_loadings[:, :2], labels=feature_names, explained_variance=explained_variance_pc1_pc2)

# Save the enhanced biplot
biplot_file = os.path.join(results_file_path, 'pca_audio_features_biplot.png')
plt.savefig(biplot_file, dpi=300)
plt.show()

print(f"Biplot saved to {biplot_file}")
