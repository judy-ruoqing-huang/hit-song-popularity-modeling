import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
results_file_path = 'billboard_spotify_analysis/results'
coefficients_file = os.path.join(results_file_path, 'regression_results.csv')

# Load the regression coefficients data
coefficients_df = pd.read_csv(coefficients_file)
chart_title = 'Regression Coefficients by Model and Audio Feature'

# Filter out low-impact features based on an importance threshold
importance_threshold = 0.5  # Adjust threshold as needed to display impactful features only
filtered_coefficients_df = coefficients_df[
    coefficients_df[['Lasso Coefficients', 'Elastic Net Coefficients', 'Linear Regression Coefficients']].abs().max(axis=1) > importance_threshold
]

# Melt the DataFrame for easy plotting with Seaborn
coefficients_melted = filtered_coefficients_df.melt(id_vars='Feature', 
                                                    var_name='Model', 
                                                    value_name='Coefficient')

# Plot settings
plt.figure(figsize=(14, 8))
sns.barplot(x='Feature', y='Coefficient', hue='Model', data=coefficients_melted, 
            palette=['#0072B2', '#D55E00', '#009E73'])  # Distinct colorblind-friendly colors

# Update chart aesthetics
plt.title(chart_title, fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust font size and rotation for readability
plt.xlabel('Feature', fontsize=14, weight='bold')
plt.ylabel('Coefficient', fontsize=14, weight='bold')
plt.legend(title='Model', fontsize=12, title_fontsize='13')

plt.tight_layout()

# Save the updated bar chart
bar_chart_output_file = os.path.join(results_file_path, 'regression_coefficients_bar_chart.png')
plt.savefig(bar_chart_output_file)
plt.show()
print(f"Filtered regression coefficients bar chart saved to {bar_chart_output_file}")
