import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
results_file_path = 'billboard_spotify_analysis/results'
ensemble_importance_file = os.path.join(results_file_path, 'ensemble_feature_importances.csv')
ensemble_df = pd.read_csv(ensemble_importance_file)

# Titles
bar_chart_title = 'Feature Importances by Model and Audio Feature'
scatter_rf_title = 'Random Forest: Actual vs Predicted Popularity'
scatter_gb_title = 'Gradient Boosting: Actual vs Predicted Popularity'

# --- Bar Chart of Feature Importances ---

# Create a bar chart with enhanced readability
plt.figure(figsize=(14, 8))  # Increase figure size for readability
ax = ensemble_df.plot(
    x='Feature', 
    y=['Random Forest Importance', 'Gradient Boosting Importance'],
    kind='bar', 
    figsize=(14, 8), 
    color=['#0072B2', '#D55E00'],  # Blue and Orange for colorblind-friendly scheme
    width=0.75
)

# Update chart aesthetics
plt.title(bar_chart_title, fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels and increase font size
plt.yticks(fontsize=12)
plt.xlabel('Feature', fontsize=14, weight='bold')
plt.ylabel('Importance', fontsize=14, weight='bold')

plt.legend(title='Model', fontsize=12, title_fontsize='13')
plt.tight_layout()

# Save the improved chart
bar_chart_output_file = os.path.join(results_file_path, 'ensemble_feature_importance_bar_chart.png')
plt.savefig(bar_chart_output_file)
plt.show()
print(f"Improved feature importance bar chart saved to {bar_chart_output_file}")

# --- Scatter Plot for Random Forest ---

# Load actual vs predicted values for Random Forest and Gradient Boosting
y_test_file = os.path.join(results_file_path, 'ensemble_actual_vs_predicted_popularity.csv')
y_test = pd.read_csv(y_test_file)

plt.figure(figsize=(8, 6))
plt.scatter(y_test['Actual Popularity'], y_test['Random Forest Predicted Popularity'], alpha=0.5, color='#0072B2')
plt.title(scatter_rf_title, fontsize=16, weight='bold')
plt.xlabel('Actual Popularity', fontsize=14, weight='bold')
plt.ylabel('Predicted Popularity', fontsize=14, weight='bold')
plt.plot([min(y_test['Actual Popularity']), max(y_test['Actual Popularity'])],
         [min(y_test['Actual Popularity']), max(y_test['Actual Popularity'])], color='red')
plt.tight_layout()

# Save the scatter plot
scatter_rf_output_file = os.path.join(results_file_path, 'rf_actual_vs_predicted_scatter.png')
plt.savefig(scatter_rf_output_file)
plt.show()
print(f"Random Forest scatter plot saved to {scatter_rf_output_file}")

# --- Scatter Plot for Gradient Boosting ---

plt.figure(figsize=(8, 6))
plt.scatter(y_test['Actual Popularity'], y_test['Gradient Boosting Predicted Popularity'], alpha=0.5, color='#D55E00')
plt.title(scatter_gb_title, fontsize=16, weight='bold')
plt.xlabel('Actual Popularity', fontsize=14, weight='bold')
plt.ylabel('Predicted Popularity', fontsize=14, weight='bold')
plt.plot([min(y_test['Actual Popularity']), max(y_test['Actual Popularity'])],
         [min(y_test['Actual Popularity']), max(y_test['Actual Popularity'])], color='red')
plt.tight_layout()

# Save the scatter plot
scatter_gb_output_file = os.path.join(results_file_path, 'gb_actual_vs_predicted_scatter.png')
plt.savefig(scatter_gb_output_file)
plt.show()
print(f"Gradient Boosting scatter plot saved to {scatter_gb_output_file}")
