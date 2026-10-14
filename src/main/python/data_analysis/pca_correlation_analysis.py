import os
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

# Function to format p-values according to best practices
def format_p_value(p_value):
    if p_value < 0.001:
        return f"{p_value:.2e}"  # Scientific notation for small p-values (e.g., 1.77E-16)
    else:
        return round(p_value, 3)  # Round to three decimal places for larger p-values

# Set up paths
data_file_path = 'billboard_spotify_analysis/data'
results_file_path = 'billboard_spotify_analysis/results'
pca_result_file = os.path.join(results_file_path, 'pca_results.csv')

# Load the PCA data with PC1 and Popularity
data = pd.read_csv(pca_result_file)
X = data[['PC1']].values
y = data['Popularity'].values

# Initialize dictionaries to store results
pearson_results = {}
spearman_results = {}

# Correlation analysis for PC1
pearson_corr, pearson_pval = pearsonr(data['PC1'], data['Popularity'])
spearman_corr, spearman_pval = spearmanr(data['PC1'], data['Popularity'])

# Store the correlation results
pearson_results['PC1'] = (round(pearson_corr, 4), format_p_value(pearson_pval))
spearman_results['PC1'] = (round(spearman_corr, 4), format_p_value(spearman_pval))

# Regression models
regression_results = {}

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
regression_results['Linear Regression'] = linear_model.score(X, y)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)
regression_results['Lasso'] = lasso_model.score(X, y)

# Elastic Net Regression
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_model.fit(X, y)
regression_results['Elastic Net'] = elastic_net_model.score(X, y)

# Ensemble models
ensemble_results = {}

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf_model, X, y, scoring='r2', cv=5)
ensemble_results['Random Forest'] = rf_scores.mean()

# Gradient Boosting Regression
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_scores = cross_val_score(gb_model, X, y, scoring='r2', cv=5)
ensemble_results['Gradient Boosting'] = gb_scores.mean()

# Format the results in a DataFrame
results_df = pd.DataFrame({
    'Feature': ['PC1'],
    'Pearson Correlation (r)': [pearson_results['PC1'][0]],
    'Pearson p-value': [pearson_results['PC1'][1]],
    'Spearman Correlation (ρ)': [spearman_results['PC1'][0]],
    'Spearman p-value': [spearman_results['PC1'][1]],
    'Linear Regression R²': [regression_results['Linear Regression']],
    'Lasso R²': [regression_results['Lasso']],
    'Elastic Net R²': [regression_results['Elastic Net']],
    'Random Forest R²': [ensemble_results['Random Forest']],
    'Gradient Boosting R²': [ensemble_results['Gradient Boosting']]
})

# Save the results to a CSV file
results_file = os.path.join(results_file_path, 'pca_regression_ensemble_results.csv')
os.makedirs(results_file_path, exist_ok=True)
results_df.to_csv(results_file, index=False)

# Display the results
print(results_df)
