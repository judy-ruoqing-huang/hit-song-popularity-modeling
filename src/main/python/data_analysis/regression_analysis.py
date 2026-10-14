import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Function to format numbers consistently (e.g., 0.0001 or scientific notation for small numbers)
def format_number(num):
    if abs(num) < 0.001:
        return f"{num:.2e}"  # Scientific notation for very small numbers
    else:
        return round(num, 4)  # Round to four decimal places for other numbers

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

# Set paths
data_file_path = 'billboard_spotify_analysis/data/processed'
results_file_path = 'billboard_spotify_analysis/results'
processed_data_file = os.path.join(data_file_path, 'processed_billboard_spotify_data.csv')

# Load the processed dataset
data = pd.read_csv(processed_data_file)

# Dynamically find all columns related to 'Key' and 'Mode' (in case they were one-hot encoded)
key_columns = [col for col in data.columns if col.startswith('Key')]
mode_columns = [col for col in data.columns if col.startswith('Mode')]

# Select features and target variable
# We drop Key (0 = C) and Mode (0 = Minor) to avoid multicollinearity
features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo', 'Time Signature'] + key_columns[1:] + mode_columns[1:]  # Drop first 'Key' and 'Mode'
target = 'Popularity'

X = data[features]
y = data[target]

# Split the data into training and test sets (random_state ensures consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)

# --- Lasso Regression ---
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# --- Elastic Net Regression ---
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train, y_train)
y_pred_elastic = elastic_net.predict(X_test)

# Round 'y_test', 'y_pred_linear', 'y_pred_lasso', and 'y_pred_elastic' to integer values (between 0 and 100)
y_test_rounded = y_test.round().astype(int).clip(0, 100)
y_pred_linear_rounded = y_pred_linear.round().astype(int).clip(0, 100)
y_pred_lasso_rounded = y_pred_lasso.round().astype(int).clip(0, 100)
y_pred_elastic_rounded = y_pred_elastic.round().astype(int).clip(0, 100)

# Combine actual and predicted values from all three models into one DataFrame
regression_predictions_df = pd.DataFrame({
    'Actual Popularity': y_test_rounded,
    'Linear Regression Predicted Popularity': y_pred_linear_rounded,
    'Lasso Predicted Popularity': y_pred_lasso_rounded,
    'Elastic Net Predicted Popularity': y_pred_elastic_rounded
})

# Save combined actual vs predicted results to 'regression_actual_vs_predicted_popularity.csv'
regression_predictions_file = os.path.join(results_file_path, 'regression_actual_vs_predicted_popularity.csv')
regression_predictions_df.to_csv(regression_predictions_file, index=False)

# Output performance metrics
r2_linear = format_number(r2_score(y_test, y_pred_linear))
mse_linear = format_number(mean_squared_error(y_test, y_pred_linear))

r2_lasso = format_number(r2_score(y_test, y_pred_lasso))
mse_lasso = format_number(mean_squared_error(y_test, y_pred_lasso))

r2_elastic = format_number(r2_score(y_test, y_pred_elastic))
mse_elastic = format_number(mean_squared_error(y_test, y_pred_elastic))

# Apply formatting to coefficients as well
lasso_coef = [format_number(coef) for coef in lasso.coef_]
elastic_net_coef = [format_number(coef) for coef in elastic_net.coef_]
linear_reg_coef = [format_number(coef) for coef in linear_reg.coef_]

# Create coefficient DataFrame
coefficients_df = pd.DataFrame({
    'Feature': features,
    'Lasso Coefficients': lasso_coef,
    'Elastic Net Coefficients': elastic_net_coef,
    'Linear Regression Coefficients': linear_reg_coef
})

# Add explanations for Key in the format 'Key (X = Pitch)'
for key_col in key_explanation.keys():
    if key_col in coefficients_df['Feature'].values:
        key_num = key_col.split('_')[1]
        coefficients_df.loc[coefficients_df['Feature'] == key_col, 'Feature'] = f"Key ({key_num} = {key_explanation[key_col]})"

# Add explanations for Mode in the format 'Mode (0 = Minor)' or 'Mode (1 = Major)'
for mode_col in mode_explanation.keys():
    if mode_col in coefficients_df['Feature'].values:
        mode_num = mode_col.split('_')[1]
        coefficients_df.loc[coefficients_df['Feature'] == mode_col, 'Feature'] = f"Mode ({mode_num} = {mode_explanation[mode_col]})"

# Save the coefficients with Key and Mode explanations
coefficients_file = os.path.join(results_file_path, 'regression_results.csv')
coefficients_df.to_csv(coefficients_file, index=False)

# Output the performance metrics for each model
performance = {
    'Model': ['Lasso', 'Elastic Net', 'Linear Regression'],
    'R² Score': [r2_lasso, r2_elastic, r2_linear],
    'MSE': [mse_lasso, mse_elastic, mse_linear]
}

# Save performance to 'regression_performance.csv'
performance_df = pd.DataFrame(performance)
performance_file = os.path.join(results_file_path, 'regression_performance.csv')
performance_df.to_csv(performance_file, index=False)

print(f"Regression performance saved to {performance_file}")
print(f"Regression coefficients saved to {coefficients_file}")
print(f"Combined actual vs predicted results saved to {regression_predictions_file}")
