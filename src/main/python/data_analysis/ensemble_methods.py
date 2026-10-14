import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# Select features and target variable (Retain Key_0 and Mode_0)
key_columns = [col for col in data.columns if col.startswith('Key')]
mode_columns = [col for col in data.columns if col.startswith('Mode')]
features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo', 'Time Signature'] + key_columns + mode_columns  # Retaining all Key and Mode columns
target = 'Popularity'

X = data[features]
y = data[target]

# Split the data into training and test sets (random_state ensures consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest Regressor ---
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# --- Gradient Boosting Regressor ---
gradient_boosting = GradientBoostingRegressor(random_state=42)
gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)

# Round 'y_test', 'y_pred_rf', and 'y_pred_gb' to integer values (between 0 and 100)
y_test_rounded = y_test.round().astype(int).clip(0, 100).reset_index(drop=True)
y_pred_rf_rounded = pd.Series(y_pred_rf.round().astype(int).clip(0, 100), index=X_test.index).reset_index(drop=True)
y_pred_gb_rounded = pd.Series(y_pred_gb.round().astype(int).clip(0, 100), index=X_test.index).reset_index(drop=True)

# Ensure indices align for all predictions and actual values
predictions_df = pd.DataFrame({
    'Actual Popularity': y_test_rounded,
    'Random Forest Predicted Popularity': y_pred_rf_rounded,
    'Gradient Boosting Predicted Popularity': y_pred_gb_rounded
})

# Save Actual Popularity and Predicted Popularity from both models in a single file
predictions_file = os.path.join(results_file_path, 'ensemble_actual_vs_predicted_popularity.csv')
predictions_df.to_csv(predictions_file, index=False)

# Output performance metrics
r2_rf = format_number(r2_score(y_test, y_pred_rf))
mse_rf = format_number(mean_squared_error(y_test, y_pred_rf))
r2_gb = format_number(r2_score(y_test, y_pred_gb))
mse_gb = format_number(mean_squared_error(y_test, y_pred_gb))

performance = {
    'Model': ['Random Forest', 'Gradient Boosting'],
    'R² Score': [r2_rf, r2_gb],
    'MSE': [mse_rf, mse_gb]
}

# Save performance to 'ensemble_performance.csv'
performance_df = pd.DataFrame(performance)
performance_file = os.path.join(results_file_path, 'ensemble_performance.csv')
performance_df.to_csv(performance_file, index=False)

# Save feature importances for both models
rf_feature_importances = [format_number(val) for val in random_forest.feature_importances_]
gb_feature_importances = [format_number(val) for val in gradient_boosting.feature_importances_]

feature_importances_df = pd.DataFrame({
    'Feature': features,
    'Random Forest Importance': rf_feature_importances,
    'Gradient Boosting Importance': gb_feature_importances
})

# Add explanations for Key in the format 'Key (X = Pitch)'
for key_col in key_explanation.keys():
    if key_col in feature_importances_df['Feature'].values:
        key_num = key_col.split('_')[1]
        feature_importances_df.loc[feature_importances_df['Feature'] == key_col, 'Feature'] = f"Key ({key_num} = {key_explanation[key_col]})"

# Add explanations for Mode in the format 'Mode (0 = Minor)' or 'Mode (1 = Major)'
for mode_col in mode_explanation.keys():
    if mode_col in feature_importances_df['Feature'].values:
        mode_num = mode_col.split('_')[1]
        feature_importances_df.loc[feature_importances_df['Feature'] == mode_col, 'Feature'] = f"Mode ({mode_num} = {mode_explanation[mode_col]})"

# Save the feature importances
feature_importances_file = os.path.join(results_file_path, 'ensemble_feature_importances.csv')
feature_importances_df.to_csv(feature_importances_file, index=False)

print(f"Ensemble methods performance saved to {performance_file}")
print(f"Feature importances saved to {feature_importances_file}")
print(f"Predictions saved to {predictions_file}")
