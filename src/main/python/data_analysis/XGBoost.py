import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set paths
data_file_path = 'billboard_spotify_analysis/data/processed'
results_file_path = 'billboard_spotify_analysis/results'
processed_data_file = os.path.join(data_file_path, 'processed_billboard_spotify_data.csv')

# Load the processed dataset
data = pd.read_csv(processed_data_file)

# Dynamically find all columns related to 'Key' and 'Mode' except for Key_0 and Mode_0
key_columns = [col for col in data.columns if col.startswith('Key') and col != 'Key_0']
mode_columns = [col for col in data.columns if col.startswith('Mode') and col != 'Mode_0']

# Select features including Key and Mode (dropping Key_0 and Mode_0)
features = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 
            'Valence', 'Loudness', 'Tempo', 'Time Signature'] + key_columns + mode_columns
target = 'Popularity'

X = data[features]
y = data[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_xgb_model = grid_search.best_estimator_

# Predict on the test set
y_pred_xgb = best_xgb_model.predict(X_test)

# Evaluate the model performance
r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# Output the performance metrics
r2_xgb_rounded = round(r2_xgb, 4)
mse_xgb_rounded = round(mse_xgb, 4)
print(f"XGBoost - R² Score: {r2_xgb_rounded}, MSE: {mse_xgb_rounded}")

# Save the predictions and performance to CSV
y_test_df = pd.DataFrame({'Actual Popularity': y_test, 'Predicted Popularity': y_pred_xgb})
y_test_output_file = os.path.join(results_file_path, 'xgb_y_test_predictions.csv')
y_test_df.to_csv(y_test_output_file, index=False)

# Save model performance to CSV
performance = {'Model': ['XGBoost'], 'R² Score': [r2_xgb_rounded], 'MSE': [mse_xgb_rounded]}
performance_df = pd.DataFrame(performance)
performance_file = os.path.join(results_file_path, 'xgb_performance.csv')
performance_df.to_csv(performance_file, index=False)

print(f"XGBoost predictions saved to {y_test_output_file}")
print(f"XGBoost performance saved to {performance_file}")
