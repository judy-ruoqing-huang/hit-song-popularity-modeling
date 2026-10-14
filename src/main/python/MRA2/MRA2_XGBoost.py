# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
X = data[columns_to_visualize]  # The musical features
y = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Align X and y (remove corresponding rows from X where y is NaN)
X = X.loc[y.index]

# Adjust the labels to start from 0 (XGBoost expects labels starting from 0)
y_adjusted = y - 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted, test_size=0.3, random_state=42)

# XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Fit the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy}")

# Feature importance
feature_importances = pd.Series(xgb_model.feature_importances_, index=columns_to_visualize)
print("Feature Importances:\n", feature_importances)

# Output results to a file
output_file_path = 'BillboardExplain/data/xgboost_results.txt'
with open(output_file_path, 'w') as f:
    f.write(f"XGBoost Model Accuracy: {accuracy}\n\n")
    f.write("Feature Importances:\n")
    f.write(feature_importances.to_string())

print(f"Results saved to: {output_file_path}")
