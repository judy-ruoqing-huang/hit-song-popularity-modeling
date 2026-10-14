# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'BillboardExplain/data/processed_music_dataset.csv'
data = pd.read_csv(file_path)

# Define features (X) and target (y)
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
X = data[columns_to_visualize]  # The musical features
y = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Align X and y (remove corresponding rows from X where y is NaN)
X = X.loc[y.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = gb_model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Model Accuracy: {accuracy}")

# Feature importance
feature_importances = pd.Series(gb_model.feature_importances_, index=columns_to_visualize)
print("Feature Importances:\n", feature_importances)

# Output results to a file
output_file_path = 'BillboardExplain/data/gradient_boosting_results.txt'
with open(output_file_path, 'w') as f:
    f.write(f"Gradient Boosting Model Accuracy: {accuracy}\n\n")
    f.write("Feature Importances:\n")
    f.write(feature_importances.to_string())

print(f"Results saved to: {output_file_path}")
