# Import necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mord import LogisticAT  # For ordinal logistic regression

# Load the dataset
file_path = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')), 'data', 'processed_music_dataset.csv')
data = pd.read_csv(file_path)

# Define features (X) and target (y)
columns_to_visualize = ['Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Speechiness', 'Valence', 'Loudness', 'Tempo']
X = data[columns_to_visualize]  # The musical features
y = data['Rank Category Ordinal'].dropna()  # Drop NaN in the target variable

# Align X and y (remove corresponding rows from X where y is NaN)
X = X.loc[y.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ordinal Logistic Regression
ordinal_model = LogisticAT(alpha=1.0)  # alpha can be adjusted for regularization

# Fit the model
ordinal_model.fit(X_train, y_train)

# Predict on the test set
y_pred = ordinal_model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Feature coefficients
coefficients = pd.Series(ordinal_model.coef_, index=columns_to_visualize)

# Output results to a file
output_file_path = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')), 'data', 'ordinal_logistic_regression_results.txt')
with open(output_file_path, 'w') as f:
    f.write(f"Model Accuracy: {accuracy}\n\n")
    f.write("Feature Coefficients:\n")
    f.write(coefficients.to_string())

print(f"Results saved to: {output_file_path}")
