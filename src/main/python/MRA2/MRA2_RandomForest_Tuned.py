# Import necessary libraries
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Define the parameter grid for Randomized Search
param_grid = {
    'n_estimators': [100, 200, 500],               # Number of trees
    'max_depth': [None, 10, 20, 30],               # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],               # Minimum number of samples to split a node
    'min_samples_leaf': [1, 2, 4],                 # Minimum number of samples to form a leaf node
    'bootstrap': [True, False]                     # Whether bootstrap samples are used
}

# Initialize the RandomForest model
rf_model = RandomForestClassifier(random_state=42)

# RandomizedSearchCV: Random search of hyperparameters with cross-validation
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, 
                                   n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best model after hyperparameter tuning
best_rf_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy after Tuning: {accuracy}")

# Feature importance
feature_importances = pd.Series(best_rf_model.feature_importances_, index=columns_to_visualize)
print("Feature Importances after Tuning:\n", feature_importances)

# Output results to a file
output_file_path = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')), 'data', 'random_forest_tuned_results.txt')
with open(output_file_path, 'w') as f:
    f.write(f"Random Forest Model Accuracy after Tuning: {accuracy}\n\n")
    f.write("Feature Importances:\n")
    f.write(feature_importances.to_string())

print(f"Results saved to: {output_file_path}")
