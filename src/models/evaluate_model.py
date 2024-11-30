import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Define file paths
model_path = './models/pipeline/pipeline_model.joblib'
test_data_path = './data/interim/test_with_predictions.csv'
metrics_output_dir = './models/metrics'
metrics_output_path = f"{metrics_output_dir}/metrics.json"

# Load the model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'")
pipeline = joblib.load(model_path)
print(f"Model loaded from '{model_path}'")

# Load the test dataset
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found at '{test_data_path}'")
test_data = pd.read_csv(test_data_path)
print(f"Test data loaded from '{test_data_path}'")

# Define the target column
target_col = 'Depression'  # Replace with actual target column name
if target_col not in test_data.columns:
    raise ValueError(f"Target column '{target_col}' not found in test data")

# Separate features and target
X_test = test_data.drop(columns=[target_col, 'Predicted_Depression'], errors='ignore')
y_test = test_data[target_col]

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1_score': f1_score(y_test, y_pred, average='weighted')
}

# Print metrics
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Save metrics to JSON
os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)  # Ensure the directory exists
with open(metrics_output_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to '{metrics_output_path}'")
