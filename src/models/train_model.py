import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import yaml

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)


# Load the preprocessed train and test datasets
train = pd.read_csv('./data/interim/train_processed.csv')
test = pd.read_csv('./data/interim/test_processed.csv')

# Define the target variable
target_col = 'Depression'  # Replace with the actual target column name if different

# Define the categorical and numerical columns
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Ensure the target column is not part of the feature columns
if target_col in categorical_cols:
    categorical_cols.remove(target_col)
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Split the features and target variable
X = train[categorical_cols + numerical_cols]
y = train[target_col]

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params['train_model']['test_size'], random_state=42)

# Define transformers for numerical and categorical columns
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Define ColumnTransformer with transformers for each type of column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Define the pipeline with preprocessing and RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on validation data
y_pred = pipeline.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the model to the models directory
model_path = './models/pipeline/pipeline_model.joblib'
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
joblib.dump(pipeline, model_path)
print(f"Model saved to '{model_path}'")

# Prepare the test dataset and make predictions
X_test = test[categorical_cols + numerical_cols]
test_predictions = pipeline.predict(X_test)

# Save test predictions to a CSV file
test['Predicted_Depression'] = test_predictions  # Replace 'Predicted_Depression' if another column name is required
test.to_csv('./data/interim/test_with_predictions.csv', index=False)
print("Test predictions saved to './data/interim/test_with_predictions.csv'")
