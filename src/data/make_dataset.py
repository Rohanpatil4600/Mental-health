import pandas as pd
from sklearn.model_selection import train_test_split
import os
df = pd.read_csv("/Users/rohan/depression-prediction/data/raw/final_depression_dataset_1.csv")

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

raw_data_path = os.path.join("data", "raw")
os.makedirs(raw_data_path, exist_ok=True)  # Creates 'data/raw' if it doesn't exist

processed_data_path = os.path.join("data", "processed")
os.makedirs(processed_data_path, exist_ok=True)  # Creates 'data/processed' if it doesn't exist

# Save train and test data to the 'processed' directory
train_data.to_csv(os.path.join(processed_data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(processed_data_path, "test.csv"), index=False)
