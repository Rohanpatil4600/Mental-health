import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os


import pandas as pd

def preprocess_data(df):
    """
    Applies a series of data transformations to clean and preprocess the dataset.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Define criteria
    cities_to_include = [
        'Kalyan', 'Patna', 'Vasai-Virar', 'Kolkata', 'Ahmedabad', 'Meerut', 'Ludhiana', 
        'Pune', 'Rajkot', 'Visakhapatnam', 'Srinagar', 'Mumbai', 'Indore', 'Agra', 
        'Surat', 'Varanasi', 'Vadodara', 'Hyderabad', 'Kanpur', 'Jaipur', 'Thane', 
        'Lucknow', 'Nagpur', 'Bangalore', 'Chennai', 'Ghaziabad', 'Delhi', 'Bhopal', 
        'Faridabad', 'Nashik'
    ]
    professions_to_drop = [
        'Academic', 'Profession', 'Yogesh', 'BCA', 'Unemployed', 'LLM', 'PhD', 'MBA', 
        'Dev', 'BE', 'Family Consultant', 'Visakhapatnam', 'Pranav', 'Analyst', 'M.Ed', 
        'Moderate', 'Nagpur', 'B.Ed', 'Unveil', 'Patna', 'MBBS', 'Working Professional', 
        'Medical Doctor', 'BBA', 'City Manager', 'FamilyVirar', 'B.Com', 'Yuvraj'
    ]
    sleep_hours_to_keep = [
        'Less than 5 hours', '7-8 hours', 'More than 8 hours', '5-6 hours'
    ]
    dietary_habits_to_keep = ['Moderate', 'Unhealthy', 'Healthy']
    degrees_to_keep = [
        "Class 12", "B.Ed", "B.Arch", "B.Com", "B.Pharm", "BCA", "M.Ed", "MCA", 
        "BBA", "BSc", "MSc", "LLM", "M.Pharm", "M.Tech", "B.Tech", "LLB", "BHM", 
        "MBA", "BA", "ME", "MD", "MHM", "PhD", "BE", "M.Com", "MBBS", "MA"
    ]
    columns_to_impute = [
        "Academic Pressure", "Work Pressure", "CGPA", 
        "Study Satisfaction", "Job Satisfaction"
    ]
    bins = [18, 29, 42, 51, 60]
    labels = ['18-29', '30-42', '43-51', '52-60']

    # Apply filters and clean data
    df = df[df['City'].isin(cities_to_include)]
    df = df[~df['Profession'].isin(professions_to_drop)]
    df = df[df['Sleep Duration'].isin(sleep_hours_to_keep)]
    df = df[df['Dietary Habits'].isin(dietary_habits_to_keep)]
    df = df[df['Degree'].isin(degrees_to_keep)]

    # Drop rows with null Financial Stress
    df.drop(df[df['Financial Stress'].isnull()].index, inplace=True)

    # Impute missing values with the median
    for column in columns_to_impute:
        df[column].fillna(df[column].median(), inplace=True)

    # Fill missing Profession based on Age
    df.loc[(df['Profession'].isnull()) & (df['Age'] < 25), 'Profession'] = 'Student'
    df.loc[(df['Profession'].isnull()) & (df['Age'] >= 25), 'Profession'] = 'Unemployed'

    # Drop unnecessary columns
    df.drop(['Name'], axis=1, inplace=True)

    # Bin ages into groups
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
    df.drop('Age', axis=1, inplace=True)

    return df

# Apply the function to both train and test datasets
train_data = preprocess_data(pd.read_csv('./data/processed/train.csv'))
test_data = preprocess_data(pd.read_csv('./data/processed/test.csv'))


# Create the interim data directory
interim_data_path = os.path.join("data", "interim")
os.makedirs(interim_data_path, exist_ok=True)

# Save the processed train and test data to the 'interim' directory
train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)


