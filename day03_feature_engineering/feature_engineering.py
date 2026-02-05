import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def clean_data():
    print("--- Loading Titanic Dataset ---")
    df = sns.load_dataset('titanic')

    # 1. Drop useless columns
    df = df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class'])

    # 2. Impute Missing Values
    # Fill Age with Median
    imputer = SimpleImputer(strategy='median')
    df['age'] = imputer.fit_transform(df[['age']])
    # Fill Embarked with Mode (Most Frequent)
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # 3. Encode Categorical Data
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex']) # Male/Female -> 0/1
    df['embarked'] = le.fit_transform(df['embarked']) # S/C/Q -> 0/1/2

    # 4. Feature Engineering
    df['family_size'] = df['sibsp'] + df['parch']

    # 5. Scale Numerical Values (Age and Fare)
    scaler = StandardScaler()
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

    # Save to CSV for the next steps
    df.to_csv('titanic_cleaned.csv', index=False)
    print("Data Cleaned and Saved to 'titanic_cleaned.csv'")

if __name__ == "__main__":
    clean_data()
