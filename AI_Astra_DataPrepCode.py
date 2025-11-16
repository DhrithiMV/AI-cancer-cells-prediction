import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def generate_synthetic_data():
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'Patient_ID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 85, size=n_samples),
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.48, 0.52]),
        'Height_cm': np.random.normal(170, 10, size=n_samples),
        'Weight_kg': np.random.normal(75, 15, size=n_samples),
        'BMI': np.random.uniform(18, 35, size=n_samples),
        'Systolic_BP': np.random.normal(120, 20, size=n_samples),
        'Diastolic_BP': np.random.normal(80, 15, size=n_samples),
        'Heart_Rate': np.random.normal(70, 12, size=n_samples),
        'Temperature_F': np.random.normal(98.6, 1.5, size=n_samples),
        'Blood_Sugar': np.random.normal(100, 30, size=n_samples),
        'Cholesterol': np.random.normal(200, 40, size=n_samples),
        'Hemoglobin': np.random.normal(14, 2, size=n_samples),
        'Smoking': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.25, 0.75]),
        'Alcohol_Consumption': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7]),
        'Exercise_Hours_Week': np.random.poisson(3, size=n_samples),
        'Diabetes': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.15, 0.85]),
        'Hypertension': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.20, 0.80]),
        'Heart_Disease': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.10, 0.90]),
        'Hospital_Visits_Year': np.random.poisson(2, size=n_samples),
        'Insurance_Type': np.random.choice(['Public', 'Private', 'Uninsured'], size=n_samples, p=[0.4, 0.5, 0.1]),
        'Risk_Score': np.random.uniform(0, 100, size=n_samples)
    })

    # Correct BMI
    data['BMI'] = data['Weight_kg'] / ((data['Height_cm'] / 100) ** 2)

    # Target variable
    data['Target_variable'] = np.where(
        (data['Risk_Score'] > 70) |
        (data['Heart_Disease'] == 'Yes') |
        (data['Diabetes'] == 'Yes'), 1, 0
    )

    # ---- Impurities ----
    # Outliers in Age
    outlier_indices_age = np.random.choice(data.index, size=5, replace=False)
    data.loc[outlier_indices_age, 'Age'] = 8000

    # Unrealistic weights
    outlier_indices_weight = np.random.choice(data.index, size=5, replace=False)
    data.loc[outlier_indices_weight[:3], 'Weight_kg'] = 0.5
    data.loc[outlier_indices_weight[3:], 'Weight_kg'] = 1000

    # Inconsistent gender labels
    gender_boy_indices = data[data['Gender'] == 'Male'].sample(frac=0.02, random_state=42).index
    gender_girl_indices = data[data['Gender'] == 'Female'].sample(frac=0.02, random_state=42).index
    data.loc[gender_boy_indices, 'Gender'] = 'Boy'
    data.loc[gender_girl_indices, 'Gender'] = 'Girl'

    # Duplicate rows
    num_duplicates = np.random.randint(5, 11)
    duplicate_rows = data.sample(n=num_duplicates, random_state=42)
    data = pd.concat([data, duplicate_rows], ignore_index=True)

    return data

def clean_patient_data(df):
    # 1. Fix Age outliers
    df.loc[(df['Age'] < 18) | (df['Age'] > 85), 'Age'] = np.nan
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # 2. Fix Weight outliers
    df.loc[(df['Weight_kg'] < 30) | (df['Weight_kg'] > 200), 'Weight_kg'] = np.nan
    df['Weight_kg'].fillna(df['Weight_kg'].median(), inplace=True)

    # 3. Fix Gender labels
    gender_map = {'Boy': 'Male', 'Girl': 'Female', 'Male': 'Male', 'Female': 'Female'}
    df['Gender'] = df['Gender'].map(gender_map)

    # 4. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 5. Recalculate BMI
    df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)

    return df

if __name__ == "__main__":
    print("Generating synthetic patient data with impurities...")
    dirty_data = generate_synthetic_data()
    print(f"Dirty dataset shape: {dirty_data.shape}")

    # Save dirty data
    dirty_data.to_csv('dirty_patient_data.csv', index=False)

    print("Cleaning dataset...")
    cleaned_data = clean_patient_data(dirty_data)
    cleaned_data.to_csv("AI_Astra_CleanedDataset.csv", index=False)

    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print("Cleaning completed. Saved to 'AI_Astra_CleanedDataset.csv'")
