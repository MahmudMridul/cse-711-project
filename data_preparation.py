# data_preparation.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def load_and_prepare_data(csv_path):
    """Load and preprocess the hospital data"""
    df = pd.read_csv(csv_path)
    
    # Convert admission_date to datetime
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    
    # Extract temporal features
    df['year'] = df['admission_date'].dt.year
    df['month'] = df['admission_date'].dt.month
    df['day'] = df['admission_date'].dt.day
    df['day_of_week'] = df['admission_date'].dt.dayofweek
    df['day_of_year'] = df['admission_date'].dt.dayofyear
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['hospital_name', 'condition_type', 'patient_age_group', 
                       'patient_gender', 'severity_level', 'seasonal_indicator']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

def partition_data_by_hospital(df):
    """Partition data by hospital to simulate federated clients"""
    hospital_data = {}
    hospitals = df['hospital_name'].unique()
    
    for hospital in hospitals:
        hospital_df = df[df['hospital_name'] == hospital].copy()
        hospital_data[hospital] = hospital_df
        print(f"{hospital}: {len(hospital_df)} records")
    
    return hospital_data

def prepare_features_and_target(df, label_encoders):
    """Prepare features (X) and target (y) for modeling"""
    
    # Features for prediction
    feature_cols = [
        'year', 'month', 'day', 'day_of_week', 'day_of_year',
        'hospital_name_encoded',
        'patient_age_group_encoded', 'patient_gender_encoded',
        'severity_level_encoded', 'seasonal_indicator_encoded',
        'comorbid_conditions_count', 'daily_medication_dosage',
        'emergency_visit_count', 'readmission_count',
        'temperature_2m_mean_C', 'temperature_2m_max_C', 'temperature_2m_min_C',
        'apparent_temperature_mean_C', 'apparent_temperature_max_C', 
        'apparent_temperature_min_C', 'wind_speed_10m_max_km_h',
        'wind_gusts_10m_max_km_h', 'wind_direction_10m_dominant_degree',
        'shortwave_radiation_sum_MJ_m_square'
    ]
    
    X = df[feature_cols].values
    y = df['condition_type_encoded'].values
    
    return X, y, feature_cols

# Usage
if __name__ == "__main__":
    # Load data
    df, label_encoders = load_and_prepare_data('data/data_clean.csv')
    
    # Partition by hospital
    hospital_data = partition_data_by_hospital(df)
    
    # Save partitioned data
    import os
    os.makedirs('data/clients', exist_ok=True)
    
    for hospital_name, hospital_df in hospital_data.items():
        safe_name = hospital_name.replace(' ', '_').replace('/', '_')
        hospital_df.to_csv(f'data/clients/{safe_name}.csv', index=False)
    
    # Save label encoders
    with open('data/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print("\nData preparation complete!")