# predict_admissions.py
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from federated_model import ConditionPredictor
from data_preparation import prepare_features_and_target

def load_trained_model(model_path, input_dim, num_conditions):
    """Load the trained federated model"""
    model = ConditionPredictor(input_dim, num_conditions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_conditions_for_date(hospital_name, target_date, model, label_encoders, scaler):
    """
    Predict condition types likely to result in admissions for a hospital on a specific date
    """
    
    # Prepare input features
    date_obj = pd.to_datetime(target_date)
    
    # Create a sample input (you'd need historical weather data for accurate predictions)
    input_data = {
        'year': date_obj.year,
        'month': date_obj.month,
        'day': date_obj.day,
        'day_of_week': date_obj.dayofweek,
        'day_of_year': date_obj.dayofyear,
        'hospital_name_encoded': label_encoders['hospital_name'].transform([hospital_name])[0],
        # Add other features with reasonable defaults or fetch from weather API
        'patient_age_group_encoded': 0,  # You'd want to iterate over all age groups
        'patient_gender_encoded': 0,
        'severity_level_encoded': 1,
        'seasonal_indicator_encoded': (date_obj.month % 12 + 3) // 3 - 1,
        'comorbid_conditions_count': 1,
        'daily_medication_dosage': 20.0,
        'emergency_visit_count': 2,
        'readmission_count': 1,
        'temperature_2m_mean_C': 25.0,  # Default values
        'temperature_2m_max_C': 30.0,
        'temperature_2m_min_C': 20.0,
        'apparent_temperature_mean_C': 24.0,
        'apparent_temperature_max_C': 29.0,
        'apparent_temperature_min_C': 19.0,
        'wind_speed_10m_max_km_h': 15.0,
        'wind_gusts_10m_max_km_h': 25.0,
        'wind_direction_10m_dominant_degree': 180.0,
        'shortwave_radiation_sum_MJ_m_square': 20.0,
    }
    
    # Convert to numpy array
    X = np.array([list(input_data.values())])
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
    
    # Get condition names and probabilities
    conditions = label_encoders['condition_type'].classes_
    results = list(zip(conditions, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

# Usage example
if __name__ == "__main__":
    # Load label encoders
    with open('data/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Load trained model (you'd save this after federated training)
    # model = load_trained_model('models/federated_model.pth', 24, num_conditions)
    
    # Example prediction
    hospital = "King Saud Hospital"
    date = "2025-01-15"
    
    print(f"\nPredicted condition admissions for {hospital} on {date}:")
    print("-" * 60)
    # results = predict_conditions_for_date(hospital, date, model, label_encoders, scaler)
    # for condition, prob in results:
    #     print(f"{condition:20s}: {prob*100:5.2f}%")