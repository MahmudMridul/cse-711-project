# compare_federated_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time
from collections import defaultdict

# Import our federated learning implementations
from federated_xgboost import XGBoostFederatedClient, XGBoostFederatedServer
from federated_random_forest import RandomForestFederatedClient, RandomForestFederatedServer
from federated_logistic_regression import LogisticRegressionFederatedClient, LogisticRegressionFederatedServer

def load_and_prepare_client_data(client_file, label_encoders):
    """Load and prepare data for a single client"""
    df = pd.read_csv(client_file)
    
    # Prepare features
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
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def simulate_federated_xgboost(hospital_data, num_classes, num_rounds=5):
    """Simulate federated XGBoost training"""
    print("\n" + "="*70)
    print("FEDERATED XGBOOST")
    print("="*70)
    
    # Create clients
    clients = []
    for i, (hospital_name, data) in enumerate(hospital_data.items()):
        X_train, X_test, y_train, y_test = data
        client = XGBoostFederatedClient(i, X_train, y_train, X_test, y_test, num_classes)
        clients.append(client)
    
    # Create server
    server = XGBoostFederatedServer(num_classes)
    
    # Federated training rounds
    accuracies = []
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Each client trains locally
        client_models = []
        for client in clients:
            model = client.train_local_model(n_estimators=50)
            client_models.append(model)
        
        # Evaluate each client
        round_accuracies = []
        for client in clients:
            acc = client.evaluate()
            round_accuracies.append(acc)
        
        avg_accuracy = np.mean(round_accuracies)
        accuracies.append(avg_accuracy)
        print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
        
        # Server aggregates (simplified)
        # In production, implement proper model aggregation
    
    return accuracies

def simulate_federated_random_forest(hospital_data, num_classes):
    """Simulate federated Random Forest training"""
    print("\n" + "="*70)
    print("FEDERATED RANDOM FOREST")
    print("="*70)
    
    # Create clients
    clients = []
    for i, (hospital_name, data) in enumerate(hospital_data.items()):
        X_train, X_test, y_train, y_test = data
        client = RandomForestFederatedClient(
            i, X_train, y_train, X_test, y_test, trees_per_client=50
        )
        clients.append(client)
    
    # Create server
    server = RandomForestFederatedServer(num_classes)
    
    # Each client trains local forest
    print("\nTraining local forests...")
    client_trees = []
    accuracies = []
    
    for client in clients:
        model = client.train_local_model()
        trees = client.get_trees()
        client_trees.append(trees)
        acc = client.evaluate()
        accuracies.append(acc)
    
    # Server aggregates all trees
    global_trees = server.aggregate_forests(client_trees)
    
    avg_accuracy = np.mean(accuracies)
    print(f"\nAverage Local Accuracy: {avg_accuracy*100:.2f}%")
    
    return accuracies

def simulate_federated_logistic_regression(hospital_data, num_classes, num_rounds=10):
    """Simulate federated Logistic Regression training"""
    print("\n" + "="*70)
    print("FEDERATED LOGISTIC REGRESSION")
    print("="*70)
    
    # Create clients
    clients = []
    for i, (hospital_name, data) in enumerate(hospital_data.items()):
        X_train, X_test, y_train, y_test = data
        client = LogisticRegressionFederatedClient(
            i, X_train, y_train, X_test, y_test, num_classes
        )
        clients.append(client)
    
    # Create server
    server = LogisticRegressionFederatedServer(num_classes)
    
    # Federated training rounds
    accuracies = []
    global_weights = None
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Each client trains with current global weights
        client_weights = []
        sample_counts = []
        
        for client in clients:
            model = client.train_local_model(global_weights)
            weights = client.get_weights()
            count = client.get_sample_count()
            
            client_weights.append(weights)
            sample_counts.append(count)
        
        # Server aggregates weights
        global_weights = server.aggregate_weights(client_weights, sample_counts)
        
        # Evaluate with global weights
        round_accuracies = []
        for client in clients:
            client.model.set_weights(global_weights)
            y_pred = client.model.predict(client.X_test)
            acc = accuracy_score(client.y_test, y_pred)
            round_accuracies.append(acc)
        
        avg_accuracy = np.mean(round_accuracies)
        accuracies.append(avg_accuracy)
        print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
    
    return accuracies

def main():
    """Main comparison function"""
    
    # Load label encoders
    with open('data/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    num_classes = len(label_encoders['condition_type'].classes_)
    
    # Load data for each hospital
    import glob
    client_files = sorted(glob.glob('data/clients/*.csv'))
    
    hospital_data = {}
    for client_file in client_files:
        hospital_name = client_file.split('/')[-1].replace('.csv', '')
        data = load_and_prepare_client_data(client_file, label_encoders)
        hospital_data[hospital_name] = data
    
    print(f"\nLoaded data for {len(hospital_data)} hospitals")
    print(f"Number of condition types: {num_classes}")
    
    # Store results
    results = {}
    
    # Test each model
    start_time = time.time()
    
    # 1. XGBoost
    try:
        xgb_acc = simulate_federated_xgboost(hospital_data, num_classes, num_rounds=5)
        results['XGBoost'] = xgb_acc
    except Exception as e:
        print(f"XGBoost failed: {e}")
        results['XGBoost'] = None
    
    # 2. Random Forest
    try:
        rf_acc = simulate_federated_random_forest(hospital_data, num_classes)
        results['RandomForest'] = rf_acc
    except Exception as e:
        print(f"Random Forest failed: {e}")
        results['RandomForest'] = None
    
    # 3. Logistic Regression
    try:
        lr_acc = simulate_federated_logistic_regression(hospital_data, num_classes, num_rounds=10)
        results['LogisticRegression'] = lr_acc
    except Exception as e:
        print(f"Logistic Regression failed: {e}")
        results['LogisticRegression'] = None
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for model_name, accuracies in results.items():
        if accuracies:
            best_acc = max(accuracies)
            final_acc = accuracies[-1]
            print(f"\n{model_name}:")
            print(f"  Best Accuracy: {best_acc*100:.2f}%")
            print(f"  Final Accuracy: {final_acc*100:.2f}%")
    
    print(f"\nTotal training time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()