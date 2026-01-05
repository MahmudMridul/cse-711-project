import utils as u
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

pd.options.future.infer_string = True


class FederatedClient:
    """Represents a single hospital/client in federated learning"""
    
    def __init__(self, facility_name: str, data: pd.DataFrame):
        print(f"Initializing client for facility: {facility_name} with {len(data)} records")
        self.facility_name = facility_name
        self.data = data
        self.label_encoders = {}
        self.X_train = None
        self.y_train = None
        print(f"Client {facility_name} initialized.")
        
    def preprocess_data(self):
        """Preprocess data for the client"""
        df = self.data.copy()
        
        # Separate features and target
        y = df['length_of_stay'].values
        X = df.drop(['length_of_stay', 'facility_name'], axis=1)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'string']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(-1)
        
        self.X_train = X.values
        self.y_train = y
        
        return self.X_train, self.y_train
    
    def train_model_cv(self, model_type: str, n_splits: int = 5) -> Dict:
        """Train model with cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        trained_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # Initialize model based on type
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='hist',
                    device='cuda',
                    n_jobs=-1
                )
            elif model_type == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    n_jobs=-1,
                    verbose=-1
                )
            elif model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )    
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Validate
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)
            trained_models.append(model)

        print({
            'models': trained_models,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        })
        
        return {
            'models': trained_models,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
    
    
    def evaluate_models(self, models_dict: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all models for this client"""
        results = {}
        
        for model_type, models in models_dict.items():
            model_results = []
            for idx, model in enumerate(models):
                pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                model_results.append({
                    'model_index': idx + 1,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })
            results[model_type] = model_results
        
        return results


class FederatedEnsemble:
    """Manages federated learning across multiple clients with ensemble methods"""
    
    def __init__(self, data: pd.DataFrame, facility_column: str = 'facility_name'):
        print("Initializing Federated Ensemble...")
        self.data = data
        self.facility_column = facility_column
        self.clients = {}
        self.model_types = ['xgboost', 'lightgbm', 'random_forest']
        self.global_models = {model_type: [] for model_type in self.model_types}
        print("Federated Ensemble Initialized.")
        
    def create_clients(self, facility_names: List[str]):
        """Create client instances for each facility"""
        for facility in facility_names:
            print(f"\nCreating client for facility: {facility}")
            facility_data = self.data[self.data[self.facility_column] == facility].copy()
            if len(facility_data) > 0:
                self.clients[facility] = FederatedClient(facility, facility_data)
                print(f"Created client for {facility}: {len(facility_data)} records")
    
    def train_federated(self, n_splits: int = 5):
        """Train models across all clients using federated learning"""
        print("\n" + "="*70)
        print("STARTING FEDERATED TRAINING")
        print("="*70)
        
        for model_type in self.model_types:
            print(f"\n{'='*70}")
            print(f"Training {model_type.upper()} across all clients")
            print(f"{'='*70}")
            
            for facility_name, client in self.clients.items():
                print(f"\nClient: {facility_name}")
                
                # Preprocess data if not already done
                if client.X_train is None:
                    client.preprocess_data()
                
                # Train with cross-validation
                results = client.train_model_cv(model_type, n_splits)
                
                # Store trained models
                self.global_models[model_type].extend(results['models'])
                
                print(f"  CV RMSE: {results['mean_cv_score']:.4f} (+/- {results['std_cv_score']:.4f})")
                print(f"  Fold scores: {[f'{s:.4f}' for s in results['cv_scores']]}")
    
    def predict_ensemble(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble of all models"""
        all_predictions = []
        
        for model_type in self.model_types:
            model_predictions = []
            for model in self.global_models[model_type]:
                pred = model.predict(X_test)
                model_predictions.append(pred)
            
            # Average predictions from same model type
            avg_pred = np.mean(model_predictions, axis=0)
            all_predictions.append(avg_pred)
        
        # Average across all model types
        ensemble_prediction = np.mean(all_predictions, axis=0)
        return ensemble_prediction
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Evaluate ensemble model on test data"""
        print("\n" + "="*70)
        print("OVERALL ENSEMBLE EVALUATION")
        print("="*70)
        
        # Get predictions from each model type
        predictions_by_type = {}
        for model_type in self.model_types:
            type_predictions = []
            for model in self.global_models[model_type]:
                pred = model.predict(X_test)
                type_predictions.append(pred)
            predictions_by_type[model_type] = np.mean(type_predictions, axis=0)
        
        # Ensemble prediction
        ensemble_pred = self.predict_ensemble(X_test)
        
        # Evaluate each model type
        print("\nIndividual Model Type Performance:")
        for model_type, pred in predictions_by_type.items():
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            print(f"\n{model_type.upper()}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
        
        # Evaluate ensemble
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"\nOVERALL ENSEMBLE:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        overall_results = {
            'ensemble_rmse': rmse,
            'ensemble_mae': mae,
            'ensemble_r2': r2,
            'total_test_samples': len(y_test),
            'num_clients': len(self.clients),
            'num_model_types': len(self.model_types),
            'total_models': sum(len(models) for models in self.global_models.values())
        }
        
        return ensemble_pred, overall_results
        
        def evaluate_per_client(self, test_df: pd.DataFrame):
            """Evaluate models on each client's test data separately"""
            print("\n" + "="*70)
            print("PER-CLIENT EVALUATION")
            print("="*70)
            
            client_results = []
            client_ensemble_results = []
            individual_model_results = []
            
            for facility_name, client in self.clients.items():
                print(f"\nEvaluating on {facility_name} test data...")
                
                # Get test data for this facility
                facility_test = test_df[test_df['facility_name'] == facility_name].copy()
                
                if len(facility_test) == 0:
                    print(f"  No test data available for {facility_name}")
                    continue
                
                X_test, y_test = prepare_test_data(facility_test, client.label_encoders)
                
                print(f"  Test samples: {len(y_test)}")
                
                # Evaluate each model type
                predictions_by_type = {}
                for model_type in self.model_types:
                    type_predictions = []
                    for idx, model in enumerate(self.global_models[model_type]):
                        pred = model.predict(X_test)
                        type_predictions.append(pred)
                        
                        # Store individual model results
                        rmse = np.sqrt(mean_squared_error(y_test, pred))
                        mae = mean_absolute_error(y_test, pred)
                        r2 = r2_score(y_test, pred)
                        
                        individual_model_results.append({
                            'facility_name': facility_name,
                            'model_type': model_type,
                            'model_index': idx + 1,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'test_samples': len(y_test)
                        })
                    
                    # Average predictions for this model type
                    avg_pred = np.mean(type_predictions, axis=0)
                    predictions_by_type[model_type] = avg_pred
                    
                    # Calculate metrics for averaged model type
                    rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
                    mae = mean_absolute_error(y_test, avg_pred)
                    r2 = r2_score(y_test, avg_pred)
                    
                    print(f"  {model_type.upper()}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                
                # Ensemble prediction (average across all model types)
                ensemble_pred = np.mean(list(predictions_by_type.values()), axis=0)
                
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                
                print(f"  ENSEMBLE: RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")
                
                # Store client ensemble results
                client_ensemble_results.append({
                    'facility_name': facility_name,
                    'ensemble_rmse': ensemble_rmse,
                    'ensemble_mae': ensemble_mae,
                    'ensemble_r2': ensemble_r2,
                    'test_samples': len(y_test)
                })
            
            return individual_model_results, client_ensemble_results

    def evaluate_per_client(self, test_df: pd.DataFrame):
        """Evaluate models on each client's test data separately"""
        print("\n" + "="*70)
        print("PER-CLIENT EVALUATION")
        print("="*70)
        
        client_results = []
        client_ensemble_results = []
        individual_model_results = []
    
        for facility_name, client in self.clients.items():
            print(f"\nEvaluating on {facility_name} test data...")
            
            # Get test data for this facility
            facility_test = test_df[test_df['facility_name'] == facility_name].copy()
            
            if len(facility_test) == 0:
                print(f"  No test data available for {facility_name}")
                continue
            
            X_test, y_test = prepare_test_data(facility_test, client.label_encoders)
            
            print(f"  Test samples: {len(y_test)}")
            
            # Evaluate each model type
            predictions_by_type = {}
            for model_type in self.model_types:
                type_predictions = []
                for idx, model in enumerate(self.global_models[model_type]):
                    pred = model.predict(X_test)
                    type_predictions.append(pred)
                    
                    # Store individual model results
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    mae = mean_absolute_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    
                    individual_model_results.append({
                        'facility_name': facility_name,
                        'model_type': model_type,
                        'model_index': idx + 1,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'test_samples': len(y_test)
                    })
                
                # Average predictions for this model type
                avg_pred = np.mean(type_predictions, axis=0)
                predictions_by_type[model_type] = avg_pred
                
                # Calculate metrics for averaged model type
                rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
                mae = mean_absolute_error(y_test, avg_pred)
                r2 = r2_score(y_test, avg_pred)
                
                print(f"  {model_type.upper()}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            # Ensemble prediction (average across all model types)
            ensemble_pred = np.mean(list(predictions_by_type.values()), axis=0)
            
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            
            print(f"  ENSEMBLE: RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")
            
            # Store client ensemble results
            client_ensemble_results.append({
                'facility_name': facility_name,
                'ensemble_rmse': ensemble_rmse,
                'ensemble_mae': ensemble_mae,
                'ensemble_r2': ensemble_r2,
                'test_samples': len(y_test)
            })
        
        return individual_model_results, client_ensemble_results

    def save_results_to_csv(self, individual_results: List[Dict], 
                        client_ensemble_results: List[Dict],
                        overall_results: Dict,
                        output_dir: str = '.'):
        """Save all results to CSV files"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Individual model results per client
        df_individual = pd.DataFrame(individual_results)
        df_individual = df_individual.sort_values(['facility_name', 'model_type', 'model_index'])
        individual_file = os.path.join(output_dir, 'individual_model_results.csv')
        df_individual.to_csv(individual_file, index=False)
        print(f"\n✓ Saved individual model results to: {individual_file}")
        
        # 2. Ensemble results per client
        df_client_ensemble = pd.DataFrame(client_ensemble_results)
        df_client_ensemble = df_client_ensemble.sort_values('facility_name')
        client_ensemble_file = os.path.join(output_dir, 'client_ensemble_results.csv')
        df_client_ensemble.to_csv(client_ensemble_file, index=False)
        print(f"✓ Saved client ensemble results to: {client_ensemble_file}")
        
        # 3. Overall ensemble results
        df_overall = pd.DataFrame([overall_results])
        overall_file = os.path.join(output_dir, 'overall_ensemble_results.csv')
        df_overall.to_csv(overall_file, index=False)
        print(f"✓ Saved overall ensemble results to: {overall_file}")

def prepare_test_data(test_df: pd.DataFrame, label_encoders: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare test data for evaluation
    
    Args:
        test_df: Test dataframe
        label_encoders: Dictionary of label encoders from training (optional)
    
    Returns:
        X_test, y_test as numpy arrays
    """
    df = test_df.copy()
    
    # Separate target
    y_test = df['length_of_stay'].values
    
    # Drop target and facility_name
    X_test = df.drop(['length_of_stay'], axis=1)
    if 'facility_name' in X_test.columns:
        X_test = X_test.drop(['facility_name'], axis=1)
    
    # Encode categorical variables
    categorical_cols = X_test.select_dtypes(include=['object', 'string']).columns
    
    for col in categorical_cols:
        if label_encoders and col in label_encoders:
            # Use existing encoder
            le = label_encoders[col]
            # Handle unseen categories
            X_test[col] = X_test[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            # Create new encoder
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
    
    # Handle missing values
    X_test = X_test.fillna(-1)
    
    return X_test.values, y_test

# Main execution code
if __name__ == "__main__":
    
    # Load your data
    print("Loading data...")
    df = pd.read_parquet(u.FILE_V9)
    
    # Define facility names
    facility_names = [
        "Mount Sinai Hospital",
        "North Shore University Hospital",
        "New York Presbyterian Hospital - Columbia Presbyterian Center",
        "New York Presbyterian Hospital - New York Weill Cornell Center",
        "Montefiore Medical Center - Henry & Lucy Moses Div",
        "Maimonides Medical Center",
        "Long Island Jewish Medical Center",
        "New York Methodist Hospital",
        "Strong Memorial Hospital",
        "Albany Medical Center Hospital",
        "University Hospital",
        "Winthrop-University Hospital",
        "Mount Sinai Beth Israel",
        "NYU Hospitals Center",
        "Staten Island University Hosp-North"
    ]
    
    # Filter data for specified facilities only
    df = df[df['facility_name'].isin(facility_names)].copy()
    print(f"Total records: {len(df)}")
    
    # Split data into train and test (80-20 split)
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['facility_name']  # Ensure all facilities in both sets
    )
    
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")
    
    print("Initialize federated ensemble with training data")
    fed_ensemble = FederatedEnsemble(train_df, facility_column='facility_name')
    
    # Create clients for each facility
    fed_ensemble.create_clients(facility_names)
    
    # Train federated models with cross-validation
    fed_ensemble.train_federated(n_splits=5)

    first_client = list(fed_ensemble.clients.values())[0]
    X_test, y_test = prepare_test_data(test_df, first_client.label_encoders)
    
    print(f"\nTest set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 1. Evaluate per client
    individual_results, client_ensemble_results = fed_ensemble.evaluate_per_client(test_df)
    
    # 2. Evaluate overall ensemble
    predictions, overall_results = fed_ensemble.evaluate_on_test(X_test, y_test)
    
    # 3. Save all results to CSV
    fed_ensemble.save_results_to_csv(
        individual_results,
        client_ensemble_results,
        overall_results,
        output_dir='./federated_results'
    )
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)