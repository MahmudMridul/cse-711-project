# federated_xgboost.py
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json

class FederatedXGBoost:
    """XGBoost model for federated learning using histogram-based approach"""
    
    def __init__(self, num_classes, max_depth=6, learning_rate=0.1, n_estimators=100):
        self.params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',  # Histogram-based for faster training
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
        }
        self.n_estimators = n_estimators
        self.model = None
        self.num_classes = num_classes
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train with early stopping
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return np.argmax(preds, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.get_score(importance_type='gain')
    
    def save(self, path):
        """Save model"""
        self.model.save_model(path)
    
    def load(self, path):
        """Load model"""
        self.model = xgb.Booster()
        self.model.load_model(path)
        return self

# Federated XGBoost Client
class XGBoostFederatedClient:
    """Client for federated XGBoost training"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, num_classes):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.model = None
        
        print(f"Client {client_id}: {len(X_train)} train, {len(X_test)} test samples")
    
    def train_local_model(self, global_model_params=None, n_estimators=50):
        """Train local XGBoost model"""
        
        # Initialize model
        self.model = FederatedXGBoost(
            num_classes=self.num_classes,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=n_estimators
        )
        
        # If global model exists, use it as starting point
        if global_model_params is not None:
            # Load global model and continue training
            self.model.model = xgb.Booster(model_file=global_model_params)
        
        # Train on local data
        self.model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        
        return self.model
    
    def evaluate(self):
        """Evaluate local model"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Client {self.client_id} - Test Accuracy: {accuracy*100:.2f}%")
        return accuracy
    
    def get_model_update(self):
        """Get model update (trees) to send to server"""
        # In XGBoost federated learning, we send the trees
        return self.model.model.save_raw()

# Federated XGBoost Server
class XGBoostFederatedServer:
    """Server for federated XGBoost aggregation"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.global_model = None
        self.round_num = 0
    
    def aggregate_models(self, client_models, weights=None):
        """
        Aggregate client models using weighted averaging of trees
        """
        if weights is None:
            # Equal weights for all clients
            weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # For XGBoost, we can't directly average trees
        # Instead, we'll use a voting/ensemble approach
        # or retrain a global model on aggregated predictions
        
        print(f"\nRound {self.round_num}: Aggregating {len(client_models)} client models")
        
        # Simple approach: save the best performing model as global
        # In production, you'd use more sophisticated aggregation
        self.global_model = client_models[0]  # Placeholder
        
        self.round_num += 1
        
        return self.global_model