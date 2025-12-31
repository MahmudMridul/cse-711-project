# federated_random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle

class FederatedRandomForest:
    """Random Forest for Federated Learning"""
    
    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = None
    
    def train(self, X_train, y_train):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_trees(self):
        """Get individual trees from the forest"""
        return self.model.estimators_
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.model.feature_importances_

class RandomForestFederatedClient:
    """Client for federated Random Forest"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, 
                 trees_per_client=20):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.trees_per_client = trees_per_client
        self.model = None
    
    def train_local_model(self):
        """Train local Random Forest"""
        self.model = FederatedRandomForest(
            n_estimators=self.trees_per_client,
            max_depth=15,
            min_samples_split=5
        )
        
        self.model.train(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Client {self.client_id} - Accuracy: {accuracy*100:.2f}%")
        
        return self.model
    
    def get_trees(self):
        """Get trees to send to server"""
        return self.model.get_trees()
    
    def evaluate(self, model=None):
        """Evaluate model"""
        if model is None:
            model = self.model
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        return accuracy

class RandomForestFederatedServer:
    """Server for federated Random Forest"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.global_forest = None
        self.all_trees = []
    
    def aggregate_forests(self, client_trees_list):
        """
        Aggregate Random Forests from all clients
        by combining all their trees into one global forest
        """
        print(f"\nAggregating forests from {len(client_trees_list)} clients")
        
        # Collect all trees from all clients
        self.all_trees = []
        for client_trees in client_trees_list:
            self.all_trees.extend(client_trees)
        
        print(f"Total trees in global forest: {len(self.all_trees)}")
        
        # Create a new RandomForestClassifier with all trees
        # Note: This is a simplified approach
        # In practice, you might want to select best performing trees
        
        return self.all_trees
    
    def create_global_model(self):
        """Create global model from aggregated trees"""
        # This requires some sklearn internals manipulation
        # Simplified version: use the aggregated trees
        return self.all_trees