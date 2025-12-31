# federated_logistic_regression.py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class FederatedLogisticRegression:
    """Logistic Regression for Federated Learning"""
    
    def __init__(self, num_classes, penalty='l2', C=1.0, max_iter=1000):
        self.num_classes = num_classes
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train logistic regression"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_weights(self):
        """Get model weights for federated averaging"""
        return {
            'coef': self.model.coef_,
            'intercept': self.model.intercept_
        }
    
    def set_weights(self, weights):
        """Set model weights from federated averaging"""
        self.model.coef_ = weights['coef']
        self.model.intercept_ = weights['intercept']

class LogisticRegressionFederatedClient:
    """Client for federated Logistic Regression"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, num_classes):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.model = None
    
    def train_local_model(self, global_weights=None):
        """Train local model"""
        self.model = FederatedLogisticRegression(
            num_classes=self.num_classes,
            C=1.0,
            max_iter=1000
        )
        
        # If global weights exist, initialize with them
        if global_weights is not None:
            self.model.set_weights(global_weights)
        
        # Train on local data
        self.model.train(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Client {self.client_id} - Accuracy: {accuracy*100:.2f}%")
        
        return self.model
    
    def get_weights(self):
        """Get model weights"""
        return self.model.get_weights()
    
    def get_sample_count(self):
        """Get number of training samples"""
        return len(self.X_train)

class LogisticRegressionFederatedServer:
    """Server for federated Logistic Regression"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.global_weights = None
    
    def aggregate_weights(self, client_weights_list, sample_counts):
        """
        Aggregate weights using weighted averaging (FedAvg)
        weighted by number of samples each client has
        """
        total_samples = sum(sample_counts)
        
        # Initialize aggregated weights
        aggregated_coef = np.zeros_like(client_weights_list[0]['coef'])
        aggregated_intercept = np.zeros_like(client_weights_list[0]['intercept'])
        
        # Weighted average
        for weights, count in zip(client_weights_list, sample_counts):
            weight = count / total_samples
            aggregated_coef += weight * weights['coef']
            aggregated_intercept += weight * weights['intercept']
        
        self.global_weights = {
            'coef': aggregated_coef,
            'intercept': aggregated_intercept
        }
        
        return self.global_weights