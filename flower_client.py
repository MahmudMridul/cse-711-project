# flower_client.py
import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from federated_model import ConditionPredictor, get_parameters, set_parameters, train_model, evaluate_model
from data_preparation import prepare_features_and_target
import pickle

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_path, label_encoders, num_conditions, input_dim):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load client data
        df = pd.read_csv(data_path)
        X, y, _ = prepare_features_and_target(df, label_encoders)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        self.model = ConditionPredictor(input_dim, num_conditions)
        
        print(f"Client {client_id} initialized with {len(X_train)} training samples")
    
    def get_parameters(self, config):
        """Return current model parameters"""
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """Train model with parameters from server"""
        set_parameters(self.model, parameters)
        
        epochs = config.get("epochs", 5)
        learning_rate = config.get("learning_rate", 0.001)
        
        print(f"\nClient {self.client_id} training for {epochs} epochs...")
        loss, accuracy = train_model(
            self.model, self.train_loader, epochs, learning_rate, self.device
        )
        
        return get_parameters(self.model), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate model with parameters from server"""
        set_parameters(self.model, parameters)
        
        loss, accuracy = evaluate_model(self.model, self.test_loader, self.device)
        
        print(f"Client {self.client_id} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

def create_client(client_id, data_path, label_encoders, num_conditions, input_dim):
    """Factory function to create a client"""
    return HospitalClient(client_id, data_path, label_encoders, num_conditions, input_dim)