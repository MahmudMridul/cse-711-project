# federated_server.py
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from clients using weighted average"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def start_federated_server(num_rounds=10, min_clients=2):
    """Start the federated learning server"""
    
    # Define strategy with FedAvg (Federated Averaging)
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=min_clients,  # Minimum clients for training
        min_evaluate_clients=min_clients,  # Minimum clients for evaluation
        min_available_clients=min_clients,  # Wait for all clients to be available
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate evaluation metrics
    )
    
    # Start Flower server
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Server")
    print(f"Number of rounds: {num_rounds}")
    print(f"Minimum clients: {min_clients}")
    print(f"{'='*60}\n")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_federated_server(num_rounds=20, min_clients=8)