# run_federated_simulation.py
import subprocess
import time
import os
import glob
import pickle
import pandas as pd
from multiprocessing import Process

def start_server():
    """Start the federated server"""
    print("Starting federated server...")
    os.system("python federated_server.py")

def start_client(client_id, data_path, label_encoders_path):
    """Start a federated client"""
    import flwr as fl
    from flower_client import create_client
    import pickle
    
    print(f"Starting client {client_id}...")
    
    # Load label encoders
    with open(label_encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Determine number of conditions and input dimensions
    num_conditions = len(label_encoders['condition_type'].classes_)
    input_dim = 24  # Number of features we're using
    
    # Create and start client
    client = create_client(client_id, data_path, label_encoders, num_conditions, input_dim)
    
    # Connect to server
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )

def run_simulation():
    """Run the complete federated learning simulation"""
    
    # Get all client data files
    client_files = sorted(glob.glob('data/clients/*.csv'))
    label_encoders_path = 'data/label_encoders.pkl'
    
    if not client_files:
        print("Error: No client data files found. Run data_preparation.py first.")
        return
    
    print(f"\nFound {len(client_files)} hospital clients")
    for i, f in enumerate(client_files):
        print(f"  Client {i}: {os.path.basename(f)}")
    
    # Start server in a separate process
    server_process = Process(target=start_server)
    server_process.start()
    
    # Wait for server to start
    time.sleep(5)
    
    # Start all clients in separate processes
    client_processes = []
    for i, client_file in enumerate(client_files):
        p = Process(target=start_client, args=(i, client_file, label_encoders_path))
        p.start()
        client_processes.append(p)
        time.sleep(1)  # Stagger client starts
    
    # Wait for all clients to finish
    for p in client_processes:
        p.join()
    
    # Terminate server
    server_process.terminate()
    server_process.join()
    
    print("\n" + "="*60)
    print("Federated Learning Simulation Complete!")
    print("="*60)

if __name__ == "__main__":
    run_simulation()