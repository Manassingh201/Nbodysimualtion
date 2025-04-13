import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

def test_data_loading():
    """Test function to load and examine the data structure"""
    print("Testing data loading...")
    data_dir = 'final'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not file_list:
        print("No CSV files found in the final directory!")
        return
    
    # Load first file
    file_path = os.path.join(data_dir, file_list[0])
    print(f"Loading file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print("\nDataFrame Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        print("\nShape of DataFrame:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_graph_from_data(df):
    """Test function to create a graph from the data"""
    print("\nCreating graph from data...")
    try:
        # Get the actual column names
        columns = df.columns.tolist()
        print(f"Actual columns: {columns}")
        
        # Select only the relevant feature columns (assuming they are in first 7 columns)
        feature_cols = columns[:7] if len(columns) >= 7 else columns
        print(f"Using feature columns: {feature_cols}")
        
        # Extract features for all bodies
        features = df[feature_cols].values
        print(f"Features shape: {features.shape}")
        
        # Let's check a small sample of the data
        print(f"Sample of features (first 5 rows):\n{features[:5]}")
        
        # Each row might represent a body at a specific timestep
        # assuming 100 bodies and 5001 timesteps (500100 / 100 = 5001)
        num_bodies = 100
        num_timesteps = len(df) // num_bodies
        print(f"Assuming {num_timesteps} timesteps for {num_bodies} bodies")
        
        # For testing, let's just use data from the first timestep
        first_timestep_data = features[:num_bodies]
        print(f"First timestep data shape: {first_timestep_data.shape}")
        
        # Create node features for the first timestep
        x = torch.FloatTensor(first_timestep_data)
        print(f"Node features tensor shape: {x.shape}")
        
        # Create edges (fully connected graph)
        edge_index = []
        for i in range(num_bodies):
            for j in range(num_bodies):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.LongTensor(edge_index).t()
        print(f"Edge index shape: {edge_index.shape}")
        
        # Create the graph data object
        data = Data(x=x, edge_index=edge_index)
        print("Graph data created successfully!")
        return data, features, num_timesteps, num_bodies
    except Exception as e:
        print(f"Error creating graph: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # If no batch, just take the mean of all node features
            x = torch.mean(x, dim=0, keepdim=True)
        
        x = self.lin(x)
        return x

def test_model(num_features):
    """Test function to create and run a simple model"""
    print("\nTesting model creation and forward pass...")
    try:
        # Create a small test graph
        num_nodes = 100  # Number of bodies
        x = torch.randn(num_nodes, num_features)
        
        # Create a fully connected graph
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.LongTensor(edge_index).t()
        
        # Create model
        model = SimpleGNN(num_features=num_features, hidden_channels=32, num_classes=num_features)
        print("Model created successfully!")
        
        # Test forward pass
        out = model(x, edge_index)
        print(f"Output shape: {out.shape}")
        print("Forward pass successful!")
        return model
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Starting debug process...")
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        return
    
    # Test graph creation
    data, features, num_timesteps, num_bodies = create_graph_from_data(df)
    if data is None:
        return
    
    # Test model with the actual number of features
    num_features = data.x.shape[1]
    model = test_model(num_features)
    if model is None:
        return
    
    print("\nAll tests completed successfully!")

if __name__ == '__main__':
    main() 