import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

class NBodyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(NBodyDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.scaler = StandardScaler()
        
    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        # Read the CSV file
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        df = pd.read_csv(file_path)
        
        # Assuming the CSV contains columns: x, y, z, vx, vy, vz, mass for each body
        # We'll create a graph where each node represents a body
        num_bodies = len(df) // 7  # Assuming 7 features per body
        
        # Reshape the data
        features = df.values.reshape(num_bodies, 7)
        
        # Create node features
        x = torch.FloatTensor(features)
        
        # Create edges (fully connected graph)
        edge_index = []
        for i in range(num_bodies):
            for j in range(num_bodies):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.LongTensor(edge_index).t()
        
        # Create the graph data object
        data = Data(x=x, edge_index=edge_index)
        
        if self.transform:
            data = self.transform(data)
            
        return data

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Third Graph Convolution Layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final linear layer
        x = self.lin(x)
        
        return x

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.batch)
        
        # Compute loss (you'll need to define your loss function based on your specific task)
        loss = F.mse_loss(out, data.y)  # Example loss function
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = NBodyDataset(data_dir='final')
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = GNN(num_features=7, hidden_channels=64, num_classes=7).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train_model(model, train_loader, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'gnn_nbody_model.pth')

if __name__ == '__main__':
    main() 