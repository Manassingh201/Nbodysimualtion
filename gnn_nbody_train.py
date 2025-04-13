import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NBodyGraphDataset:
    def __init__(self, csv_file, num_bodies=100, test_size=0.2, normalize=True):
        self.num_bodies = num_bodies
        print(f"Loading data from {csv_file}...")
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Data loaded: {self.df.shape}")
        
        # Extract feature columns and target columns
        feature_cols = [' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ', ' Mass']
        
        # Normalize features if requested
        if normalize:
            scaler = StandardScaler()
            self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])
            print("Features normalized")
        
        # Get unique timesteps
        self.timesteps = self.df['Time'].unique()
        print(f"Found {len(self.timesteps)} timesteps")
        
        # Split timesteps into train and test
        train_timesteps, test_timesteps = train_test_split(
            self.timesteps, test_size=test_size, random_state=42
        )
        
        self.train_timesteps = sorted(train_timesteps)
        self.test_timesteps = sorted(test_timesteps)
        
        print(f"Training on {len(self.train_timesteps)} timesteps")
        print(f"Testing on {len(self.test_timesteps)} timesteps")
    
    def _create_graph_data(self, timestep):
        """Create a graph for a specific timestep"""
        # Get data for this timestep
        df_t = self.df[self.df['Time'] == timestep]
        
        # Extract features
        features = df_t[[' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ', ' Mass']].values
        
        # Create node features
        x = torch.FloatTensor(features)
        
        # Create fully connected graph (each body connects to all others)
        edge_index = []
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.LongTensor(edge_index).t()
        
        # If this is not the last timestep, get the next timestep for targets
        if timestep < max(self.timesteps):
            next_timestep = self.timesteps[np.where(self.timesteps == timestep)[0][0] + 1]
            df_next = self.df[self.df['Time'] == next_timestep]
            targets = df_next[[' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ']].values
            y = torch.FloatTensor(targets)
        else:
            # For the last timestep, use the current values as targets
            y = torch.FloatTensor(features[:, :6])  # Exclude mass from targets
        
        # Create graph data
        return Data(x=x, edge_index=edge_index, y=y)
    
    def get_train_loader(self, batch_size=1):
        """Get training data loader"""
        train_dataset = []
        for t in tqdm(self.train_timesteps, desc="Preparing training data"):
            train_dataset.append(self._create_graph_data(t))
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_test_loader(self, batch_size=1):
        """Get test data loader"""
        test_dataset = []
        for t in tqdm(self.test_timesteps, desc="Preparing test data"):
            test_dataset.append(self._create_graph_data(t))
        
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class NBodyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NBodyGNN, self).__init__()
        
        # Graph Convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Output layer
        self.out = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch=None):
        # First Conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second Conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third Conv layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # No global pooling - we want predictions for each node (body)
        # Just apply final linear layer
        x = self.out(x)
        
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss - MSE over positions and velocities
        loss = F.mse_loss(out, data.y)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def run_training(csv_file, num_epochs=50, hidden_channels=64, batch_size=16):
    # Create dataset
    dataset = NBodyGraphDataset(csv_file)
    
    # Get data loaders
    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)
    
    # Create model
    # Input features: 7 (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass)
    # Output features: 6 (predict next pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
    model = NBodyGNN(in_channels=7, hidden_channels=hidden_channels, out_channels=6).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # For early stopping
    best_test_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # For tracking metrics
    train_losses = []
    test_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        test_loss = evaluate(model, test_loader)
        test_losses.append(test_loss)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {epoch_time:.2f}s')
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_nbody_gnn_model.pt')
            patience_counter = 0
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs!")
                break
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig('nbody_gnn_loss.png')
    
    print("Training completed!")
    print(f"Best test loss: {best_test_loss:.4f}")
    
    return model

def predict_simulation(model, initial_state, num_steps=50):
    """Run a physics simulation using the trained GNN model"""
    model.eval()
    
    # Make sure initial_state is a DataFrame with the correct structure
    assert isinstance(initial_state, pd.DataFrame)
    assert initial_state.shape[0] == 100  # 100 bodies
    
    # Store all predicted states
    all_states = [initial_state.copy()]
    
    # Current state to evolve
    current_state = initial_state.copy()
    
    with torch.no_grad():
        for step in tqdm(range(num_steps), desc="Simulating"):
            # Prepare graph from current state
            features = current_state[[' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ', ' Mass']].values
            x = torch.FloatTensor(features).to(device)
            
            # Create edges
            edge_index = []
            for i in range(100):  # 100 bodies
                for j in range(100):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.LongTensor(edge_index).t().to(device)
            
            # Run model to predict next state
            predictions = model(x, edge_index)
            predictions = predictions.cpu().numpy()
            
            # Update positions and velocities
            current_state[' PosX'] = predictions[:, 0]
            current_state[' PosY'] = predictions[:, 1]
            current_state[' PosZ'] = predictions[:, 2]
            current_state[' VelX'] = predictions[:, 3]
            current_state[' VelY'] = predictions[:, 4]
            current_state[' VelZ'] = predictions[:, 5]
            
            # Update time step
            current_state['Time'] = all_states[-1]['Time'].iloc[0] + 0.01  # Assuming 0.01 time step
            
            # Store this state
            all_states.append(current_state.copy())
    
    # Combine all states into a single DataFrame
    result = pd.concat(all_states)
    
    return result

def main():
    # File containing simulation data
    csv_file = 'final/nbody_simulation_data.csv'
    
    # Train the model
    model = run_training(
        csv_file=csv_file,
        num_epochs=50,
        hidden_channels=128,
        batch_size=16
    )
    
    # Optional: Use model to predict a simulation
    df = pd.read_csv(csv_file)
    initial_state = df[df['Time'] == df['Time'].min()].iloc[:100]  # First timestep
    
    print("\nRunning simulation with trained model...")
    prediction = predict_simulation(model, initial_state, num_steps=100)
    
    # Save prediction
    prediction.to_csv('gnn_nbody_simulation.csv', index=False)
    print("Simulation results saved to gnn_nbody_simulation.csv")

if __name__ == '__main__':
    main() 