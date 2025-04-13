import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import glob

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NBodyGraphDataset:
    def __init__(self, data_dir='final', test_size=0.2, normalize=True, add_physics_features=True):
        self.num_bodies = 100
        self.add_physics_features = add_physics_features
        
        print(f"Loading data from {data_dir}...")
        
        # Get all CSV files in the directory
        file_list = glob.glob(os.path.join(data_dir, "*.csv"))
        print(f"Found {len(file_list)} CSV files")
        
        # Load and combine all data
        dfs = []
        for file in tqdm(file_list, desc="Loading files"):
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Total data loaded: {self.df.shape}")
        
        # Extract feature columns and target columns
        self.feature_cols = [' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ', ' Mass']
        
        # Normalize features if requested
        if normalize:
            scaler = StandardScaler()
            self.df[self.feature_cols] = scaler.fit_transform(self.df[self.feature_cols])
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
    
    def _add_physics_features(self, df_t):
        """Add physics-informed features to the data"""
        features = df_t[self.feature_cols].values
        num_bodies = features.shape[0]
        
        # For each body, add relative position and velocity to every other body
        # Also add distance and gravitational force
        extra_features = []
        
        for i in range(num_bodies):
            body_i = features[i]
            pos_i = body_i[0:3]  # x, y, z
            vel_i = body_i[3:6]  # vx, vy, vz
            mass_i = body_i[6]   # mass
            
            # Calculate relative features to all other bodies
            body_features = []
            
            for j in range(num_bodies):
                if i == j:
                    continue
                
                body_j = features[j]
                pos_j = body_j[0:3]
                mass_j = body_j[6]
                
                # Relative position (vector from j to i)
                rel_pos = pos_i - pos_j
                
                # Distance between bodies (squared to avoid sqrt)
                dist_squared = np.sum(rel_pos**2)
                dist = np.sqrt(dist_squared)
                
                # Gravitational force magnitude (G=1 for simplicity)
                if dist > 1e-10:  # Avoid division by zero
                    grav_force = mass_i * mass_j / dist_squared
                else:
                    grav_force = 0
                
                # Direction of force (normalized relative position)
                if dist > 1e-10:
                    force_direction = rel_pos / dist
                else:
                    force_direction = np.zeros(3)
                
                # Force vector
                force_vector = -grav_force * force_direction  # Attractive force is negative
                
                # Features to add: distance, grav_force, force_vector (3)
                body_features.extend([dist, grav_force, *force_vector])
            
            # Add aggregated features (mean, max)
            body_features = np.array(body_features)
            aggr_features = [
                np.mean(body_features),
                np.max(body_features),
                np.min(body_features)
            ]
            
            extra_features.append(aggr_features)
        
        # Convert to numpy array and combine with original features
        extra_features = np.array(extra_features)
        enhanced_features = np.concatenate([features, extra_features], axis=1)
        
        return enhanced_features
    
    def _create_graph_data(self, timestep):
        """Create a graph for a specific timestep"""
        # Get data for this timestep
        df_t = self.df[self.df['Time'] == timestep]
        
        # Extract features
        if self.add_physics_features:
            features = self._add_physics_features(df_t)
        else:
            features = df_t[self.feature_cols].values
        
        # Create node features
        x = torch.FloatTensor(features)
        
        # Create fully connected graph (each body connects to all others)
        edge_index = []
        edge_attr = []  # For storing edge features
        
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j:
                    edge_index.append([i, j])
                    
                    # Calculate edge features
                    pos_i = df_t.iloc[i][[' PosX', ' PosY', ' PosZ']].values
                    pos_j = df_t.iloc[j][[' PosX', ' PosY', ' PosZ']].values
                    rel_pos = pos_i - pos_j
                    distance = np.linalg.norm(rel_pos)
                    
                    # Edge features: distance, relative position
                    edge_feat = [distance, *rel_pos]
                    edge_attr.append(edge_feat)
        
        edge_index = torch.LongTensor(edge_index).t()
        edge_attr = torch.FloatTensor(edge_attr)
        
        # If this is not the last timestep, get the next timestep for targets
        if timestep < max(self.timesteps):
            next_timestep = self.timesteps[np.where(self.timesteps == timestep)[0][0] + 1]
            df_next = self.df[self.df['Time'] == next_timestep]
            targets = df_next[[' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ']].values
            y = torch.FloatTensor(targets)
        else:
            # For the last timestep, use the current values as targets
            y = torch.FloatTensor(df_t[[' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ']].values)
        
        # Create graph data
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
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

class PhysicsMessagePassing(MessagePassing):
    """Message passing layer with physics-based information"""
    def __init__(self, in_channels, out_channels):
        super(PhysicsMessagePassing, self).__init__(aggr='add')  # "Add" aggregation
        
        # Neural networks for transforming node and edge features
        self.node_mlp = torch.nn.Sequential(
            Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            Linear(out_channels, out_channels)
        )
        
        # Neural networks for calculating messages
        self.message_mlp = torch.nn.Sequential(
            Linear(in_channels + 4, out_channels),  # node features + edge features
            torch.nn.ReLU(),
            Linear(out_channels, out_channels)
        )
        
        # Update function
        self.update_mlp = torch.nn.Sequential(
            Linear(in_channels + out_channels, out_channels),
            torch.nn.ReLU(),
            Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        node_features = self.node_mlp(x)
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, node_features=node_features)
    
    def message(self, x_i, x_j, edge_attr):
        # Create message based on source node, destination node, and edge features
        # x_i: destination node features [num_edges, in_channels]
        # x_j: source node features [num_edges, in_channels]
        # edge_attr: edge features [num_edges, 4]
        
        # Concatenate source node features with edge features
        message_input = torch.cat([x_j, edge_attr], dim=1)
        
        # Calculate message
        return self.message_mlp(message_input)
    
    def update(self, aggr_out, x):
        # aggr_out: aggregated messages [num_nodes, out_channels]
        # x: original node features [num_nodes, in_channels]
        
        # Concatenate node features with aggregated messages
        update_input = torch.cat([x, aggr_out], dim=1)
        
        # Update node features
        return self.update_mlp(update_input)

class PhysicsInformedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=4, num_layers=3):
        super(PhysicsInformedGNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # Message passing layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(PhysicsMessagePassing(hidden_channels, hidden_channels))
        
        # Output projection
        self.output_proj = Linear(hidden_channels, out_channels)
        
        # Layer normalization
        self.layer_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
        
        # Dropout
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        # Initial projection
        x = self.input_proj(x)
        
        # Message passing layers with residual connections
        for i in range(self.num_layers):
            # Message passing
            x_res = self.convs[i](x, edge_index, edge_attr)
            
            # Residual connection
            x = x + x_res
            
            # Layer normalization
            x = self.layer_norms[i](x)
            
            # Dropout
            x = self.dropout(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr)
        
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
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def run_training(data_dir='final', num_epochs=100, hidden_channels=128, batch_size=8, patience=15):
    # Create dataset with physics-informed features
    dataset = NBodyGraphDataset(data_dir=data_dir, add_physics_features=True)
    
    # Get data loaders
    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)
    
    # Get the number of features (original + physics-informed)
    sample_data = next(iter(train_loader))
    in_channels = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1)
    
    print(f"Input features: {in_channels}")
    print(f"Edge features: {edge_dim}")
    
    # Create model
    # Input: node features (original + physics-informed)
    # Output: 6 (predict next pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
    model = PhysicsInformedGNN(
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        out_channels=6,
        edge_dim=edge_dim,
        num_layers=4
    ).to(device)
    
    # Print model summary
    print(model)
    
    # Optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # For early stopping
    best_test_loss = float('inf')
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
        
        # Update learning rate
        scheduler.step(test_loss)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Time: {epoch_time:.2f}s')
        
        # Early stopping with longer patience
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_nbody_gnn_model_improved.pt')
            print("New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs!")
                break
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_losses, label='Train Loss')  # Using log scale
    plt.semilogy(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('nbody_gnn_loss_improved.png')
    
    print("Training completed!")
    print(f"Best test loss: {best_test_loss:.6f}")
    
    # Load the best model
    model.load_state_dict(torch.load('best_nbody_gnn_model_improved.pt'))
    
    return model, dataset

def predict_simulation(model, dataset, num_steps=100):
    """Run a physics simulation using the trained GNN model"""
    model.eval()
    
    # Get initial state (first timestep)
    initial_timestep = dataset.timesteps[0]
    initial_df = dataset.df[dataset.df['Time'] == initial_timestep]
    
    # Create the initial graph
    initial_graph = dataset._create_graph_data(initial_timestep)
    
    # Variables to store the simulation
    all_predictions = [initial_df[[' PosX', ' PosY', ' PosZ', ' VelX', ' VelY', ' VelZ']].values]
    all_times = [initial_timestep]
    
    # Current state
    current_x = initial_graph.x.to(device)
    current_edge_index = initial_graph.edge_index.to(device)
    current_edge_attr = initial_graph.edge_attr.to(device)
    
    print("Running simulation...")
    
    with torch.no_grad():
        for step in tqdm(range(num_steps), desc="Simulating"):
            # Run model to predict next state
            next_state = model(current_x, current_edge_index, current_edge_attr)
            next_state_np = next_state.cpu().numpy()
            
            # Store prediction
            all_predictions.append(next_state_np)
            
            # Update time
            all_times.append(all_times[-1] + 0.01)  # Assuming 0.01 time step
            
            # Update current state for next iteration - need to update x with the predictions
            # Keep the physics-informed features unchanged for simplicity
            if dataset.add_physics_features:
                # Update only the position and velocity parts of the features
                # Assuming the first 6 elements are pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
                current_x[:, :6] = next_state
            else:
                # If not using physics features, just update positions and velocities
                # and keep mass unchanged
                current_x[:, :6] = next_state
    
    # Create DataFrame for all predictions
    result_dfs = []
    
    for i, (time, pred) in enumerate(zip(all_times, all_predictions)):
        # Create DataFrame for this timestep
        df = pd.DataFrame()
        df['Time'] = [time] * dataset.num_bodies
        df[' BodyID'] = range(dataset.num_bodies)
        df[' PosX'] = pred[:, 0]
        df[' PosY'] = pred[:, 1]
        df[' PosZ'] = pred[:, 2]
        df[' VelX'] = pred[:, 3]
        df[' VelY'] = pred[:, 4]
        df[' VelZ'] = pred[:, 5]
        
        # Add mass from the initial state (unchanged)
        df[' Mass'] = initial_df[' Mass'].values
        
        result_dfs.append(df)
    
    # Combine all DataFrames
    result = pd.concat(result_dfs, ignore_index=True)
    
    return result

def main():
    # Train the model
    model, dataset = run_training(
        data_dir='final',
        num_epochs=150,
        hidden_channels=256,
        batch_size=4,
        patience=20
    )
    
    # Use model to predict a simulation
    print("\nRunning simulation with trained model...")
    prediction = predict_simulation(model, dataset, num_steps=200)
    
    # Save prediction
    prediction.to_csv('gnn_nbody_simulation_improved.csv', index=False)
    print("Simulation results saved to gnn_nbody_simulation_improved.csv")

if __name__ == '__main__':
    main() 