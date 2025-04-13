import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_simulation_comparison(original_csv, gnn_csv, save_path='nbody_comparison.mp4'):
    """
    Create a 3D animation comparing the original and GNN-predicted simulations
    
    Args:
        original_csv: Path to the original simulation CSV
        gnn_csv: Path to the GNN-predicted simulation CSV
        save_path: Path to save the animation MP4
    """
    print("Loading original simulation data...")
    original_df = pd.read_csv(original_csv)
    
    print("Loading GNN-predicted simulation data...")
    gnn_df = pd.read_csv(gnn_csv)
    
    # Get all unique timesteps
    original_timesteps = original_df['Time'].unique()
    gnn_timesteps = gnn_df['Time'].unique()
    
    # Select common timesteps for comparison
    common_timesteps = sorted(set(original_timesteps) & set(gnn_timesteps))
    if len(common_timesteps) > 100:
        # If too many timesteps, sample 100 evenly
        indices = np.linspace(0, len(common_timesteps) - 1, 100, dtype=int)
        common_timesteps = [common_timesteps[i] for i in indices]
    
    print(f"Creating animation with {len(common_timesteps)} frames...")
    
    # Initialize the figure and 3D axes
    fig = plt.figure(figsize=(15, 7))
    
    # Original simulation plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Simulation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # GNN-predicted simulation plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('GNN Predicted Simulation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Find min and max coordinates for consistent axes limits
    min_x = min(original_df[' PosX'].min(), gnn_df[' PosX'].min())
    max_x = max(original_df[' PosX'].max(), gnn_df[' PosX'].max())
    min_y = min(original_df[' PosY'].min(), gnn_df[' PosY'].min())
    max_y = max(original_df[' PosY'].max(), gnn_df[' PosY'].max())
    min_z = min(original_df[' PosZ'].min(), gnn_df[' PosZ'].min())
    max_z = max(original_df[' PosZ'].max(), gnn_df[' PosZ'].max())
    
    # Set axis limits
    ax1.set_xlim([min_x, max_x])
    ax1.set_ylim([min_y, max_y])
    ax1.set_zlim([min_z, max_z])
    
    ax2.set_xlim([min_x, max_x])
    ax2.set_ylim([min_y, max_y])
    ax2.set_zlim([min_z, max_z])
    
    # Initialize scatter plots with empty data
    original_scatter = ax1.scatter([], [], [], s=2, c='blue', alpha=0.8)
    gnn_scatter = ax2.scatter([], [], [], s=2, c='red', alpha=0.8)
    
    # Add timestamp text
    time_text = fig.text(0.5, 0.9, '', ha='center')
    
    # Update function for animation
    def update(frame):
        timestep = common_timesteps[frame]
        
        # Get data for this timestep
        original_frame_data = original_df[original_df['Time'] == timestep]
        gnn_frame_data = gnn_df[gnn_df['Time'] == timestep]
        
        # Update scatter plot data
        original_scatter._offsets3d = (original_frame_data[' PosX'], 
                                     original_frame_data[' PosY'], 
                                     original_frame_data[' PosZ'])
        
        gnn_scatter._offsets3d = (gnn_frame_data[' PosX'], 
                                gnn_frame_data[' PosY'], 
                                gnn_frame_data[' PosZ'])
        
        # Update timestamp
        time_text.set_text(f'Time: {timestep:.2f}')
        
        return original_scatter, gnn_scatter, time_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(common_timesteps), 
                        blit=False, interval=50)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    ani.save(save_path, writer=writer)
    
    print(f"Animation saved to {save_path}")

def visualize_simulation_3d(csv_file, title='N-body Simulation', save_path='nbody_simulation.mp4'):
    """
    Create a 3D animation of a single simulation
    
    Args:
        csv_file: Path to the simulation CSV
        title: Title for the animation
        save_path: Path to save the animation MP4
    """
    print(f"Loading simulation data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get all unique timesteps
    timesteps = df['Time'].unique()
    if len(timesteps) > 100:
        # If too many timesteps, sample 100 evenly
        indices = np.linspace(0, len(timesteps) - 1, 100, dtype=int)
        timesteps = [timesteps[i] for i in indices]
    
    print(f"Creating animation with {len(timesteps)} frames...")
    
    # Initialize the figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Find min and max coordinates for consistent axes limits
    min_x = df[' PosX'].min()
    max_x = df[' PosX'].max()
    min_y = df[' PosY'].min()
    max_y = df[' PosY'].max()
    min_z = df[' PosZ'].min()
    max_z = df[' PosZ'].max()
    
    # Set axis limits
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    
    # Initialize scatter plot with empty data
    scatter = ax.scatter([], [], [], s=10, c=[], cmap='viridis', alpha=0.8)
    
    # Add timestamp text
    time_text = fig.text(0.5, 0.9, '', ha='center')
    
    # Update function for animation
    def update(frame):
        timestep = timesteps[frame]
        
        # Get data for this timestep
        frame_data = df[df['Time'] == timestep]
        
        # Update scatter plot data
        scatter._offsets3d = (frame_data[' PosX'], 
                             frame_data[' PosY'], 
                             frame_data[' PosZ'])
        
        # Color by mass or velocity magnitude
        colors = np.sqrt(frame_data[' VelX']**2 + frame_data[' VelY']**2 + frame_data[' VelZ']**2)
        scatter.set_array(colors)
        
        # Update timestamp
        time_text.set_text(f'Time: {timestep:.2f}')
        
        return scatter, time_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(timesteps), 
                        blit=False, interval=50)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    ani.save(save_path, writer=writer)
    
    print(f"Animation saved to {save_path}")

def plot_trajectory_comparison(original_csv, gnn_csv, body_ids=[0, 1, 2], save_path='trajectory_comparison.png'):
    """
    Plot the trajectories of selected bodies to compare original and GNN predictions
    
    Args:
        original_csv: Path to the original simulation CSV
        gnn_csv: Path to the GNN-predicted simulation CSV
        body_ids: List of body IDs to plot
        save_path: Path to save the plot
    """
    print("Loading data for trajectory comparison...")
    orig_df = pd.read_csv(original_csv)
    gnn_df = pd.read_csv(gnn_csv)
    
    # Get common timesteps
    orig_times = orig_df['Time'].unique()
    gnn_times = gnn_df['Time'].unique()
    common_times = sorted(set(orig_times) & set(gnn_times))
    
    fig, axes = plt.subplots(len(body_ids), 3, figsize=(15, 5*len(body_ids)))
    
    for i, body_id in enumerate(body_ids):
        orig_body_data = orig_df[orig_df[' BodyID'] == body_id]
        gnn_body_data = gnn_df[gnn_df[' BodyID'] == body_id]
        
        # Only use common timesteps
        orig_body_data = orig_body_data[orig_body_data['Time'].isin(common_times)]
        gnn_body_data = gnn_body_data[gnn_body_data['Time'].isin(common_times)]
        
        # Plot X position over time
        axes[i, 0].plot(orig_body_data['Time'], orig_body_data[' PosX'], 'b-', label='Original')
        axes[i, 0].plot(gnn_body_data['Time'], gnn_body_data[' PosX'], 'r--', label='GNN')
        axes[i, 0].set_title(f'Body {body_id} - X Position')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('X Position')
        axes[i, 0].legend()
        
        # Plot Y position over time
        axes[i, 1].plot(orig_body_data['Time'], orig_body_data[' PosY'], 'b-', label='Original')
        axes[i, 1].plot(gnn_body_data['Time'], gnn_body_data[' PosY'], 'r--', label='GNN')
        axes[i, 1].set_title(f'Body {body_id} - Y Position')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Y Position')
        axes[i, 1].legend()
        
        # Plot Z position over time
        axes[i, 2].plot(orig_body_data['Time'], orig_body_data[' PosZ'], 'b-', label='Original')
        axes[i, 2].plot(gnn_body_data['Time'], gnn_body_data[' PosZ'], 'r--', label='GNN')
        axes[i, 2].set_title(f'Body {body_id} - Z Position')
        axes[i, 2].set_xlabel('Time')
        axes[i, 2].set_ylabel('Z Position')
        axes[i, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Trajectory comparison saved to {save_path}")

def main():
    original_csv = 'final/nbody_simulation_data.csv'
    gnn_csv = 'gnn_nbody_simulation.csv'
    
    # Check if the GNN prediction file exists
    if not os.path.exists(gnn_csv):
        print(f"GNN prediction file {gnn_csv} not found. Run gnn_nbody_train.py first.")
        return
    
    # Visualize the original simulation
    visualize_simulation_3d(original_csv, title='Original N-body Simulation', 
                           save_path='original_simulation.mp4')
    
    # Visualize the GNN-predicted simulation
    visualize_simulation_3d(gnn_csv, title='GNN-Predicted N-body Simulation', 
                           save_path='gnn_simulation.mp4')
    
    # Create a comparison visualization
    visualize_simulation_comparison(original_csv, gnn_csv, 
                                   save_path='simulation_comparison.mp4')
    
    # Plot trajectory comparisons for a few bodies
    plot_trajectory_comparison(original_csv, gnn_csv, body_ids=[0, 10, 50], 
                              save_path='trajectory_comparison.png')

if __name__ == '__main__':
    main() 