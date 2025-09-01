import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Import your existing modules (assuming they're available)
from eventpoints import EventPointNet, custom_collate_fn
from loadevents import event_extractor

class ODEFunc(nn.Module):
    """
    Neural ODE function that learns latent dynamics dz/dt = f(z, t)
    """
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim
        
        # Neural network that learns the dynamics function f(z, t)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize with small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute dz/dt = f(z, t)
        Args:
            t: [1] current time (scalar for all batch elements)
            z: [B, latent_dim] current latent state
        Returns:
            dz_dt: [B, latent_dim] time derivative of latent state
        """
        batch_size = z.shape[0]
        
        # Expand time to match batch size
        t_expanded = t.expand(batch_size, 1)  # [B, 1]
        
        # Concatenate latent state with time
        z_t = torch.cat([z, t_expanded], dim=1)  # [B, latent_dim + 1]
        
        # Compute dynamics
        dz_dt = self.net(z_t)  # [B, latent_dim]
        
        return dz_dt

class PoseRegressionHead(nn.Module):
    """
    Maps latent state to 6-DOF pose (position + orientation quaternion)
    """
    def __init__(self, latent_dim: int = 256, pose_dim: int = 7):
        super(PoseRegressionHead, self).__init__()
        
        self.pose_regressor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pose_dim)  # [x, y, z, qx, qy, qz, qw]
        )
        
        # Initialize last layer with small weights
        nn.init.normal_(self.pose_regressor[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.pose_regressor[-1].bias, val=0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim] or [B, T, latent_dim] latent states
        Returns:
            poses: [B, 7] or [B, T, 7] predicted poses
        """
        original_shape = z.shape
        if len(original_shape) == 3:  # [B, T, latent_dim]
            B, T, D = original_shape
            z = z.view(B * T, D)
            poses = self.pose_regressor(z)  # [B*T, 7]
            poses = poses.view(B, T, 7)  # [B, T, 7]
        else:  # [B, latent_dim]
            poses = self.pose_regressor(z)  # [B, 7]
        
        # Normalize quaternion (last 4 elements)
        if len(poses.shape) == 3:  # [B, T, 7]
            quat = poses[:, :, 3:]  # [B, T, 4]
            quat_norm = F.normalize(quat, p=2, dim=-1)
            poses = torch.cat([poses[:, :, :3], quat_norm], dim=-1)
        else:  # [B, 7]
            quat = poses[:, 3:]  # [B, 4]
            quat_norm = F.normalize(quat, p=2, dim=-1)
            poses = torch.cat([poses[:, :3], quat_norm], dim=-1)
        
        return poses

class NeuralODEEventPosePredictor(nn.Module):
    """
    Complete system: EventPointNet + Neural ODE + Pose Regression
    """
    def __init__(self, 
                 latent_dim: int = 256,
                 ode_hidden_dim: int = 512,
                 use_stereo: bool = True,
                 rtol: float = 1e-3,
                 atol: float = 1e-4):
        super(NeuralODEEventPosePredictor, self).__init__()
        
        self.latent_dim = latent_dim
        self.use_stereo = use_stereo
        
        # Event encoder (your existing PointNet++)
        self.event_encoder = EventPointNet(latent_dim=latent_dim, use_stereo=use_stereo)
        
        # Neural ODE dynamics function
        self.ode_func = ODEFunc(latent_dim=latent_dim, hidden_dim=ode_hidden_dim)
        
        # Pose regression head
        self.pose_head = PoseRegressionHead(latent_dim=latent_dim, pose_dim=7)
        
        # ODE solver parameters
        self.rtol = rtol
        self.atol = atol
    
    def encode_events(self, left_events: torch.Tensor, 
                     right_events: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode event data to initial latent state z0
        """
        z0 = self.event_encoder(left_events, right_events)  # [B, latent_dim]
        return z0
    
    def predict_trajectory(self, z0: torch.Tensor, 
                          time_steps: torch.Tensor) -> torch.Tensor:
        """
        Predict latent trajectory using Neural ODE
        Args:
            z0: [B, latent_dim] initial latent state
            time_steps: [T] time points to predict at
        Returns:
            z_trajectory: [B, T, latent_dim] latent states over time
        """
        # Solve ODE: integrate from t=0 to time_steps
        z_trajectory = odeint(
            self.ode_func,
            z0,
            time_steps,
            rtol=self.rtol,
            atol=self.atol,
            method='dopri5'  # Adaptive Runge-Kutta method
        )  # [T, B, latent_dim]
        
        # Transpose to [B, T, latent_dim]
        z_trajectory = z_trajectory.permute(1, 0, 2)
        
        return z_trajectory
    
    def forward(self, left_events: torch.Tensor,
                time_steps: torch.Tensor,
                right_events: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: Events -> Latent -> ODE -> Poses
        Args:
            left_events: [B, N, 4] left camera events
            time_steps: [T] time points to predict poses at
            right_events: [B, N, 4] right camera events (optional)
        Returns:
            Dictionary containing:
                - z0: [B, latent_dim] initial latent state
                - z_trajectory: [B, T, latent_dim] latent trajectory
                - poses: [B, T, 7] predicted poses over time
        """
        # 1. Encode events to initial latent state
        z0 = self.encode_events(left_events, right_events)  # [B, latent_dim]
        
        # 2. Predict latent trajectory using Neural ODE
        z_trajectory = self.predict_trajectory(z0, time_steps)  # [B, T, latent_dim]
        
        # 3. Predict poses from latent trajectory
        poses = self.pose_head(z_trajectory)  # [B, T, 7]
        
        return {
            'z0': z0,
            'z_trajectory': z_trajectory,
            'poses': poses
        }

class PoseLoss(nn.Module):
    """
    Combined loss for position and orientation
    """
    def __init__(self, position_weight: float = 1.0, orientation_weight: float = 1.0):
        super(PoseLoss, self).__init__()
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
    
    def quaternion_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute angular distance between quaternions
        Args:
            q1, q2: [..., 4] quaternions (x, y, z, w)
        Returns:
            angular_distance: [...] angular distance in radians
        """
        # Ensure quaternions are normalized
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # Compute dot product (cosine of half angle)
        dot_product = torch.sum(q1 * q2, dim=-1)
        
        # Handle sign ambiguity (q and -q represent same rotation)
        dot_product = torch.abs(dot_product)
        
        # Clamp to avoid numerical issues
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Angular distance: 2 * arccos(|dot_product|)
        angular_distance = 2.0 * torch.acos(dot_product)
        
        return angular_distance
    
    def forward(self, pred_poses: torch.Tensor, 
                gt_poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_poses: [B, T, 7] predicted poses
            gt_poses: [B, T, 7] ground truth poses
        Returns:
            Dictionary with total loss and components
        """
        # Split position and orientation
        pred_pos = pred_poses[..., :3]  # [B, T, 3]
        pred_quat = pred_poses[..., 3:]  # [B, T, 4]
        gt_pos = gt_poses[..., :3]  # [B, T, 3]
        gt_quat = gt_poses[..., 3:]  # [B, T, 4]
        
        # Position loss (L2)
        position_loss = F.mse_loss(pred_pos, gt_pos)
        
        # Orientation loss (quaternion angular distance)
        orientation_loss = self.quaternion_distance(pred_quat, gt_quat).mean()
        
        # Combined loss
        total_loss = (self.position_weight * position_loss + 
                     self.orientation_weight * orientation_loss)
        
        return {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'orientation_loss': orientation_loss
        }

def create_time_sequence(duration: float = 1.0, num_steps: int = 10) -> torch.Tensor:
    """
    Create time sequence for ODE integration
    Args:
        duration: total time duration
        num_steps: number of time steps
    Returns:
        time_steps: [num_steps] time points
    """
    return torch.linspace(0, duration, num_steps)

def prepare_batch_for_ode(batch: Dict, device: torch.device, 
                         num_time_steps: int = 10) -> Dict:
    """
    Prepare batch data for Neural ODE training
    """
    # Move data to device
    left_events = batch['left_events'].to(device)  # [B, N, 4]
    right_events = batch['right_events'].to(device) if 'right_events' in batch else None
    left_poses = batch['left_poses'].to(device)  # [B, 7]
    right_poses = batch['right_poses'].to(device)  # [B, 7]
    
    # Create time sequence
    time_steps = create_time_sequence(duration=1.0, num_steps=num_time_steps).to(device)
    
    # For training, we'll predict poses at these time steps
    # Here we'll create a simple interpolation between left and right poses
    batch_size = left_events.shape[0]
    gt_poses = torch.zeros(batch_size, num_time_steps, 7).to(device)
    
    for i in range(num_time_steps):
        alpha = i / (num_time_steps - 1) if num_time_steps > 1 else 0
        # Linear interpolation for position
        gt_poses[:, i, :3] = (1 - alpha) * left_poses[:, :3] + alpha * right_poses[:, :3]
        # SLERP for quaternion (simplified - using linear interpolation for demo)
        gt_poses[:, i, 3:] = (1 - alpha) * left_poses[:, 3:] + alpha * right_poses[:, 3:]
        # Normalize quaternion
        gt_poses[:, i, 3:] = F.normalize(gt_poses[:, i, 3:], p=2, dim=-1)
    
    return {
        'left_events': left_events,
        'right_events': right_events,
        'time_steps': time_steps,
        'gt_poses': gt_poses
    }

def train_neural_ode_model(model: NeuralODEEventPosePredictor,
                          dataloader: DataLoader,
                          num_epochs: int = 100,
                          lr: float = 1e-3,
                          device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          save_dir: str = "./checkpoints",
                          save_every: int = 10,
                          save_best: bool = True):
    """
    Training loop for Neural ODE Event Pose Predictor with model saving
    Args:
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        save_best: Whether to save best model based on validation loss
    """
    import os
    from datetime import datetime
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    pose_loss_fn = PoseLoss(position_weight=1.0, orientation_weight=0.1)
    
    # Training tracking
    best_loss = float('inf')
    training_history = {
        'epoch': [],
        'avg_loss': [],
        'pos_loss': [],
        'orient_loss': [],
        'lr': []
    }
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_pos_loss = 0.0
        total_orient_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Prepare batch
                prepared_batch = prepare_batch_for_ode(batch, device, num_time_steps=5)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(
                    prepared_batch['left_events'],
                    prepared_batch['time_steps'],
                    prepared_batch['right_events']
                )
                
                # Compute loss
                loss_dict = pose_loss_fn(outputs['poses'], prepared_batch['gt_poses'])
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_pos_loss += loss_dict['position_loss'].item()
                total_orient_loss += loss_dict['orientation_loss'].item()
                num_batches += 1
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: '
                          f'Loss={loss.item():.6f}, '
                          f'Pos={loss_dict["position_loss"].item():.6f}, '
                          f'Orient={loss_dict["orientation_loss"].item():.6f}')
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch averages
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_pos = total_pos_loss / num_batches
            avg_orient = total_orient_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            
            # Update training history
            training_history['epoch'].append(epoch + 1)
            training_history['avg_loss'].append(avg_loss)
            training_history['pos_loss'].append(avg_pos)
            training_history['orient_loss'].append(avg_orient)
            training_history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
            print(f'  Average Loss: {avg_loss:.6f}')
            print(f'  Average Position Loss: {avg_pos:.6f}')
            print(f'  Average Orientation Loss: {avg_orient:.6f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if save_best and avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_loss, 
                              training_history, best_model_path)
                print(f'  ✅ New best model saved! Loss: {best_loss:.6f}')
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_loss, 
                              training_history, checkpoint_path)
                print(f'  💾 Checkpoint saved at epoch {epoch+1}')
            
            print('-' * 50)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'final_model_{timestamp}.pth')
    save_checkpoint(model, optimizer, scheduler, num_epochs, avg_loss, 
                   training_history, final_model_path)
    print(f'\n🎉 Training completed! Final model saved: {final_model_path}')
    
    return training_history

def save_checkpoint(model: NeuralODEEventPosePredictor, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   training_history: Dict,
                   filepath: str):
    """
    Save model checkpoint with all training state
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'training_history': training_history,
        'model_config': {
            'latent_dim': model.latent_dim,
            'use_stereo': model.use_stereo,
            'rtol': model.rtol,
            'atol': model.atol
        }
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath: str, 
                   device: torch.device = torch.device('cpu')) -> Dict:
    """
    Load model checkpoint
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
    Returns:
        Dictionary containing model, optimizer, scheduler, and metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Reconstruct model from config
    model_config = checkpoint['model_config']
    model = NeuralODEEventPosePredictor(
        latent_dim=model_config['latent_dim'],
        use_stereo=model_config['use_stereo'],
        rtol=model_config['rtol'],
        atol=model_config['atol']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Reconstruct optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Loss: {checkpoint['loss']:.6f}")
    print(f"   Model config: {model_config}")
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'training_history': checkpoint['training_history']
    }

def save_model_for_inference(model: NeuralODEEventPosePredictor, 
                           filepath: str):
    """
    Save model for inference only (lighter weight)
    """
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'latent_dim': model.latent_dim,
            'use_stereo': model.use_stereo,
            'rtol': model.rtol,
            'atol': model.atol
        }
    }
    torch.save(model_data, filepath)
    print(f"💾 Inference model saved: {filepath}")

def load_model_for_inference(filepath: str, 
                           device: torch.device = torch.device('cpu')) -> NeuralODEEventPosePredictor:
    """
    Load model for inference only
    """
    model_data = torch.load(filepath, map_location=device)
    
    # Reconstruct model
    config = model_data['model_config']
    model = NeuralODEEventPosePredictor(
        latent_dim=config['latent_dim'],
        use_stereo=config['use_stereo'],
        rtol=config['rtol'],
        atol=config['atol']
    )
    
    # Load weights
    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Inference model loaded: {config}")
    return model

def plot_training_history(training_history: Dict, save_path: str = None):
    """
    Plot training history curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = training_history['epoch']
    
    # Total loss
    axes[0, 0].plot(epochs, training_history['avg_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Average Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Position loss
    axes[0, 1].plot(epochs, training_history['pos_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Position Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].grid(True)
    
    # Orientation loss
    axes[1, 0].plot(epochs, training_history['orient_loss'], 'g-', linewidth=2)
    axes[1, 0].set_title('Orientation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Angular Loss (rad)')
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(epochs, training_history['lr'], 'orange', linewidth=2)
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved: {save_path}")
    
    plt.show()

def visualize_trajectory_prediction(model: NeuralODEEventPosePredictor,
                                   sample_batch: Dict,
                                   device: torch.device):
    """
    Visualize predicted trajectory vs ground truth
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare sample
        prepared_batch = prepare_batch_for_ode(sample_batch, device, num_time_steps=20)
        
        # Predict
        outputs = model(
            prepared_batch['left_events'][:1],  # Take first sample
            prepared_batch['time_steps'],
            prepared_batch['right_events'][:1] if prepared_batch['right_events'] is not None else None
        )
        
        # Extract data for plotting
        pred_poses = outputs['poses'][0].cpu().numpy()  # [T, 7]
        gt_poses = prepared_batch['gt_poses'][0].cpu().numpy()  # [T, 7]
        time_steps = prepared_batch['time_steps'].cpu().numpy()  # [T]
        
        # Plot trajectories
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Position plots
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[0, 0].plot(time_steps, pred_poses[:, i], 'r-', label=f'Pred {label}', alpha=0.7)
            axes[0, 0].plot(time_steps, gt_poses[:, i], 'b--', label=f'GT {label}', alpha=0.7)
        axes[0, 0].set_title('Position Trajectory')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Orientation plots (quaternion components)
        for i, label in enumerate(['qx', 'qy', 'qz', 'qw']):
            axes[0, 1].plot(time_steps, pred_poses[:, i+3], 'r-', alpha=0.7)
            axes[0, 1].plot(time_steps, gt_poses[:, i+3], 'b--', alpha=0.7)
        axes[0, 1].set_title('Orientation Trajectory (Quaternion)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Quaternion Components')
        axes[0, 1].grid(True)
        
        # 3D trajectory
        axes[1, 0].plot(pred_poses[:, 0], pred_poses[:, 1], 'r-', label='Predicted', alpha=0.7, linewidth=2)
        axes[1, 0].plot(gt_poses[:, 0], gt_poses[:, 1], 'b--', label='Ground Truth', alpha=0.7, linewidth=2)
        axes[1, 0].scatter(pred_poses[0, 0], pred_poses[0, 1], c='red', s=100, marker='o', label='Start')
        axes[1, 0].scatter(pred_poses[-1, 0], pred_poses[-1, 1], c='red', s=100, marker='s', label='End')
        axes[1, 0].set_title('2D Trajectory (X-Y)')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axis('equal')
        
        # Latent space visualization
        z_trajectory = outputs['z_trajectory'][0].cpu().numpy()  # [T, latent_dim]
        # Plot first few latent dimensions
        for i in range(min(5, z_trajectory.shape[1])):
            axes[1, 1].plot(time_steps, z_trajectory[:, i], label=f'z[{i}]', alpha=0.7)
        axes[1, 1].set_title('Latent Space Evolution')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Latent Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def main():
    """
    Main function demonstrating the Neural ODE Event Pose Prediction system
    """
    print("Neural ODE Event Pose Prediction System")
    print("=" * 50)
    
    # Model parameters
    latent_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = NeuralODEEventPosePredictor(
        latent_dim=latent_dim,
        ode_hidden_dim=512,
        use_stereo=True,
        rtol=1e-3,
        atol=1e-4
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example with synthetic data (replace with your actual dataset)
    batch_size = 4
    n_events = 1024
    num_time_steps = 10
    
    # Create synthetic event data
    left_events = torch.randn(batch_size, n_events, 4)
    right_events = torch.randn(batch_size, n_events, 4)
    
    # Add padding (your format: x=-1, y=-1, t=-1, p=0)
    left_events[:, -100:, :] = torch.tensor([-1, -1, -1, 0])
    right_events[:, -150:, :] = torch.tensor([-1, -1, -1, 0])
    
    # Create time sequence
    time_steps = create_time_sequence(duration=1.0, num_steps=num_time_steps)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(left_events, time_steps, right_events)
        
        print(f"\nForward Pass Results:")
        print(f"  Initial latent z0 shape: {outputs['z0'].shape}")
        print(f"  Latent trajectory shape: {outputs['z_trajectory'].shape}")
        print(f"  Predicted poses shape: {outputs['poses'].shape}")
        
        # Print sample predictions
        sample_poses = outputs['poses'][0]  # First batch element
        print(f"\nSample pose trajectory:")
        for t, pose in enumerate(sample_poses[:5]):  # First 5 time steps
            pos = pose[:3].numpy()
            quat = pose[3:].numpy()
            print(f"  t={t}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                  f"quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
    
    print("\n" + "=" * 50)
    print("System ready for training with your event dataset!")
    print("To use with your data:")
    print("1. Load your EventExtractionDataset")
    print("2. Create DataLoader with custom_collate_fn")
    print("3. Call train_neural_ode_model(model, dataloader)")
    
    # Uncomment below to train with your actual dataset
    """
    # Load your dataset
    dataset_root = "/path/to/your/dataset"
    dataset = event_extractor(dataset_root, N=1024)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    
    # Train the model with saving
    training_history = train_neural_ode_model(
        model=model,
        dataloader=dataloader,
        num_epochs=50,
        lr=1e-3,
        device=device,
        save_dir="./neural_ode_checkpoints",
        save_every=5,  # Save every 5 epochs
        save_best=True  # Save best model
    )
    
    # Plot training curves
    plot_training_history(training_history, save_path="./training_history.png")
    
    # Save final model for inference
    save_model_for_inference(model, "./neural_ode_final_model.pth")
    
    # Example: Load model for inference later
    # loaded_model = load_model_for_inference("./neural_ode_final_model.pth", device)
    
    # Visualize results
    sample_batch = next(iter(dataloader))
    visualize_trajectory_prediction(model, sample_batch, device)
    """

def resume_training_example():
    """
    Example of how to resume training from a checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_data = load_checkpoint("./checkpoints/best_model_20240101_120000.pth", device)
    
    model = checkpoint_data['model']
    optimizer = checkpoint_data['optimizer'] 
    scheduler = checkpoint_data['scheduler']
    start_epoch = checkpoint_data['epoch']
    
    print(f"Resuming training from epoch {start_epoch}")
    
    # Continue training...
    # train_neural_ode_model(model, dataloader, num_epochs=100, ...)

if __name__ == "__main__":
    main()