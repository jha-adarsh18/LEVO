import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class ODEFunc(nn.Module):
    """
    Neural ODE function that learns latent space dynamics
    dz/dt = f(z, t) where z is the latent representation
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Neural network to learn the dynamics function
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize with small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute dz/dt at time t
        Args:
            t: [1] current time (scalar)
            z: [batch_size, latent_dim] current latent state
        Returns:
            dz_dt: [batch_size, latent_dim] time derivative
        """
        batch_size = z.shape[0]
        
        # Expand time to match batch size
        t_expanded = t.expand(batch_size, 1)  # [batch_size, 1]
        
        # Concatenate latent state with time
        zt = torch.cat([z, t_expanded], dim=1)  # [batch_size, latent_dim + 1]
        
        # Compute derivative
        dz_dt = self.net(zt)  # [batch_size, latent_dim]
        
        return dz_dt

class NeuralODESolver(nn.Module):
    """
    Neural ODE solver using Euler method (can be replaced with more sophisticated solvers)
    """
    def __init__(self, ode_func: ODEFunc, method: str = 'euler'):
        super(NeuralODESolver, self).__init__()
        self.ode_func = ode_func
        self.method = method
    
    def forward(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Solve ODE from initial condition z0 over time span t_span
        Args:
            z0: [batch_size, latent_dim] initial condition
            t_span: [num_timesteps] time points to evaluate
        Returns:
            trajectory: [batch_size, num_timesteps, latent_dim] solution trajectory
        """
        device = z0.device
        batch_size, latent_dim = z0.shape
        num_timesteps = len(t_span)
        
        # Initialize trajectory tensor
        trajectory = torch.zeros(batch_size, num_timesteps, latent_dim, device=device)
        trajectory[:, 0] = z0
        
        # Current state
        z_current = z0
        
        # Solve using Euler method
        for i in range(1, num_timesteps):
            t_current = t_span[i-1]
            dt = t_span[i] - t_span[i-1]
            
            # Compute derivative
            dz_dt = self.ode_func(t_current, z_current)
            
            # Euler step
            z_current = z_current + dt * dz_dt
            trajectory[:, i] = z_current
        
        return trajectory

class LatentODEModule(nn.Module):
    """
    Complete Neural ODE module for learning latent dynamics
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super(LatentODEModule, self).__init__()
        self.latent_dim = latent_dim
        
        # ODE function and solver
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.ode_solver = NeuralODESolver(self.ode_func)
        
        # Optional: latent space regularization
        self.latent_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, z_sequence: torch.Tensor, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Learn dynamics from a sequence of latent representations
        Args:
            z_sequence: [batch_size, seq_len, latent_dim] sequence of latent features
            time_deltas: [seq_len] time intervals between observations (in seconds)
        Returns:
            predicted_sequence: [batch_size, seq_len, latent_dim] predicted trajectory
        """
        batch_size, seq_len, latent_dim = z_sequence.shape
        device = z_sequence.device
        
        # Start from first observation
        z0 = z_sequence[:, 0]  # [batch_size, latent_dim]
        
        # Create cumulative time span
        t_span = torch.zeros(seq_len, device=device)
        t_span[1:] = torch.cumsum(time_deltas[1:], dim=0)
        
        # Solve ODE
        trajectory = self.ode_solver(z0, t_span)  # [batch_size, seq_len, latent_dim]
        
        # Optional normalization
        trajectory = self.latent_norm(trajectory)
        
        return trajectory
    
    def predict_future(self, z0: torch.Tensor, future_times: torch.Tensor) -> torch.Tensor:
        """
        Predict future latent states given initial condition
        Args:
            z0: [batch_size, latent_dim] initial latent state
            future_times: [num_future] future time points
        Returns:
            future_trajectory: [batch_size, num_future, latent_dim]
        """
        return self.ode_solver(z0, future_times)

class PoseRegressor(nn.Module):
    """
    Pose regression head that takes latent features and outputs 6DoF poses
    """
    def __init__(self, latent_dim: int, output_dim: int = 7):  # 7 for translation + quaternion
        super(PoseRegressor, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Multi-layer pose regression network
        self.pose_net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize final layer with small weights for stability
        nn.init.normal_(self.pose_net[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.pose_net[-1].bias, val=0)
    
    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        """
        Regress pose from latent features
        Args:
            latent_features: [batch_size, seq_len, latent_dim] or [batch_size, latent_dim]
        Returns:
            poses: [batch_size, seq_len, output_dim] or [batch_size, output_dim]
        """
        input_shape = latent_features.shape
        
        if len(input_shape) == 3:  # Sequence input
            batch_size, seq_len, latent_dim = input_shape
            # Reshape for processing
            features_flat = latent_features.view(-1, latent_dim)  # [batch_size * seq_len, latent_dim]
            poses_flat = self.pose_net(features_flat)  # [batch_size * seq_len, output_dim]
            poses = poses_flat.view(batch_size, seq_len, self.output_dim)  # [batch_size, seq_len, output_dim]
        else:  # Single input
            poses = self.pose_net(latent_features)  # [batch_size, output_dim]
        
        return poses

class EndToEndEventSLAM(nn.Module):
    """
    Complete end-to-end model: PointNet++ → Neural ODE → Pose Regressor
    """
    def __init__(self, pointnet_model, latent_dim: int = 256, ode_hidden_dim: int = 512):
        super(EndToEndEventSLAM, self).__init__()
        
        # Components
        self.pointnet = pointnet_model  # Pre-trained or to be trained
        self.latent_ode = LatentODEModule(latent_dim, ode_hidden_dim)
        self.pose_regressor = PoseRegressor(latent_dim, output_dim=7)
        
        self.latent_dim = latent_dim
    
    def forward(self, left_events_seq: torch.Tensor, right_events_seq: torch.Tensor, 
                time_deltas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass
        Args:
            left_events_seq: [batch_size, seq_len, max_events, 4] sequence of left events
            right_events_seq: [batch_size, seq_len, max_events, 4] sequence of right events  
            time_deltas: [seq_len] time intervals between frames
        Returns:
            predicted_poses: [batch_size, seq_len, 7] predicted pose sequence
            latent_trajectory: [batch_size, seq_len, latent_dim] learned dynamics
        """
        batch_size, seq_len = left_events_seq.shape[:2]
        device = left_events_seq.device
        
        # Extract latent features from each time step
        latent_sequence = []
        
        for t in range(seq_len):
            left_t = left_events_seq[:, t]  # [batch_size, max_events, 4]
            right_t = right_events_seq[:, t]  # [batch_size, max_events, 4] 
            
            # Extract latent features using PointNet++
            with torch.set_grad_enabled(self.training):
                latent_t = self.pointnet(left_t, right_t)  # [batch_size, latent_dim]
                latent_sequence.append(latent_t)
        
        # Stack into sequence
        latent_sequence = torch.stack(latent_sequence, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Learn dynamics with Neural ODE
        latent_trajectory = self.latent_ode(latent_sequence, time_deltas)  # [batch_size, seq_len, latent_dim]
        
        # Predict poses from trajectory
        predicted_poses = self.pose_regressor(latent_trajectory)  # [batch_size, seq_len, 7]
        
        return predicted_poses, latent_trajectory
    
    def predict_future_poses(self, left_events_seq: torch.Tensor, right_events_seq: torch.Tensor,
                           time_deltas: torch.Tensor, future_times: torch.Tensor) -> torch.Tensor:
        """
        Predict future poses given current observation sequence
        Args:
            left_events_seq: [batch_size, seq_len, max_events, 4] current observations
            right_events_seq: [batch_size, seq_len, max_events, 4] current observations
            time_deltas: [seq_len] time intervals in current sequence
            future_times: [num_future] future time points to predict
        Returns:
            future_poses: [batch_size, num_future, 7] predicted future poses
        """
        with torch.no_grad():
            # Get current trajectory
            _, latent_trajectory = self.forward(left_events_seq, right_events_seq, time_deltas)
            
            # Use last latent state as initial condition for future prediction
            z_current = latent_trajectory[:, -1]  # [batch_size, latent_dim]
            
            # Predict future latent trajectory
            future_latent = self.latent_ode.predict_future(z_current, future_times)
            
            # Convert to poses
            future_poses = self.pose_regressor(future_latent)
            
            return future_poses

def compute_pose_loss(pred_poses: torch.Tensor, true_poses: torch.Tensor, 
                     translation_weight: float = 1.0, rotation_weight: float = 1.0) -> torch.Tensor:
    """
    Compute pose loss with separate weights for translation and rotation
    Args:
        pred_poses: [batch_size, seq_len, 7] predicted poses (translation + quaternion)
        true_poses: [batch_size, seq_len, 7] ground truth poses
        translation_weight: weight for translation loss
        rotation_weight: weight for rotation loss
    Returns:
        loss: weighted pose loss
    """
    # Split translation and rotation
    pred_trans = pred_poses[..., :3]  # [batch_size, seq_len, 3]
    pred_rot = pred_poses[..., 3:]    # [batch_size, seq_len, 4]
    
    true_trans = true_poses[..., :3]  # [batch_size, seq_len, 3]
    true_rot = true_poses[..., 3:]    # [batch_size, seq_len, 4]
    
    # Translation loss (L2)
    trans_loss = F.mse_loss(pred_trans, true_trans)
    
    # Rotation loss (normalize quaternions first)
    pred_rot_norm = F.normalize(pred_rot, p=2, dim=-1)
    true_rot_norm = F.normalize(true_rot, p=2, dim=-1)
    
    # Quaternion loss (1 - |dot product|)
    dot_product = torch.sum(pred_rot_norm * true_rot_norm, dim=-1)  # [batch_size, seq_len]
    rot_loss = torch.mean(1 - torch.abs(dot_product))
    
    # Combined loss
    total_loss = translation_weight * trans_loss + rotation_weight * rot_loss
    
    return total_loss

def compute_ode_regularization(latent_trajectory: torch.Tensor, time_deltas: torch.Tensor,
                              smoothness_weight: float = 0.01) -> torch.Tensor:
    """
    Compute regularization term to encourage smooth dynamics
    """
    # Compute finite differences
    dt = time_deltas[1:].unsqueeze(0).unsqueeze(-1)  # [1, seq_len-1, 1]
    dz = latent_trajectory[:, 1:] - latent_trajectory[:, :-1]  # [batch_size, seq_len-1, latent_dim]
    
    # Velocity
    velocity = dz / (dt + 1e-8)  # [batch_size, seq_len-1, latent_dim]
    
    # Acceleration (second derivative)
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [batch_size, seq_len-2, latent_dim]
    
    # Smoothness penalty
    smoothness_loss = torch.mean(acceleration ** 2)
    
    return smoothness_weight * smoothness_loss