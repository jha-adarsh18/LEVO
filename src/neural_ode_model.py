import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import math

class EventEncoder(nn.Module):
    """
    Encodes events within a 5ms slice into a fixed-size representation
    """
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=64):
        super(EventEncoder, self).__init__()
        self.input_dim = input_dim  # [x, y, t_rel, polarity]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Event embedding layers
        self.event_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for aggregating events in a slice
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Final aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, events_batch):
        """
        Args:
            events_batch: List of event tensors, each of shape (N_events, 4)
        Returns:
            Aggregated event features of shape (batch_size, output_dim)
        """
        batch_size = len(events_batch)
        device = next(self.parameters()).device
        
        aggregated_features = []
        
        for events in events_batch:
            events = events.to(device)
            
            if len(events) == 0:
                # No events in this slice - use zero embedding
                agg_feature = torch.zeros(self.output_dim, device=device)
            else:
                # Embed individual events
                event_features = self.event_embed(events)  # (N_events, output_dim)
                
                if len(event_features) == 1:
                    # Single event
                    agg_feature = event_features.squeeze(0)
                else:
                    # Multiple events - use attention for aggregation
                    event_features = event_features.unsqueeze(0)  # (1, N_events, output_dim)
                    
                    # Self-attention over events
                    attended_features, _ = self.attention(
                        event_features, event_features, event_features
                    )
                    
                    # Global max pooling
                    agg_feature = torch.max(attended_features.squeeze(0), dim=0)[0]
                
                # Final processing
                agg_feature = self.aggregator(agg_feature)
            
            aggregated_features.append(agg_feature)
        
        return torch.stack(aggregated_features)  # (batch_size, output_dim)


class ODEFunc(nn.Module):
    """
    Neural ODE function: dh/dt = f(h, t; θ)
    """
    def __init__(self, hidden_dim=256, time_embed_dim=32):
        super(ODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding using sinusoidal encoding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Tanh(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # ODE dynamics network
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + time_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize with small weights for stability
        for layer in self.dynamics:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, t, h):
        """
        Args:
            t: Time tensor (scalar or batch)
            h: Hidden state tensor (batch_size, hidden_dim)
        Returns:
            dh/dt: Time derivative of hidden state
        """
        batch_size = h.shape[0]
        
        # Handle scalar time
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Time embedding
        t_embed = self.time_embedding(t)  # (batch_size, time_embed_dim)
        
        # Concatenate hidden state and time embedding
        h_t = torch.cat([h, t_embed], dim=1)  # (batch_size, hidden_dim + time_embed_dim)
        
        # Compute dynamics
        dhdt = self.dynamics(h_t)
        
        return dhdt


class SimpleODESolver(nn.Module):
    """
    Simple Euler method ODE solver for integration over 5ms intervals
    """
    def __init__(self, ode_func, num_steps=5):
        super(SimpleODESolver, self).__init__()
        self.ode_func = ode_func
        self.num_steps = num_steps
    
    def forward(self, h0, t_start, t_end):
        """
        Integrate from t_start to t_end starting with h0
        """
        dt = (t_end - t_start) / self.num_steps
        h = h0
        
        for step in range(self.num_steps):
            t = t_start + step * dt
            dhdt = self.ode_func(t, h)
            h = h + dt * dhdt
        
        return h


class EventDrivenODECell(nn.Module):
    """
    Single ODE cell that:
    1. Integrates ODE over 5ms interval
    2. Applies event-driven update at slice boundary
    """
    def __init__(self, hidden_dim=256, event_dim=64):
        super(EventDrivenODECell, self).__init__()
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim
        
        # ODE dynamics
        self.ode_func = ODEFunc(hidden_dim)
        self.ode_solver = SimpleODESolver(self.ode_func)
        
        # Event-driven update network
        self.event_update = nn.Sequential(
            nn.Linear(hidden_dim + event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gating mechanism for event influence
        self.event_gate = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, h_prev, event_features, t_start, t_end):
        """
        Args:
            h_prev: Previous hidden state (batch_size, hidden_dim)
            event_features: Event features for this slice (batch_size, event_dim) 
            t_start: Start time of slice
            t_end: End time of slice
        Returns:
            h_next: Next hidden state after ODE integration + event update
        """
        device = h_prev.device
        
        # Convert times to tensors
        if not isinstance(t_start, torch.Tensor):
            t_start = torch.tensor(t_start, device=device, dtype=torch.float32)
        if not isinstance(t_end, torch.Tensor):
            t_end = torch.tensor(t_end, device=device, dtype=torch.float32)
        
        # 1. ODE integration over the 5ms interval
        h_ode = self.ode_solver(h_prev, t_start, t_end)
        
        # 2. Event-driven update at slice boundary
        # Concatenate ODE-evolved state with event features
        h_event_input = torch.cat([h_ode, event_features], dim=1)
        
        # Compute event-driven update
        event_update = self.event_update(h_event_input)
        
        # Apply gating based on event features
        gate = self.event_gate(event_features)
        
        # Final hidden state: ODE evolution + gated event update
        h_next = h_ode + gate * event_update
        
        return h_next


class NeuralODEEventSLAM(nn.Module):
    """
    Complete Neural ODE model for event-based SLAM
    """
    def __init__(self, hidden_dim=256, event_dim=64, pose_dim=7):
        super(NeuralODEEventSLAM, self).__init__()
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim
        self.pose_dim = pose_dim  # [tx, ty, tz, qx, qy, qz, qw]
        
        # Event encoder for 5ms slices
        self.event_encoder = EventEncoder(
            input_dim=4,  # [x, y, t_rel, polarity]
            hidden_dim=128,
            output_dim=event_dim
        )
        
        # ODE cell for temporal dynamics
        self.ode_cell = EventDrivenODECell(hidden_dim, event_dim)
        
        # Initial state network
        self.initial_state = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        
        # Pose regression head
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, pose_dim)
        )
        
        # Separate translation and rotation heads for better training
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)
        )
    
    def forward(self, left_slices, right_slices, slice_duration=5):
        """
        Args:
            left_slices: List of left camera event slices (batch_size, num_slices)
            right_slices: List of right camera event slices (batch_size, num_slices)
            slice_duration: Duration of each slice in ms
        Returns:
            poses: Predicted poses for each slice (batch_size, num_slices, 7)
            hidden_states: Hidden states for each slice (batch_size, num_slices, hidden_dim)
        """
        batch_size = len(left_slices)
        num_slices = len(left_slices[0])
        device = next(self.parameters()).device
        
        # Initialize hidden state
        h = self.initial_state.unsqueeze(0).expand(batch_size, -1)  # (batch_size, hidden_dim)
        
        poses = []
        hidden_states = []
        
        for slice_idx in range(num_slices):
            # Extract events for current slice
            left_events = [left_slices[b][slice_idx]['events'] for b in range(batch_size)]
            right_events = [right_slices[b][slice_idx]['events'] for b in range(batch_size)]
            
            # Encode events
            left_features = self.event_encoder(left_events)   # (batch_size, event_dim)
            right_features = self.event_encoder(right_events) # (batch_size, event_dim)
            
            # Combine left and right features (simple concatenation + projection)
            combined_features = torch.cat([left_features, right_features], dim=1)
            event_features = F.linear(combined_features, 
                                    weight=torch.randn(self.event_dim, self.event_dim * 2, device=device) * 0.1)
            
            # Time bounds for this slice
            t_start = slice_idx * slice_duration
            t_end = (slice_idx + 1) * slice_duration
            
            # ODE integration + event update
            h = self.ode_cell(h, event_features, t_start, t_end)
            
            # Predict pose from current hidden state
            translation = self.translation_head(h)
            rotation_raw = self.rotation_head(h) 
            rotation = F.normalize(rotation_raw, p=2, dim=1)  # Normalize quaternion
            
            pose = torch.cat([translation, rotation], dim=1)  # (batch_size, 7)
            
            poses.append(pose)
            hidden_states.append(h)
        
        # Stack outputs
        poses = torch.stack(poses, dim=1)  # (batch_size, num_slices, 7)
        hidden_states = torch.stack(hidden_states, dim=1)  # (batch_size, num_slices, hidden_dim)
        
        return poses, hidden_states
    
    def query_pose_at_time(self, hidden_states, query_times, slice_duration=5):
        """
        Query pose at arbitrary times using learned hidden states
        """
        # For simplicity, use nearest slice interpolation
        # In practice, you could integrate ODE between slices for exact timing
        batch_size, num_slices, hidden_dim = hidden_states.shape
        
        poses = []
        for t in query_times:
            slice_idx = min(int(t / slice_duration), num_slices - 1)
            h = hidden_states[:, slice_idx]  # (batch_size, hidden_dim)
            
            translation = self.translation_head(h)
            rotation_raw = self.rotation_head(h)
            rotation = F.normalize(rotation_raw, p=2, dim=1)
            
            pose = torch.cat([translation, rotation], dim=1)
            poses.append(pose)
        
        return torch.stack(poses, dim=1)  # (batch_size, len(query_times), 7)


class PoseLoss(nn.Module):
    """
    Combined loss for translation and rotation
    """
    def __init__(self, lambda_t=1.0, lambda_r=1.0):
        super(PoseLoss, self).__init__()
        self.lambda_t = lambda_t
        self.lambda_r = lambda_r
    
    def translation_loss(self, t_pred, t_gt):
        return F.mse_loss(t_pred, t_gt)
    
    def quaternion_loss(self, q_pred, q_gt):
        # Account for quaternion double cover: q and -q represent same rotation
        inner_product = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
        loss = 1.0 - inner_product
        return torch.mean(loss)
    
    def forward(self, poses_pred, poses_gt):
        """
        Args:
            poses_pred: (batch_size, num_slices, 7) predicted poses
            poses_gt: (batch_size, num_poses, 8) ground truth [t, tx, ty, tz, qx, qy, qz, qw]
        """
        batch_size, num_slices = poses_pred.shape[:2]
        
        # Interpolate ground truth poses to match prediction time points
        device = poses_pred.device
        
        total_loss = 0.0
        total_t_loss = 0.0
        total_r_loss = 0.0
        
        for b in range(batch_size):
            gt_poses = poses_gt[b]  # (num_poses, 8)
            pred_poses = poses_pred[b]  # (num_slices, 7)
            
            # Extract timestamps and poses
            gt_times = gt_poses[:, 0]  # Ground truth timestamps
            gt_translations = gt_poses[:, 1:4]  # tx, ty, tz
            gt_rotations = gt_poses[:, 4:8]    # qx, qy, qz, qw
            
            # Create query times for each slice (middle of slice)
            query_times = torch.arange(num_slices, device=device) * 5.0 + 2.5  # Middle of 5ms slices
            
            # Interpolate ground truth poses at query times
            interp_translations = []
            interp_rotations = []
            
            for t in query_times:
                # Find closest ground truth pose
                time_diffs = torch.abs(gt_times - t)
                closest_idx = torch.argmin(time_diffs)
                
                interp_translations.append(gt_translations[closest_idx])
                interp_rotations.append(gt_rotations[closest_idx])
            
            interp_translations = torch.stack(interp_translations)  # (num_slices, 3)
            interp_rotations = torch.stack(interp_rotations)        # (num_slices, 4)
            
            # Compute losses
            t_loss = self.translation_loss(pred_poses[:, :3], interp_translations)
            r_loss = self.quaternion_loss(pred_poses[:, 3:], interp_rotations)
            
            total_t_loss += t_loss
            total_r_loss += r_loss
        
        # Average over batch
        avg_t_loss = total_t_loss / batch_size
        avg_r_loss = total_r_loss / batch_size
        total_loss = self.lambda_t * avg_t_loss + self.lambda_r * avg_r_loss
        
        return total_loss, avg_t_loss, avg_r_loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0.0
    total_t_loss = 0.0
    total_r_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Extract data
        left_slices = batch['left_slices']
        right_slices = batch['right_slices']
        gt_poses = [poses.to(device) for poses in batch['poses']]
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            pred_poses, hidden_states = model(left_slices, right_slices)
            
            # Compute loss
            loss, t_loss, r_loss = criterion(pred_poses, gt_poses)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_t_loss += t_loss.item()
            total_r_loss += r_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'T_Loss': f"{t_loss.item():.6f}",
                'R_Loss': f"{r_loss.item():.6f}"
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / len(dataloader)
    avg_t_loss = total_t_loss / len(dataloader)
    avg_r_loss = total_r_loss / len(dataloader)
    
    return avg_loss, avg_t_loss, avg_r_loss


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0.0
    total_t_loss = 0.0
    total_r_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            left_slices = batch['left_slices']
            right_slices = batch['right_slices'] 
            gt_poses = [poses.to(device) for poses in batch['poses']]
            
            try:
                pred_poses, _ = model(left_slices, right_slices)
                loss, t_loss, r_loss = criterion(pred_poses, gt_poses)
                
                total_loss += loss.item()
                total_t_loss += t_loss.item()
                total_r_loss += r_loss.item()
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    avg_loss = total_loss / len(dataloader)
    avg_t_loss = total_t_loss / len(dataloader)
    avg_r_loss = total_r_loss / len(dataloader)
    
    return avg_loss, avg_t_loss, avg_r_loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Learning rate scheduler with warmup and cosine annealing"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)