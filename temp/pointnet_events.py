import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between points
    Args:
        src: [B, N, C] source points
        dst: [B, M, C] destination points
    Returns:
        dist: [B, N, M] squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    return dist

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling
    Args:
        xyz: [B, N, 3] input points
        npoint: number of points to sample
    Returns:
        centroids: [B, npoint] indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball Query
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: [B, N, 3] all points
        new_xyz: [B, S, 3] query points
    Returns:
        group_idx: [B, S, nsample] indices of grouped points
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx

def sample_and_group(npoint: int, radius: float, nsample: int, 
                    xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sampling and Grouping
    Args:
        npoint: number of points to sample
        radius: ball query radius
        nsample: number of points in each group
        xyz: [B, N, 3] input points position
        points: [B, N, D] input points features
    Returns:
        new_xyz: [B, npoint, 3] sampled points position
        new_points: [B, npoint, nsample, 3+D] grouped points
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).repeat(1, 1, 3))  # [B, npoint, 3]
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).repeat(1, 1, 1, 3))  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = torch.gather(points, 1, idx.unsqueeze(-1).repeat(1, 1, 1, points.shape[-1]))
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points

def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group all points (for global feature extraction)
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: Optional[int], radius: float, nsample: int, 
                 in_channel: int, mlp: list, group_all: bool = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, 3, N] input points position
            points: [B, D, N] input points features
        Returns:
            new_xyz: [B, 3, S] sampled points position
            new_points: [B, D', S] output points features
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # new_points: [B, npoint, nsample, 3+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, 3+D, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, npoint]
        
        return new_xyz, new_points

class EventPointNet(nn.Module):
    """
    PointNet++ for processing stereo event data chunks
    """
    def __init__(self, latent_dim: int = 256, use_stereo: bool = True):
        super(EventPointNet, self).__init__()
        self.use_stereo = use_stereo
        self.latent_dim = latent_dim
        
        # Input: (x, y, t, polarity) = 4 features per event
        # For stereo: process left and right separately then combine
        
        # Set Abstraction layers
        # SA1: Sample 512 points from input, radius=0.2, 32 neighbors
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.05, nsample=32, 
            in_channel=4, mlp=[64, 64, 128]  # 3 (xyz) + 1 (polarity) = 4 input channels
        )
        
        # SA2: Sample 128 points, radius=0.4, 64 neighbors  
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.1, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256]  # 128 from SA1 + 3 (xyz)
        )
        
        # SA3: Global feature extraction
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        
        # Final MLP for latent representation
        if use_stereo:
            # Combine left and right features
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2048, 1024),  # 1024 * 2 from left and right
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, latent_dim)
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, latent_dim)
            )
    
    def forward_single_camera(self, events: torch.Tensor) -> torch.Tensor:
        """
        Process events from single camera
        Args:
            events: [B, N, 4] where each event is (x, y, t, polarity)
        Returns:
            features: [B, 1024] global feature vector
        """
        # Convert to PointNet++ format: [B, 3, N] for xyz, [B, 1, N] for features
        xyz = events[:, :, :3].permute(0, 2, 1)  # [B, 3, N] (x, y, t as spatial coords)
        features = events[:, :, 3:].permute(0, 2, 1)  # [B, 1, N] (polarity as feature)
        
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, features)  # [B, 3, 512], [B, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 128], [B, 256, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 1], [B, 1024, 1]
        
        # Global feature
        global_feature = l3_points.squeeze(-1)  # [B, 1024]
        
        return global_feature
    
    def forward(self, left_events: torch.Tensor, right_events: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for stereo event data
        Args:
            left_events: [B, N_left, 4] left camera events
            right_events: [B, N_right, 4] right camera events (optional)
        Returns:
            latent: [B, latent_dim] latent representation
        """
        if self.use_stereo and right_events is not None:
            # Process both cameras
            left_features = self.forward_single_camera(left_events)  # [B, 1024]
            right_features = self.forward_single_camera(right_events)  # [B, 1024]
            
            # Concatenate stereo features
            combined_features = torch.cat([left_features, right_features], dim=1)  # [B, 2048]
        else:
            # Process only left camera
            combined_features = self.forward_single_camera(left_events)  # [B, 1024]
        
        # Final latent representation
        latent = self.fusion_mlp(combined_features)  # [B, latent_dim]
        
        return latent

def events_to_tensor(events_array: np.ndarray, max_events: int = 2048) -> torch.Tensor:
    """
    Convert numpy structured array to tensor format for PointNet++
    Args:
        events_array: numpy structured array with fields ['x', 'y', 't', 'p']
        max_events: maximum number of events to include (pad or truncate)
    Returns:
        tensor: [1, max_events, 4] tensor ready for PointNet++
    """
    if len(events_array) == 0:
        # Return zero tensor for empty events
        return torch.zeros(1, max_events, 4, dtype=torch.float32)
    
    # Extract fields
    x = events_array['x'].astype(np.float32)
    y = events_array['y'].astype(np.float32) 
    t = events_array['t'].astype(np.float32)
    p = events_array['p'].astype(np.float32)
    
    # Stack into tensor format
    events_tensor = np.stack([x, y, t, p], axis=1)  # [N, 4]
    
    # Handle padding/truncation
    if len(events_tensor) < max_events:
        # Pad with zeros
        padding = np.zeros((max_events - len(events_tensor), 4), dtype=np.float32)
        events_tensor = np.concatenate([events_tensor, padding], axis=0)
    elif len(events_tensor) > max_events:
        # Randomly sample max_events
        indices = np.random.choice(len(events_tensor), max_events, replace=False)
        events_tensor = events_tensor[indices]
    
    return torch.from_numpy(events_tensor).unsqueeze(0)  # [1, max_events, 4]

# Example usage function
def process_event_batch(event_data_dict: dict, model: EventPointNet, max_events: int = 2048) -> torch.Tensor:
    """
    Process a batch of event data from your dataset
    Args:
        event_data_dict: dictionary from your EventExtractionDataset
        model: trained EventPointNet model
        max_events: maximum events per camera
    Returns:
        latent: [1, latent_dim] latent representation
    """
    # Convert event arrays to tensors
    left_tensor = events_to_tensor(event_data_dict['left'], max_events)
    
    if model.use_stereo and len(event_data_dict['right']) > 0:
        right_tensor = events_to_tensor(event_data_dict['right'], max_events)
        latent = model(left_tensor, right_tensor)
    else:
        latent = model(left_tensor)
    
    return latent