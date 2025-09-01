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

def farthest_point_sample(xyz: torch.Tensor, npoint: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Farthest Point Sampling with mask support
    Args:
        xyz: [B, N, 3] input points
        npoint: number of points to sample
        mask: [B, N] mask where True=valid point, False=padding
    Returns:
        centroids: [B, npoint] indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    if mask is not None:
        # Set distance of padded points to -1 so they're never selected
        distance[~mask] = -1.0
        # Select initial point from valid points only
        valid_indices = mask.nonzero(as_tuple=False)
        if valid_indices.numel() == 0:
            # All points are padded, just select first point for each batch
            farthest = torch.zeros(B, dtype=torch.long).to(device)
        else:
            batch_valid = {}
            for idx in valid_indices:
                b, n = idx[0].item(), idx[1].item()
                if b not in batch_valid:
                    batch_valid[b] = []
                batch_valid[b].append(n)
            
            farthest = torch.zeros(B, dtype=torch.long).to(device)
            for b in range(B):
                if b in range(len(batch_valid)) and b in batch_valid:
                    farthest[b] = torch.tensor(batch_valid[b][torch.randint(0, len(batch_valid[b]), (1,)).item()]).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        if mask is not None:
            # Only update distances for valid points
            valid_mask = mask & (dist < distance)
            distance[valid_mask] = dist[valid_mask]
            # Set padded points distance to -1
            distance[~mask] = -1.0
        else:
            mask_update = dist < distance
            distance[mask_update] = dist[mask_update]
        
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Ball Query with mask support
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: [B, N, 3] all points
        new_xyz: [B, S, 3] query points
        mask: [B, N] mask where True=valid point, False=padding
    Returns:
        group_idx: [B, S, nsample] indices of grouped points
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    
    # Mask out padded points and points outside radius
    if mask is not None:
        invalid_mask = ~mask.unsqueeze(1).repeat(1, S, 1)  # [B, S, N]
        sqrdists[invalid_mask] = float('inf')
    
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask_invalid = group_idx == N
    group_idx[mask_invalid] = group_first[mask_invalid]
    
    return group_idx

def sample_and_group(npoint: int, radius: float, nsample: int, 
                    xyz: torch.Tensor, points: Optional[torch.Tensor] = None,
                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FIXED: Sampling and Grouping with proper memory contiguity handling
    Args:
        npoint: number of points to sample
        radius: grouping radius
        nsample: number of samples per group
        xyz: [B, N, 3] coordinates
        points: [B, N, D] features
        mask: [B, N] valid point mask
    Returns:
        new_xyz: [B, npoint, 3] sampled coordinates  
        new_points: [B, npoint, nsample, 3+D] grouped features
    """
    B, N, C = xyz.shape
    S = npoint
    
    # Debug information
    print(f"🔍 sample_and_group debug:")
    print(f"   xyz shape: {xyz.shape}")
    print(f"   points shape: {points.shape if points is not None else None}")
    print(f"   mask shape: {mask.shape if mask is not None else None}")
    print(f"   npoint: {npoint}, radius: {radius}, nsample: {nsample}")
    
    # Sample points using FPS
    fps_idx = farthest_point_sample(xyz, npoint, mask)  # [B, npoint]
    print(f"   fps_idx shape: {fps_idx.shape}")
    
    # Gather sampled points - FIXED: proper indexing
    new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, npoint, 3]
    print(f"   new_xyz shape: {new_xyz.shape}")
    
    # Find neighbors for each sampled point
    idx = query_ball_point(radius, nsample, xyz, new_xyz, mask)  # [B, npoint, nsample]
    print(f"   idx shape: {idx.shape}")
    
    # FIXED: Group coordinates with proper memory handling
    # Create new tensor instead of using view/reshape on potentially non-contiguous tensor
    grouped_xyz = torch.zeros(B, S, nsample, 3, device=xyz.device, dtype=xyz.dtype)
    for b in range(B):
        for s in range(S):
            for n in range(nsample):
                point_idx = idx[b, s, n]
                if point_idx < N:  # Valid index
                    grouped_xyz[b, s, n] = xyz[b, point_idx]
    
    print(f"   grouped_xyz shape: {grouped_xyz.shape}")
    
    # Normalize by subtracting centroid
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        D = points.shape[-1]
        print(f"   points feature dim: {D}")
        
        # FIXED: Group point features with proper memory handling
        grouped_points = torch.zeros(B, S, nsample, D, device=points.device, dtype=points.dtype)
        for b in range(B):
            for s in range(S):
                for n in range(nsample):
                    point_idx = idx[b, s, n]
                    if point_idx < N:  # Valid index
                        grouped_points[b, s, n] = points[b, point_idx]
        
        print(f"   grouped_points shape: {grouped_points.shape}")
        
        # Concatenate coordinates and features
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm  # [B, npoint, nsample, 3]
    
    print(f"   final new_points shape: {new_points.shape}")
    
    return new_xyz, new_points

def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor] = None,
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group all points with mask support
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    
    # Apply mask to filter out padding
    if mask is not None:
        # Create mask for grouped points
        point_mask = mask.view(B, 1, N, 1)  # [B, 1, N, 1]
        # Set padded points to zero
        new_points = new_points * point_mask.float()
    
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
    
    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIXED: Proper tensor dimension handling with memory contiguity
        Args:
            xyz: [B, 3, N] input points position
            points: [B, D, N] input points features
            mask: [B, N] mask where True=valid, False=padding
        Returns:
            new_xyz: [B, 3, S] sampled points position
            new_points: [B, D', S] output points features
        """
        print(f"🔍 PointNetSetAbstraction forward:")
        print(f"   Input xyz shape: {xyz.shape}")
        print(f"   Input points shape: {points.shape if points is not None else None}")
        print(f"   Input mask shape: {mask.shape if mask is not None else None}")
        
        # Convert from [B, C, N] to [B, N, C] for processing
        xyz = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3] - ensure contiguous
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()  # [B, N, D] - ensure contiguous
        
        print(f"   After permute - xyz: {xyz.shape}, points: {points.shape if points is not None else None}")
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, mask)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, mask)
        
        print(f"   After sampling - new_xyz: {new_xyz.shape}, new_points: {new_points.shape}")
        
        # new_points: [B, npoint, nsample, 3+D] -> [B, 3+D, nsample, npoint]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # Ensure contiguous after permute
        print(f"   After permute for conv: {new_points.shape}")
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            print(f"   After conv{i}: {new_points.shape}")
        
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()  # [B, 3, npoint] - ensure contiguous
        
        print(f"   Final output - new_xyz: {new_xyz.shape}, new_points: {new_points.shape}")
        
        return new_xyz, new_points

class EventPointNet(nn.Module):
    """
    PointNet++ for processing stereo event data chunks with padding mask support
    """
    def __init__(self, latent_dim: int = 256, use_stereo: bool = True):
        super(EventPointNet, self).__init__()
        self.use_stereo = use_stereo
        self.latent_dim = latent_dim
        
        # Set Abstraction layers - FIXED: input channel calculation
        # SA1: Sample 512 points from input, radius=0.05, 32 neighbors
        # Input: 3 (xyz) + 1 (polarity) = 4 channels AFTER grouping adds xyz offset (3) 
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.05, nsample=32, 
            in_channel=3 + 1, mlp=[64, 64, 128]  # 3 (xyz offset) + 1 (polarity) = 4
        )
        
        # SA2: Sample 128 points, radius=0.1, 64 neighbors  
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.1, nsample=64,
            in_channel=3 + 128, mlp=[128, 128, 256]  # 3 (xyz offset) + 128 from SA1
        )
        
        # SA3: Global feature extraction
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=3 + 256, mlp=[256, 512, 1024], group_all=True  # 3 (xyz offset) + 256 from SA2
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
    
    def create_event_mask(self, events: torch.Tensor) -> torch.Tensor:
        """
        Create mask for valid events (non-padding)
        Your padding has x=-1, y=-1, t=-1, p=0
        Args:
            events: [B, N, 4] events tensor
        Returns:
            mask: [B, N] True for valid events, False for padding
        """
        # Valid events have x >= 0 (since your normalized coordinates are [0,1])
        mask = events[:, :, 0] >= 0  # Check x coordinate
        return mask
    
    def forward_single_camera(self, events: torch.Tensor) -> torch.Tensor:
        """
        Process events from single camera with masking
        Args:
            events: [B, N, 4] where each event is (x, y, t, polarity)
        Returns:
            features: [B, 1024] global feature vector
        """
        print(f"🔍 forward_single_camera input shape: {events.shape}")
        
        # Create mask for valid (non-padding) events
        mask = self.create_event_mask(events)  # [B, N]
        print(f"🔍 Valid events mask shape: {mask.shape}, valid count per batch: {mask.sum(dim=1)}")
        
        # Convert to PointNet++ format: [B, 3, N] for xyz, [B, 1, N] for features
        xyz = events[:, :, :3].permute(0, 2, 1).contiguous()  # [B, 3, N] (x, y, t as spatial coords)
        features = events[:, :, 3:].permute(0, 2, 1).contiguous()  # [B, 1, N] (polarity as feature)
        
        print(f"🔍 xyz shape: {xyz.shape}, features shape: {features.shape}")
        
        # Set Abstraction layers with mask
        l1_xyz, l1_points = self.sa1(xyz, features, mask)  # [B, 3, 512], [B, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 128], [B, 256, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 1], [B, 1024, 1]
        
        # Global feature
        global_feature = l3_points.squeeze(-1)  # [B, 1024]
        print(f"🔍 Final global feature shape: {global_feature.shape}")
        
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

def structured_array_to_tensor(events_structured: np.ndarray) -> torch.Tensor:
    """
    Convert your dataset's structured array to tensor format
    Args:
        events_structured: numpy structured array with fields ['x', 'y', 't', 'p']
    Returns:
        tensor: [N, 4] tensor ready for PointNet++
    """
    if len(events_structured) == 0:
        return torch.zeros(0, 4, dtype=torch.float32)
    
    # Extract fields and stack
    x = events_structured['x'].astype(np.float32)
    y = events_structured['y'].astype(np.float32) 
    t = events_structured['t'].astype(np.float32)
    p = events_structured['p'].astype(np.float32)
    
    events_tensor = np.stack([x, y, t, p], axis=1)  # [N, 4]
    return torch.from_numpy(events_tensor)

def custom_collate_fn(batch):
    """
    Custom collate function for your dataset with proper padding
    Args:
        batch: list of samples from EventExtractionDataset
    Returns:
        batched data ready for EventPointNet with consistent tensor sizes
    """
    left_events = []
    right_events = []
    left_poses = []
    right_poses = []
    
    # Find max sequence length for padding
    max_left_len = 0
    max_right_len = 0
    
    for sample in batch:
        left_tensor = structured_array_to_tensor(sample['left_events_strip'])
        right_tensor = structured_array_to_tensor(sample['right_events_strip'])
        max_left_len = max(max_left_len, left_tensor.shape[0])
        max_right_len = max(max_right_len, right_tensor.shape[0])
    
    # Pad sequences to max length
    for sample in batch:
        # Convert structured arrays to tensors
        left_tensor = structured_array_to_tensor(sample['left_events_strip'])
        right_tensor = structured_array_to_tensor(sample['right_events_strip'])
        
        # Pad with your padding format: x=-1, y=-1, t=-1, p=0
        if left_tensor.shape[0] < max_left_len:
            pad_size = max_left_len - left_tensor.shape[0]
            padding = torch.full((pad_size, 4), -1.0)
            padding[:, 3] = 0.0  # Set polarity to 0 for padding
            left_tensor = torch.cat([left_tensor, padding], dim=0)
        
        if right_tensor.shape[0] < max_right_len:
            pad_size = max_right_len - right_tensor.shape[0]
            padding = torch.full((pad_size, 4), -1.0)
            padding[:, 3] = 0.0  # Set polarity to 0 for padding  
            right_tensor = torch.cat([right_tensor, padding], dim=0)
        
        left_events.append(left_tensor)
        right_events.append(right_tensor)
        left_poses.append(torch.from_numpy(sample['left_pose']).float())
        right_poses.append(torch.from_numpy(sample['right_pose']).float())
    
    return {
        'left_events': torch.stack(left_events),  # [B, max_left_len, 4]
        'right_events': torch.stack(right_events),  # [B, max_right_len, 4]
        'left_poses': torch.stack(left_poses),  # [B, 7]
        'right_poses': torch.stack(right_poses),  # [B, 7]
    }

# Example usage and testing
def main():
    print("🧪 Testing Fixed EventPointNet...")
    
    # Initialize model
    model = EventPointNet(latent_dim=256, use_stereo=True)
    
    # Create test data with padding
    batch_size, n_events = 2, 100  # Small for testing
    left_events = torch.randn(batch_size, n_events, 4)
    right_events = torch.randn(batch_size, n_events, 4)
    
    # Normalize coordinates to [0,1] range for valid events
    left_events[:, :, 0] = torch.sigmoid(left_events[:, :, 0])  # x
    left_events[:, :, 1] = torch.sigmoid(left_events[:, :, 1])  # y
    left_events[:, :, 2] = torch.sigmoid(left_events[:, :, 2])  # t
    left_events[:, :, 3] = torch.sign(left_events[:, :, 3])     # polarity (-1 or 1)
    
    right_events[:, :, 0] = torch.sigmoid(right_events[:, :, 0])
    right_events[:, :, 1] = torch.sigmoid(right_events[:, :, 1]) 
    right_events[:, :, 2] = torch.sigmoid(right_events[:, :, 2])
    right_events[:, :, 3] = torch.sign(right_events[:, :, 3])
    
    # Add some padding (your format: x=-1, y=-1, t=-1, p=0)
    left_events[:, -20:, :] = torch.tensor([-1, -1, -1, 0])  # Last 20 events are padding
    right_events[:, -30:, :] = torch.tensor([-1, -1, -1, 0])  # Last 30 events are padding
    
    print(f"Input shapes: left_events={left_events.shape}, right_events={right_events.shape}")
    
    try:
        with torch.no_grad():
            latent = model(left_events, right_events)
            print(f"✅ Success! Output latent shape: {latent.shape}")  # Should be [2, 256]
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()