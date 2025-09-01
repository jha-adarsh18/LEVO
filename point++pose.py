import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import math

from loadevents import EventExtractionDataset

# Proper PointNet++ implementation with correct FPS and grouping
def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Iterative farthest point sampling
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
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


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, nsample]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
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

    def forward(self, xyz, points, mask=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            mask: input mask, [B, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sampled points data, [B, D', S]
            new_mask: sampled mask, [B, S]
        """
        # xyz should be [B, N, 3] for processing
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        
        if self.group_all:
            new_xyz, new_points, new_mask = self.sample_and_group_all(xyz, points, mask)
        else:
            new_xyz, new_points, new_mask = self.sample_and_group(xyz, points, mask)
        
        # new_points: [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = torch.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 3)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, npoint]
        return new_xyz, new_points, new_mask

    def sample_and_group(self, xyz, points, mask):
        """
        Input:
            xyz: [B, N, 3]
            points: [B, D, N]
            mask: [B, N]
        """
        B, N, C = xyz.shape
        
        # Apply mask to xyz for sampling - set masked points to very far
        if mask is not None:
            xyz_masked = xyz.clone()
            xyz_masked[~mask] = 1e6  # Set masked points very far away
        else:
            xyz_masked = xyz
        
        # Sample points using FPS
        fps_idx = farthest_point_sample(xyz_masked, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        
        # Create mask for sampled points
        if mask is not None:
            new_mask = index_points(mask.unsqueeze(-1).float(), fps_idx).squeeze(-1) > 0.5  # [B, npoint]
        else:
            new_mask = torch.ones(B, self.npoint, dtype=torch.bool, device=xyz.device)
        
        # Group points around centroids
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # [B, npoint, nsample]
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)  # [B, npoint, nsample, 3]
        
        if points is not None:
            grouped_points = index_points(points.permute(0, 2, 1), idx)  # [B, npoint, nsample, D]
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm  # [B, npoint, nsample, 3]
        
        new_points = new_points.permute(0, 3, 1, 2)  # [B, C+D, npoint, nsample]
        
        return new_xyz, new_points, new_mask

    def sample_and_group_all(self, xyz, points, mask):
        """
        Group all points together
        """
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        
        if mask is not None:
            new_mask = torch.any(mask, dim=1, keepdim=True)  # [B, 1]
        else:
            new_mask = torch.ones(B, 1, dtype=torch.bool, device=xyz.device)
        
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            grouped_points = points.permute(0, 2, 1).view(B, 1, N, -1)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
            
        new_points = new_points.permute(0, 3, 1, 2)  # [B, C+D, 1, N]
        
        return new_xyz, new_points, new_mask


class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim=4, output_dim=1024):
        super(PointNetPlusPlus, self).__init__()
        self.input_dim = input_dim
        
        # Set abstraction layers with proper parameters for event data
        # SA1: Sample 1024 points from input, group with radius 0.05 in normalized coordinates
        # Fixed: input_dim - 3 should be the feature dimension (polarity = 1 dimension)
        self.sa1 = PointNetSetAbstraction(1024, 0.05, 32, 1 + 3, [64, 64, 128])  # 3 (relative xyz) + 1 (polarity)
        # SA2: Sample 256 points, group with radius 0.1
        self.sa2 = PointNetSetAbstraction(256, 0.1, 64, 128 + 3, [128, 128, 256])
        # SA3: Sample 64 points, group with radius 0.2
        self.sa3 = PointNetSetAbstraction(64, 0.2, 128, 256 + 3, [256, 512, 1024])
        # SA4: Global aggregation
        self.sa4 = PointNetSetAbstraction(None, None, None, 1024 + 3, [512, output_dim], group_all=True)
        
    def forward(self, xyz, mask=None):
        # xyz: [B, N, 4] - [x, y, t, polarity]
        B, N, _ = xyz.shape
        
        # Split coordinates and features
        points_xyz = xyz[:, :, :3]  # [B, N, 3] - [x, y, t]
        points_features = xyz[:, :, 3:].transpose(2, 1)  # [B, 1, N] - polarity
        points_xyz = points_xyz.transpose(2, 1)  # [B, 3, N]
        
        # Set abstraction layers
        l1_xyz, l1_points, l1_mask = self.sa1(points_xyz, points_features, mask)
        l2_xyz, l2_points, l2_mask = self.sa2(l1_xyz, l1_points, l1_mask)
        l3_xyz, l3_points, l3_mask = self.sa3(l2_xyz, l2_points, l2_mask)
        l4_xyz, l4_points, l4_mask = self.sa4(l3_xyz, l3_points, l3_mask)
        
        # Global feature - should be [B, output_dim, 1]
        global_feature = l4_points.squeeze(-1)  # [B, output_dim]
        
        return global_feature

class PoseRegressionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, dropout=0.3):
        super(PoseRegressionHead, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Translation head
        self.translation_head = nn.Linear(hidden_dim // 2, 3)
        
        # Rotation head
        self.rotation_head = nn.Linear(hidden_dim // 2, 4)
        
    def forward(self, features):
        shared_features = self.shared(features)
        
        translation = self.translation_head(shared_features)
        rotation_raw = self.rotation_head(shared_features)
        
        # Normalize quaternion to unit length
        rotation = nn.functional.normalize(rotation_raw, p=2, dim=1)
        
        return translation, rotation

class EventPoseModel(nn.Module):
    def __init__(self, input_dim=4, feature_dim=1024):
        super(EventPoseModel, self).__init__()
        self.encoder = PointNetPlusPlus(input_dim, feature_dim)
        self.pose_head = PoseRegressionHead(feature_dim)
        
    def forward(self, events, mask=None):
        features = self.encoder(events, mask)
        translation, rotation = self.pose_head(features)
        return translation, rotation

def quaternion_loss(q_pred, q_gt):
    """
    Quaternion loss using inner product (accounts for double cover)
    """
    # Ensure both quaternions are normalized
    q_pred = nn.functional.normalize(q_pred, p=2, dim=1)
    q_gt = nn.functional.normalize(q_gt, p=2, dim=1)
    
    inner_product = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    loss = 1.0 - inner_product
    return torch.mean(loss)

def translation_loss(t_pred, t_gt):
    """
    L2 loss for translation
    """
    return nn.functional.mse_loss(t_pred, t_gt)

class PoseLoss(nn.Module):
    def __init__(self, lambda_t=1.0, lambda_r=10.0):  # Higher weight for rotation
        super(PoseLoss, self).__init__()
        self.lambda_t = lambda_t
        self.lambda_r = lambda_r
        
    def forward(self, t_pred, q_pred, t_gt, q_gt):
        loss_t = translation_loss(t_pred, t_gt)
        loss_r = quaternion_loss(q_pred, q_gt)
        total_loss = self.lambda_t * loss_t + self.lambda_r * loss_r
        return total_loss, loss_t, loss_r

def collate_fn(batch):
    """
    Collate function that handles fixed-size events
    """
    left_events = []
    right_events = []
    left_masks = []
    right_masks = []
    left_poses = []
    right_poses = []
    n_left_events = []
    n_right_events = []
    
    for sample in batch:
        left_events.append(torch.FloatTensor(sample['left_events']))
        right_events.append(torch.FloatTensor(sample['right_events']))
        left_masks.append(torch.BoolTensor(sample['left_mask']))
        right_masks.append(torch.BoolTensor(sample['right_mask']))
        left_poses.append(torch.FloatTensor(sample['left_pose']))
        right_poses.append(torch.FloatTensor(sample['right_pose']))
        n_left_events.append(sample['n_left_events'])
        n_right_events.append(sample['n_right_events'])
    
    return {
        'left_events': torch.stack(left_events),       # (B, N, 4)
        'right_events': torch.stack(right_events),     # (B, N, 4)
        'left_masks': torch.stack(left_masks),          # (B, N)
        'right_masks': torch.stack(right_masks),        # (B, N)
        'left_poses': torch.stack(left_poses),          # (B, 7)
        'right_poses': torch.stack(right_poses),        # (B, 7)
        'n_left_events': n_left_events,
        'n_right_events': n_right_events
    }

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_t_loss = 0.0
    total_r_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        left_events = batch['left_events'].to(device)
        right_events = batch['right_events'].to(device)
        left_masks = batch['left_masks'].to(device)
        right_masks = batch['right_masks'].to(device)
        left_poses = batch['left_poses'].to(device)
        right_poses = batch['right_poses'].to(device)
        
        # Split poses into translation and rotation
        left_t_gt = left_poses[:, :3]
        left_q_gt = left_poses[:, 3:]
        right_t_gt = right_poses[:, :3]
        right_q_gt = right_poses[:, 3:]
        
        optimizer.zero_grad()
        
        # Forward pass for left and right cameras
        left_t_pred, left_q_pred = model(left_events, left_masks)
        right_t_pred, right_q_pred = model(right_events, right_masks)
        
        # Compute losses
        left_loss, left_t_loss, left_r_loss = criterion(
            left_t_pred, left_q_pred, left_t_gt, left_q_gt
        )
        right_loss, right_t_loss, right_r_loss = criterion(
            right_t_pred, right_q_pred, right_t_gt, right_q_gt
        )
        
        # Total loss
        loss = (left_loss + right_loss) / 2.0
        t_loss = (left_t_loss + right_t_loss) / 2.0
        r_loss = (left_r_loss + right_r_loss) / 2.0
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step per batch for cosine schedule
        
        # Update statistics
        total_loss += loss.item()
        total_t_loss += t_loss.item()
        total_r_loss += r_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.6f}",
            'T_Loss': f"{t_loss.item():.6f}",
            'R_Loss': f"{r_loss.item():.6f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.8f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_t_loss = total_t_loss / len(dataloader)
    avg_r_loss = total_r_loss / len(dataloader)
    
    return avg_loss, avg_t_loss, avg_r_loss


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_t_loss = 0.0
    total_r_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            left_events = batch['left_events'].to(device)
            right_events = batch['right_events'].to(device)
            left_masks = batch['left_masks'].to(device)
            right_masks = batch['right_masks'].to(device)
            left_poses = batch['left_poses'].to(device)
            right_poses = batch['right_poses'].to(device)
            
            left_t_gt = left_poses[:, :3]
            left_q_gt = left_poses[:, 3:]
            right_t_gt = right_poses[:, :3]
            right_q_gt = right_poses[:, 3:]
            
            left_t_pred, left_q_pred = model(left_events, left_masks)
            right_t_pred, right_q_pred = model(right_events, right_masks)
            
            left_loss, left_t_loss, left_r_loss = criterion(
                left_t_pred, left_q_pred, left_t_gt, left_q_gt
            )
            right_loss, right_t_loss, right_r_loss = criterion(
                right_t_pred, right_q_pred, right_t_gt, right_q_gt
            )
            
            loss = (left_loss + right_loss) / 2.0
            t_loss = (left_t_loss + right_t_loss) / 2.0
            r_loss = (left_r_loss + right_r_loss) / 2.0
            
            total_loss += loss.item()
            total_t_loss += t_loss.item()
            total_r_loss += r_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_t_loss = total_t_loss / len(dataloader)
    avg_r_loss = total_r_loss / len(dataloader)
    
    return avg_loss, avg_t_loss, avg_r_loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model():
    # Configuration
    config = {
        'dataset_root': "//media/adarsh/One Touch/EventSLAM/dataset/train",
        'val_dataset_root': "//media/adarsh/One Touch/EventSLAM/dataset/val",
        'batch_size': 8,  # Reduced for PointNet++
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lambda_t': 1.0,
        'lambda_r': 10.0,  # Higher weight for rotation
        'save_dir': './checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'early_stopping_patience': 15,
        'window_duration': 5000,  # 5ms windows
        'stride': 2500,  # 2.5ms stride
        'max_events_per_strip': 2048,  # Fixed number of events
        'warmup_steps': 1000
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize dataset and dataloaders
    print("Loading datasets...")
        
    train_dataset = EventExtractionDataset(
        config['dataset_root'], 
        window_duration=config['window_duration'],
        stride=config['stride'],
        max_events_per_strip=config['max_events_per_strip']
    )
    val_dataset = EventExtractionDataset(
        config['val_dataset_root'],
        window_duration=config['window_duration'],
        stride=config['stride'],
        max_events_per_strip=config['max_events_per_strip']
    )
    
    print(f"Training windows: {len(train_dataset)}")
    print(f"Validation windows: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing PointNet++ model...")
    device = torch.device(config['device'])
    model = EventPoseModel().to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = PoseLoss(config['lambda_t'], config['lambda_r'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler - step per batch
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        config['warmup_steps'], 
        total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training with PointNet++...")
    print(f"Total training steps: {total_steps}")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Training
        train_loss, train_t_loss, train_r_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validation  
        val_loss, val_t_loss, val_r_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.6f} (T: {train_t_loss:.6f}, R: {train_r_loss:.6f})")
        print(f"Val Loss: {val_loss:.6f} (T: {val_t_loss:.6f}, R: {val_r_loss:.6f})")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training completed!")


if __name__ == "__main__":
    # Add some sanity checks before training
    print("Running sanity checks...")
    
    # Test dataset
    dataset_root = r"//media/adarsh/One Touch/EventSLAM/dataset/train"
    if os.path.exists(dataset_root):
        test_dataset = EventExtractionDataset(
            dataset_root, 
            window_duration=5000, 
            stride=2500, 
            max_events_per_strip=2048
        )
        print(f"Dataset loaded successfully with {len(test_dataset)} windows")
        
        # Test a batch
        test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(test_loader))
        print(f"Batch shapes:")
        print(f"  Left events: {batch['left_events'].shape}")
        print(f"  Left masks: {batch['left_masks'].shape}")
        print(f"  Left poses: {batch['left_poses'].shape}")
        print(f"  Event counts: {batch['n_left_events']}")
        
        # Test model forward pass
        model = EventPoseModel()
        model.eval()
        with torch.no_grad():
            t_pred, q_pred = model(batch['left_events'], batch['left_masks'])
            print(f"Model output shapes: t={t_pred.shape}, q={q_pred.shape}")
        
        print("All sanity checks passed!")
        
    else:
        print(f"Dataset path {dataset_root} not found. Please update the path.")
    
    # Start training
    train_model()