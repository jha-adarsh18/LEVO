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

    def forward(self, xyz, points):
        # xyz: (B, N, 3), points: (B, D, N)
        xyz = xyz.permute(0, 2, 1)  # (B, 3, N)
        #if points is not None:
            #points = points.permute(0, 2, 1)  # (B, D, N)

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(xyz, points)
        
        # new_points: (B, C_in, npoint, nsample)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = torch.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=3)[0]  # (B, C_out, npoint)
        new_xyz = new_xyz.permute(0, 2, 1)  # (B, npoint, 3)
        return new_xyz, new_points

    def sample_and_group(self, xyz, points):
        # Simplified sampling and grouping
        B, C, N = xyz.shape
        centroids = self.farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = self.index_points(xyz, centroids)  # (B, 3, npoint)
        
        # For simplicity, use k-nearest neighbors instead of radius search
        grouped_xyz, grouped_points = self.knn_group(xyz, new_xyz, points, self.nsample)
        
        return new_xyz, grouped_points

    def sample_and_group_all(self, xyz, points):
        # Group all points
        B, C, N = xyz.shape
        new_xyz = torch.zeros(B, 3, 1).to(xyz.device)
        grouped_xyz = xyz.view(B, C, 1, N)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, -1, 1, N)], dim=1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

    def farthest_point_sample(self, xyz, npoint):
        # Simplified FPS - random sampling for now
        B, C, N = xyz.shape
        centroids = torch.randint(0, N, (B, npoint), device=xyz.device)
        return centroids

    def index_points(self, points, idx):
        # points: (B, C, N), idx: (B, npoint)
        B, C, N = points.shape
        npoint = idx.shape[1]
        idx_expanded = idx.unsqueeze(1).expand(B, C, npoint)
        new_points = torch.gather(points, 2, idx_expanded)
        return new_points

    def knn_group(self, xyz, new_xyz, points, nsample):
        # Simplified grouping - just take first nsample points for each centroid
        B, C, N = xyz.shape
        npoint = new_xyz.shape[2]
        
        # Random grouping for simplicity
        group_idx = torch.randint(0, N, (B, npoint, nsample), device=xyz.device)
        
        grouped_xyz = xyz.unsqueeze(2).expand(B, C, npoint, N)
        grouped_xyz = torch.gather(grouped_xyz, 3, 
                                 group_idx.unsqueeze(1).expand(B, C, npoint, nsample))
        
        if points is not None:
            grouped_points = points.unsqueeze(2).expand(B, -1, npoint, N)
            grouped_points = torch.gather(grouped_points, 3,
                                        group_idx.unsqueeze(1).expand(B, points.shape[1], npoint, nsample))
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)
        else:
            grouped_points = grouped_xyz
            
        return grouped_xyz, grouped_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # Simplified interpolation - just concatenate and process
        if points1 is not None:
            points = torch.cat([points1, points2], dim=1)
        else:
            points = points2
            
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points = torch.relu(bn(conv(points)))
        return points


class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim=4, output_dim=1024):
        super(PointNetPlusPlus, self).__init__()
        self.input_dim = input_dim
        
        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, input_dim, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        
        # Global feature extraction
        self.global_conv = nn.Conv1d(1024, output_dim, 1)
        self.global_bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, xyz):
        # xyz: (B, N, 4) - [x, y, t, polarity]
        B, N, _ = xyz.shape
        
        # Split coordinates and features
        points_xyz = xyz[:, :, :3]  # (B, N, 3) - [x, y, t]
        points_features = xyz[:, :, 3:].transpose(2, 1)  # (B, 1, N) - polarity
        
        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(points_xyz, points_features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global feature
        global_feature = torch.relu(self.global_bn(self.global_conv(l3_points)))
        global_feature = torch.max(global_feature, dim=2)[0]  # (B, output_dim)
        
        return global_feature


class PoseRegressionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, dropout=0.3):
        super(PoseRegressionHead, self).__init__()
        
        # Translation head
        self.translation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Rotation head
        self.rotation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)
        )
        
    def forward(self, features):
        translation = self.translation_head(features)
        rotation_raw = self.rotation_head(features)
        
        # Normalize quaternion to unit length
        rotation = nn.functional.normalize(rotation_raw, p=2, dim=1)
        
        return translation, rotation


class EventPoseModel(nn.Module):
    def __init__(self, input_dim=4, feature_dim=1024):
        super(EventPoseModel, self).__init__()
        self.encoder = PointNetPlusPlus(input_dim, feature_dim)
        self.pose_head = PoseRegressionHead(feature_dim)
        
    def forward(self, events):
        features = self.encoder(events)
        translation, rotation = self.pose_head(features)
        return translation, rotation


def quaternion_loss(q_pred, q_gt):
    """
    Quaternion loss using inner product (accounts for double cover)
    """
    inner_product = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    loss = 1.0 - inner_product
    return torch.mean(loss)


def translation_loss(t_pred, t_gt):
    """
    L2 loss for translation
    """
    return nn.functional.mse_loss(t_pred, t_gt)


class PoseLoss(nn.Module):
    def __init__(self, lambda_t=1.0, lambda_r=1.0):
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
    Custom collate function to handle the event data
    """
    left_events = []
    right_events = []
    left_poses = []
    right_poses = []
    
    for sample in batch:
        # Convert structured array to regular array
        left_event = sample['left_events_strip']
        right_event = sample['right_events_strip']
        
        # Extract x, y, t, p fields
        left_data = np.column_stack([
            left_event['x'], left_event['y'], 
            left_event['t'], left_event['p']
        ])
        right_data = np.column_stack([
            right_event['x'], right_event['y'], 
            right_event['t'], right_event['p']
        ])
        
        left_events.append(torch.FloatTensor(left_data))
        right_events.append(torch.FloatTensor(right_data))
        left_poses.append(torch.FloatTensor(sample['left_pose']))
        right_poses.append(torch.FloatTensor(sample['right_pose']))
    
    return {
        'left_events': torch.stack(left_events),
        'right_events': torch.stack(right_events),
        'left_poses': torch.stack(left_poses),
        'right_poses': torch.stack(right_poses)
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_t_loss = 0.0
    total_r_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        left_events = batch['left_events'].to(device)
        right_events = batch['right_events'].to(device)
        left_poses = batch['left_poses'].to(device)
        right_poses = batch['right_poses'].to(device)
        
        # Split poses into translation and rotation
        left_t_gt = left_poses[:, :3]
        left_q_gt = left_poses[:, 3:]
        right_t_gt = right_poses[:, :3]
        right_q_gt = right_poses[:, 3:]
        
        optimizer.zero_grad()
        
        # Forward pass for left and right cameras
        left_t_pred, left_q_pred = model(left_events)
        right_t_pred, right_q_pred = model(right_events)
        
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
            left_poses = batch['left_poses'].to(device)
            right_poses = batch['right_poses'].to(device)
            
            left_t_gt = left_poses[:, :3]
            left_q_gt = left_poses[:, 3:]
            right_t_gt = right_poses[:, :3]
            right_q_gt = right_poses[:, 3:]
            
            left_t_pred, left_q_pred = model(left_events)
            right_t_pred, right_q_pred = model(right_events)
            
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
        'val_dataset_root': "//media/adarsh/One Touch/EventSLAM/dataset/val",  # Add validation path
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lambda_t': 1.0,
        'lambda_r': 1.0,
        'save_dir': './checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'early_stopping_patience': 10
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize dataset and dataloaders
    print("Loading datasets...")
        
    train_dataset = EventExtractionDataset(config['dataset_root'])
    val_dataset = EventExtractionDataset(config['val_dataset_root'])
    
    # For now, using placeholder - replace with actual dataset
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
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
    
    #Learning rate scheduler
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Training
        train_loss, train_t_loss, train_r_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation  
        val_loss, val_t_loss, val_r_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.6f} (T: {train_t_loss:.6f}, R: {train_r_loss:.6f})")
        print(f"Val Loss: {val_loss:.6f} (T: {val_t_loss:.6f}, R: {val_r_loss:.6f})")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
    train_model()