#!/usr/bin/env python3
"""
Training script for event-based metric learning with H5TripletDataset.

Features:
- Pose-supervised triplet mining from H5 cache
- PEVSLAM-style encoder architecture
- Adaptive triplet loss with hard negative mining
- Efficient H5 data loading with persistent workers
- Validation with recall@k metrics
- Automatic mixed precision training
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Import the H5 dataset
from h5_dataset import create_dataloader


# ============================================================================
# Model Architecture (PEVSLAM-style)
# ============================================================================

class EventEncoder(nn.Module):
    """
    Transformer-based event encoder for metric learning.
    Processes normalized events [B, N, 4] -> descriptors [B, D]
    """
    def __init__(self, 
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 descriptor_dim=256):
        super().__init__()
        
        self.d_model = d_model
        self.descriptor_dim = descriptor_dim
        
        # Input projection: [B, N, 4] -> [B, N, d_model]
        self.input_proj = nn.Linear(4, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: aggregate and project to descriptor
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.descriptor_head = nn.Sequential(
            nn.Linear(d_model, descriptor_dim),
            nn.LayerNorm(descriptor_dim)
        )
        
    def forward(self, events, mask):
        """
        Args:
            events: [B, N, 4] normalized events (x, y, t, p)
            mask: [B, N] binary mask (1 for valid, 0 for padding)
            
        Returns:
            descriptors: [B, descriptor_dim] L2-normalized descriptors
        """
        B, N, _ = events.shape
        
        # Project inputs
        x = self.input_proj(events)  # [B, N, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :N, :]
        
        # Create attention mask (transformer expects True for positions to ignore)
        attn_mask = ~mask.bool()  # [B, N]
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, N, d_model]
        
        # Masked pooling (mean of valid events)
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        x_masked = x * mask_expanded
        x_pooled = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # [B, d_model]
        
        # Apply pooling transformation
        x_pooled = self.pooling(x_pooled)
        
        # Project to descriptor
        descriptor = self.descriptor_head(x_pooled)  # [B, descriptor_dim]
        
        # L2 normalize
        descriptor = F.normalize(descriptor, p=2, dim=1)
        
        return descriptor


# ============================================================================
# Loss Functions
# ============================================================================

class AdaptiveTripletLoss(nn.Module):
    """
    Adaptive triplet loss with hard negative mining.
    Loss = max(0, ||a - p||^2 - ||a - n||^2 + margin)
    """
    def __init__(self, margin=0.5, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: [B, D] L2-normalized descriptors
            
        Returns:
            loss: scalar
            stats: dict with loss statistics
        """
        # Compute distances
        dist_ap = torch.sum((anchor - positive) ** 2, dim=1)  # [B]
        dist_an = torch.sum((anchor - negative) ** 2, dim=1)  # [B]
        
        # Triplet loss
        losses = F.relu(dist_ap - dist_an + self.margin)
        
        # Hard mining: only backprop through hardest 80% of triplets
        if self.hard_mining and self.training:
            k = int(0.8 * len(losses))
            if k > 0:
                hard_losses, _ = torch.topk(losses, k)
                loss = hard_losses.mean()
            else:
                loss = losses.mean()
        else:
            loss = losses.mean()
        
        # Statistics
        stats = {
            'loss': loss.item(),
            'dist_ap': dist_ap.mean().item(),
            'dist_an': dist_an.mean().item(),
            'margin_violations': (losses > 0).float().mean().item(),
        }
        
        return loss, stats


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    
    metrics = {
        'loss': [],
        'dist_ap': [],
        'dist_an': [],
        'margin_violations': []
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        anchor_events = batch['anchor_events'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive_events = batch['positive_events'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        negative_events = batch['negative_events'].to(device)
        negative_mask = batch['negative_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            # Forward pass
            anchor_desc = model(anchor_events, anchor_mask)
            positive_desc = model(positive_events, positive_mask)
            negative_desc = model(negative_events, negative_mask)
            
            # Compute loss
            loss, stats = loss_fn(anchor_desc, positive_desc, negative_desc)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        for key in metrics:
            metrics[key].append(stats[key])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{stats['loss']:.4f}",
            'ap': f"{stats['dist_ap']:.3f}",
            'an': f"{stats['dist_an']:.3f}"
        })
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return avg_metrics


@torch.no_grad()
def validate(model, dataloader, device, k_values=[1, 5, 10]):
    """
    Validate with recall@k metrics.
    Build descriptor database and compute retrieval accuracy.
    """
    model.eval()
    
    descriptors = []
    poses = []
    
    print("Computing descriptors...")
    for batch in tqdm(dataloader, desc="Validation"):
        anchor_events = batch['anchor_events'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        anchor_pose = batch['anchor_pose']
        
        # Compute descriptor
        desc = model(anchor_events, anchor_mask)
        
        descriptors.append(desc.cpu())
        poses.append(anchor_pose)
    
    # Stack all descriptors and poses
    descriptors = torch.cat(descriptors, dim=0)  # [N, D]
    poses = torch.cat(poses, dim=0)  # [N, 7]
    positions = poses[:, :3]  # [N, 3]
    
    N = len(descriptors)
    print(f"Built descriptor database: {N} samples")
    
    # Compute pairwise distances for ground truth
    pos_dists = torch.cdist(positions, positions)  # [N, N]
    
    # Define positives: within 0.5m
    gt_positives = (pos_dists <= 0.5) & (pos_dists > 0)  # Exclude self
    
    # Compute descriptor similarities
    desc_sims = torch.mm(descriptors, descriptors.t())  # [N, N] cosine similarity
    
    # Compute recall@k
    recalls = {}
    for k in k_values:
        # Get top-k most similar for each query
        _, topk_indices = torch.topk(desc_sims, k=k+1, dim=1)  # +1 to exclude self
        topk_indices = topk_indices[:, 1:]  # Remove self
        
        # Check if any top-k is a ground truth positive
        correct = 0
        for i in range(N):
            if gt_positives[i].sum() > 0:  # Has at least one positive
                retrieved_positives = gt_positives[i, topk_indices[i]]
                if retrieved_positives.any():
                    correct += 1
        
        recall = correct / N
        recalls[f'recall@{k}'] = recall
        print(f"Recall@{k}: {recall:.4f}")
    
    return recalls


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train event-based metric learning model')
    
    # Data arguments
    parser.add_argument('--triplet-cache', type=str, required=True,
                        help='Path to triplet_cache.h5')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing sequence folders')
    parser.add_argument('--poses-h5', type=str, required=True,
                        help='Path to poses.h5')
    parser.add_argument('--camera', type=str, default='left',
                        choices=['left', 'right'],
                        help='Camera to use (default: left)')
    
    # Model arguments
    parser.add_argument('--d-model', type=int, default=256,
                        help='Transformer model dimension (default: 256)')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    parser.add_argument('--descriptor-dim', type=int, default=256,
                        help='Output descriptor dimension (default: 256)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Triplet loss margin (default: 0.5)')
    
    # Data loading arguments
    parser.add_argument('--event-window-ms', type=int, default=50,
                        help='Event window in milliseconds (default: 50)')
    parser.add_argument('--n-events', type=int, default=1024,
                        help='Number of events per sample (default: 1024)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")
    
    # TensorBoard
    log_dir = output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Create dataloaders
    print("\nCreating training dataloader...")
    train_loader = create_dataloader(
        args.triplet_cache,
        args.data_root,
        args.poses_h5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        event_window_ms=args.event_window_ms,
        n_events=args.n_events,
        camera=args.camera
    )
    
    print(f"Training dataset: {len(train_loader.dataset)} triplets, {len(train_loader)} batches")
    
    # Note: For validation, you might want to create a separate validation split
    # For now, we'll validate on a subset of training data
    print("\nCreating validation dataloader...")
    val_loader = create_dataloader(
        args.triplet_cache,
        args.data_root,
        args.poses_h5,
        batch_size=args.batch_size,
        num_workers=args.num_workers // 2,
        event_window_ms=args.event_window_ms,
        n_events=args.n_events,
        camera=args.camera
    )
    
    # Create model
    print("\nCreating model...")
    model = EventEncoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        descriptor_dim=args.descriptor_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss function
    loss_fn = AdaptiveTripletLoss(margin=args.margin, hard_mining=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    best_recall = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch
        )
        
        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('train/lr', current_lr, epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Dist AP: {train_metrics['dist_ap']:.4f}")
        print(f"  Dist AN: {train_metrics['dist_an']:.4f}")
        print(f"  Margin violations: {train_metrics['margin_violations']:.2%}")
        print(f"  Learning rate: {current_lr:.2e}")
        
        # Validate
        if (epoch + 1) % args.validate_every == 0:
            print("\nRunning validation...")
            val_metrics = validate(model, val_loader, device)
            
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save best model
            if val_metrics['recall@1'] > best_recall:
                best_recall = val_metrics['recall@1']
                best_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': val_metrics,
                }, best_path)
                print(f"  Saved best model (recall@1: {best_recall:.4f}) to {best_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        print()
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best recall@1: {best_recall:.4f}")
    print("="*80)
    
    writer.close()


if __name__ == '__main__':
    main()