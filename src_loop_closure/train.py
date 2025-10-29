#!/usr/bin/env python3
"""
Training script for event-based metric learning with H5TripletDataset.

Features:
- Pose-supervised triplet mining from H5 cache
- PEVSLAM-style encoder architecture
- Adaptive triplet loss with hard negative mining
- FIXED: Anti-collapse mechanisms (temperature scaling, warmup, monitoring)
- Efficient H5 data loading with persistent workers
- Validation with recall@k metrics
- Automatic mixed precision training
- Gradient accumulation for larger effective batch sizes
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Import the H5 dataset
from h5_dataset import H5TripletDataset, worker_init_fn, collate_fn


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
                 num_layers=4,
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
    Adaptive triplet loss with hard negative mining and temperature scaling.
    Loss = max(0, ||a - p||^2 - ||a - n||^2 + margin)
    
    Temperature scaling prevents representation collapse.
    """
    def __init__(self, margin=0.5, hard_mining=True, temperature=1.0):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.temperature = temperature
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: [B, D] L2-normalized descriptors
            
        Returns:
            loss: scalar
            stats: dict with loss statistics
        """
        # Compute squared L2 distances (for L2-normalized vectors: dist = 2 - 2*cos_sim)
        dist_ap = torch.sum((anchor - positive) ** 2, dim=1)  # [B]
        dist_an = torch.sum((anchor - negative) ** 2, dim=1)  # [B]
        
        # Apply temperature scaling to prevent collapse
        dist_ap = dist_ap / self.temperature
        dist_an = dist_an / self.temperature
        
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
        
        # Statistics (use original distances for logging)
        stats = {
            'loss': loss.item(),
            'dist_ap': (dist_ap * self.temperature).mean().item(),
            'dist_an': (dist_an * self.temperature).mean().item(),
            'margin_violations': (losses > 0).float().mean().item(),
        }
        
        return loss, stats


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch, 
                writer, accumulation_steps=1, log_every=100):
    """Train for one epoch with gradient accumulation and collapse monitoring."""
    model.train()
    
    metrics = {
        'loss': [],
        'dist_ap': [],
        'dist_an': [],
        'margin_violations': [],
        'desc_std': [],
        'desc_norm': []
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        anchor_events = batch['anchor_events'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive_events = batch['positive_events'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        negative_events = batch['negative_events'].to(device)
        negative_mask = batch['negative_mask'].to(device)
        
        # Mixed precision training
        with autocast('cuda'):
            # Forward pass
            anchor_desc = model(anchor_events, anchor_mask)
            positive_desc = model(positive_events, positive_mask)
            negative_desc = model(negative_events, negative_mask)
            
            # ANTI-COLLAPSE: Monitor descriptor statistics
            with torch.no_grad():
                desc_std = anchor_desc.std(dim=0).mean().item()
                desc_norm = anchor_desc.norm(dim=1).mean().item()
                metrics['desc_std'].append(desc_std)
                metrics['desc_norm'].append(desc_norm)
                
                # Warning if collapse detected
                if desc_std < 0.01:
                    print(f"\n⚠️  WARNING: Descriptor collapse detected! std={desc_std:.6f}")
            
            # Compute loss
            loss, stats = loss_fn(anchor_desc, positive_desc, negative_desc)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # ANTI-COLLAPSE: Aggressive gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics (use original loss, not scaled)
        stats['loss'] = stats['loss'] * accumulation_steps
        for key in ['loss', 'dist_ap', 'dist_an', 'margin_violations']:
            metrics[key].append(stats[key])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{stats['loss']:.4f}",
            'ap': f"{stats['dist_ap']:.3f}",
            'an': f"{stats['dist_an']:.3f}",
            'std': f"{desc_std:.3f}"
        })
        
        # Log to tensorboard periodically
        if batch_idx % log_every == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train_iter/loss', stats['loss'], global_step)
            writer.add_scalar('train_iter/desc_std', desc_std, global_step)
            writer.add_scalar('train_iter/desc_norm', desc_norm, global_step)
    
    # Handle any remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return avg_metrics


@torch.no_grad()
def validate(model, dataloader, device, k_values=[1, 5, 10], max_samples=None):
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
        
        # Limit validation size if specified
        if max_samples is not None and len(descriptors) * batch['anchor_events'].shape[0] >= max_samples:
            break
    
    # Stack all descriptors and poses
    descriptors = torch.cat(descriptors, dim=0)  # [N, D]
    poses = torch.cat(poses, dim=0)  # [N, 7]
    positions = poses[:, :3]  # [N, 3]
    
    N = len(descriptors)
    print(f"Built descriptor database: {N} samples")
    
    # ANTI-COLLAPSE: Check descriptor statistics
    desc_std = descriptors.std(dim=0).mean().item()
    desc_mean_norm = descriptors.norm(dim=1).mean().item()
    print(f"Descriptor statistics: std={desc_std:.4f}, mean_norm={desc_mean_norm:.4f}")
    
    if desc_std < 0.01:
        print("⚠️  WARNING: Validation descriptors have collapsed!")
    
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
        total = 0
        for i in range(N):
            if gt_positives[i].sum() > 0:  # Has at least one positive
                retrieved_positives = gt_positives[i, topk_indices[i]]
                if retrieved_positives.any():
                    correct += 1
                total += 1
        
        recall = correct / total if total > 0 else 0.0
        recalls[f'recall@{k}'] = recall
        print(f"Recall@{k}: {recall:.4f}")
    
    # Add descriptor stats to metrics
    recalls['desc_std'] = desc_std
    recalls['desc_norm'] = desc_mean_norm
    
    return recalls


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """Warmup followed by cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train event-based metric learning model')
    
    # Data arguments
    parser.add_argument('--triplet-cache', type=str, required=True,
                        help='Path to events_cache.h5')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory containing sequence folders (for on-the-fly mode)')
    parser.add_argument('--poses-h5', type=str, default=None,
                        help='Path to poses.h5 (for on-the-fly mode)')
    parser.add_argument('--camera', type=str, default='left',
                        choices=['left', 'right'],
                        help='Camera to use (default: left)')
    parser.add_argument('--train-fraction', type=float, default=0.7,
                        help='Fraction of dataset to use for training (default: 0.7)')
    parser.add_argument('--val-fraction', type=float, default=0.15,
                        help='Fraction of dataset to use for validation (default: 0.15)')
    
    # Model arguments
    parser.add_argument('--d-model', type=int, default=256,
                        help='Transformer model dimension (default: 256)')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers (default: 4)')
    parser.add_argument('--descriptor-dim', type=int, default=256,
                        help='Output descriptor dimension (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate (default: 3e-5)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay (default: 1e-3)')
    parser.add_argument('--margin', type=float, default=0.7,
                        help='Triplet loss margin (default: 0.7)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for triplet loss (default: 0.1, lower = more stable)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs (default: 5)')
    
    # Data loading arguments
    parser.add_argument('--event-window-ms', type=int, default=50,
                        help='Event window in milliseconds (default: 50)')
    parser.add_argument('--n-events', type=int, default=1024,
                        help='Number of events per sample (default: 1024)')
    parser.add_argument('--num-workers', type=int, default=12,
                        help='Number of data loading workers (default: 12)')
    parser.add_argument('--prefetch-factor', type=int, default=8,
                        help='Prefetch factor for data loading (default: 8)')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Validate every N epochs (default: 1)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster training')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set multiprocessing start method
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass  # Already set
    
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
    
    # ========================================================================
    # Create proper non-overlapping train/val split
    # ========================================================================
    print("\n" + "="*80)
    print("Creating train/val split with NO OVERLAP...")
    print("="*80)
    
    # Create full dataset first (no subsetting yet)
    full_dataset = H5TripletDataset(
        triplet_cache_path=args.triplet_cache,
        data_root=args.data_root,
        poses_h5_path=args.poses_h5,
        event_window_ms=args.event_window_ms,
        n_events=args.n_events,
        camera=args.camera,
        use_cached_events=True,
        subset_fraction=1.0  # Use ALL data, we'll split manually
    )
    
    # Get total number of samples
    n_total = len(full_dataset)
    print(f"Total samples: {n_total}")
    
    # Create non-overlapping indices with fixed seed
    indices = np.arange(n_total)
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(indices)
    
    # Split: train_fraction, val_fraction, rest unused
    n_train = int(args.train_fraction * n_total)
    n_val = int(args.val_fraction * n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    
    print(f"Train samples: {len(train_indices)} ({args.train_fraction:.1%})")
    print(f"Val samples: {len(val_indices)} ({args.val_fraction:.1%})")
    print(f"Overlap check: {len(set(train_indices) & set(val_indices))} (should be 0)")
    
    if len(set(train_indices) & set(val_indices)) > 0:
        raise ValueError("ERROR: Train and val indices overlap!")
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers // 2,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Effective batch size: {effective_batch_size}")
    print("="*80 + "\n")
    
    # Create model
    print("Creating model...")
    model = EventEncoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        descriptor_dim=args.descriptor_dim,
        dropout=args.dropout
    ).to(device)
    
    # Compile model for faster training
    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='max-autotune')
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    print(f"Samples per parameter: {len(train_dataset) / n_params:.2f}")
    
    # Loss function with temperature scaling
    loss_fn = AdaptiveTripletLoss(
        margin=args.margin, 
        hard_mining=True,
        temperature=args.temperature
    )
    print(f"Using temperature={args.temperature} for anti-collapse")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        eta_min=1e-7
    )
    print(f"Using warmup for {args.warmup_epochs} epochs")
    
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
            model, train_loader, optimizer, loss_fn, scaler, device, epoch,
            writer, accumulation_steps=args.accumulation_steps
        )
        
        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Dist AP: {train_metrics['dist_ap']:.4f}")
        print(f"  Dist AN: {train_metrics['dist_an']:.4f}")
        print(f"  Margin violations: {train_metrics['margin_violations']:.2%}")
        print(f"  Descriptor std: {train_metrics['desc_std']:.4f} (collapse if < 0.01)")
        print(f"  Descriptor norm: {train_metrics['desc_norm']:.4f}")
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
                print(f"  ✓ Saved best model (recall@1: {best_recall:.4f}) to {best_path}")
        
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