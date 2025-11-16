import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
from tqdm import tqdm
import random
import argparse

from utils.loadevents import event_extractor
from scripts.pevslam_net import PEVSLAM


class InfoNCELoss(nn.Module):
    """
    Vectorized InfoNCE loss for contrastive learning:
    - Identifies positives (same sequence, close in time)
    - Uses all other samples in batch as negatives
    - Temperature-scaled softmax
    """
    
    def __init__(self, temperature=0.07, positive_time_threshold=5.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.positive_time_threshold = positive_time_threshold * 1e6  # Convert to microseconds
    
    def forward(self, descriptors, metadata):
        """
        Args:
            descriptors: [B, D] - normalized descriptors
            metadata: list of dicts with 'sequence', 'time_mid' keys
        
        Returns:
            loss: scalar
            metrics: dict with accuracy, num_positives, etc.
        """
        B = descriptors.shape[0]
        device = descriptors.device
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.mm(descriptors, descriptors.t()) / self.temperature
        
        # Vectorized positive mask creation
        sequences = torch.tensor([hash(m['sequence']) for m in metadata], device=device)
        times = torch.tensor([m['time_mid'] for m in metadata], device=device)
        
        # Same sequence mask [B, B]
        same_seq_mask = sequences.unsqueeze(0) == sequences.unsqueeze(1)
        
        # Time difference mask [B, B]
        time_diff = torch.abs(times.unsqueeze(0) - times.unsqueeze(1))
        close_time_mask = time_diff < self.positive_time_threshold
        
        # Positive mask: same sequence AND close time AND not self
        positive_mask = same_seq_mask & close_time_mask
        positive_mask.fill_diagonal_(False)
        
        # Negative mask: everything except self and positives
        negative_mask = ~positive_mask
        negative_mask.fill_diagonal_(False)
        
        # Count positives per sample
        num_positives = positive_mask.sum(dim=1)  # [B]
        
        # Only process samples that have positives
        valid_samples = num_positives > 0
        
        if not valid_samples.any():
            return torch.tensor(0.0, device=device), {
                'accuracy': 0.0,
                'num_positives': 0,
                'mean_pos_per_sample': 0.0
            }
        
        # Compute loss per sample using stable logsumexp
        loss_per_sample = []
        
        for i in range(B):
            if not valid_samples[i]:
                continue
            
            # Get positive and negative similarities for this sample
            pos_sims = sim_matrix[i][positive_mask[i]]  # Only positive similarities
            neg_sims = sim_matrix[i][negative_mask[i]]  # Only negative similarities
            
            if len(pos_sims) == 0 or len(neg_sims) == 0:
                continue
            
            # Compute: log(sum(exp(pos))) - log(sum(exp(pos)) + sum(exp(neg)))
            pos_term = torch.logsumexp(pos_sims, dim=0)
            
            # Combine pos and neg for denominator
            all_sims = torch.cat([pos_sims, neg_sims], dim=0)
            all_term = torch.logsumexp(all_sims, dim=0)
            
            sample_loss = -(pos_term - all_term)
            loss_per_sample.append(sample_loss)
        
        if len(loss_per_sample) == 0:
            return torch.tensor(0.0, device=device), {
                'accuracy': 0.0,
                'num_positives': 0,
                'mean_pos_per_sample': 0.0
            }
        
        total_loss = torch.stack(loss_per_sample).mean()
        
        # Compute accuracy (is max positive similarity > max negative similarity?)
        with torch.no_grad():
            max_pos_sim = sim_matrix.clone()
            max_pos_sim[~positive_mask] = -float('inf')
            max_pos_sim = max_pos_sim.max(dim=1).values  # [B]
            
            max_neg_sim = sim_matrix.clone()
            max_neg_sim[~negative_mask] = -float('inf')
            max_neg_sim = max_neg_sim.max(dim=1).values  # [B]
            
            correct = (max_pos_sim > max_neg_sim) & valid_samples
            accuracy = correct.sum().item() / valid_samples.sum().item()
        
        metrics = {
            'accuracy': accuracy,
            'num_positives': num_positives.sum().item(),
            'mean_pos_per_sample': num_positives[valid_samples].float().mean().item()
        }
        
        return total_loss, metrics


def collate_fn(batch):
    """Custom collate function to handle batch data"""
    return batch


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler):
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_positives = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_data in pbar:
        events = torch.stack([d['events'] for d in batch_data]).to(device)
        masks = torch.stack([d['mask'] for d in batch_data]).to(device)
        metadata = [{'sequence': d['sequence'], 'time_mid': d['time_mid']} for d in batch_data]
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda'):
            descriptors = model(events, masks)
            loss, metrics = criterion(descriptors, metadata)
        
        # Always track metrics
        total_loss += loss.item()
        total_accuracy += metrics['accuracy']
        total_positives += metrics['num_positives']
        num_batches += 1
        
        # Backprop only on non-zero loss
        if loss.item() > 0:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{metrics['accuracy']:.3f}",
            'pos': int(metrics['mean_pos_per_sample'])
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_accuracy


def validate(model, dataloader, device):
    """
    Validation using retrieval metrics:
    - Recall@1, @5, @10
    - Can the model retrieve the correct place?
    """
    model.eval()
    
    all_descriptors = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation"):
            events = torch.stack([d['events'] for d in batch_data]).to(device)
            masks = torch.stack([d['mask'] for d in batch_data]).to(device)
            
            with autocast(device_type='cuda'):
                descriptors = model(events, masks)
            
            all_descriptors.append(descriptors.cpu())
            
            for d in batch_data:
                all_metadata.append({
                    'sequence': d['sequence'],
                    'time_mid': d['time_mid'],
                    'time_start': d['time_start'],
                    'time_end': d['time_end']
                })
    
    all_descriptors = torch.cat(all_descriptors, dim=0)  # [N, D]
    
    # Compute similarity matrix
    similarity = torch.mm(all_descriptors, all_descriptors.t())  # [N, N]
    
    # For each query, find top-k retrievals
    recalls = {1: [], 5: [], 10: []}
    
    for i in range(len(all_metadata)):
        # Get similarities (excluding self)
        sims = similarity[i].clone()
        sims[i] = -float('inf')  # Exclude self
        
        # Top-k retrievals
        top_k_indices = torch.topk(sims, k=10, dim=0).indices.tolist()
        
        # Ground truth: same sequence, within ±1 second
        query_time = all_metadata[i]['time_mid']
        query_seq = all_metadata[i]['sequence']
        
        correct_indices = []
        for j in range(len(all_metadata)):
            if i == j:
                continue
            same_seq = all_metadata[j]['sequence'] == query_seq
            time_diff = abs(all_metadata[j]['time_mid'] - query_time)
            if same_seq and time_diff < 1e6:  # Within 1 second
                correct_indices.append(j)
        
        if len(correct_indices) == 0:
            continue  # No ground truth for this query
        
        # Check recall@k
        for k in [1, 5, 10]:
            retrieved = top_k_indices[:k]
            hit = any(idx in correct_indices for idx in retrieved)
            recalls[k].append(1.0 if hit else 0.0)
    
    # Average recalls
    recall_at_1 = np.mean(recalls[1]) if recalls[1] else 0.0
    recall_at_5 = np.mean(recalls[5]) if recalls[5] else 0.0
    recall_at_10 = np.mean(recalls[10]) if recalls[10] else 0.0
    
    return recall_at_1, recall_at_5, recall_at_10


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train PEVSLAM place recognition model')
    parser.add_argument('--dataset_root', type=str, default='/workspace/npy_cache',
                        help='Path to NPZ dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: load only 0.1%% of data for quick testing')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    if args.debug:
        print("⚠️  DEBUG MODE: Loading only 0.1% of data for testing")
    
    # Load dataset with debug flag
    print("Loading dataset...")
    dataset = event_extractor(args.dataset_root, N=1024, num_workers=4, debug=args.debug)
    
    print(f"Dataset loaded: {len(dataset.flat_samples)} samples")
    
    # Split into train/val sequences
    sequences = list(dataset.sequence_map.keys())
    random.shuffle(sequences)
    
    # In debug mode, use fewer sequences for validation
    if args.debug:
        val_sequences = sequences[:1] if len(sequences) > 1 else sequences[:1]
        train_sequences = sequences[1:] if len(sequences) > 1 else []
    else:
        # Use 10% of sequences for validation (at least 2)
        num_val = max(2, len(sequences) // 10)
        val_sequences = sequences[:num_val]
        train_sequences = sequences[num_val:]
    
    print(f"\nTraining sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create subset datasets for train/val
    class SubsetDataset:
        """Wrapper to create a subset view of the dataset"""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.flat_samples = [dataset.flat_samples[i] for i in indices]
            # Create new sequence_map for this subset
            self.sequence_map = {}
            for i in indices:
                sample = dataset.flat_samples[i]
                seq = sample['sequence']
                if seq not in self.sequence_map:
                    self.sequence_map[seq] = []
                self.sequence_map[seq].append(sample)
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    # Create train/val splits
    train_indices = [i for i, s in enumerate(dataset.flat_samples) 
                    if s['sequence'] in train_sequences]
    val_indices = [i for i, s in enumerate(dataset.flat_samples) 
                  if s['sequence'] in val_sequences]
    
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create dataloaders with sequential order (keep time close)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Sequential - keeps temporally close samples together
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # Drop incomplete batches for consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"\nBatches per epoch: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}\n")
    
    # Create model
    model = PEVSLAM(
        base_channel=4,
        descriptor_dim=256
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Loss function
    criterion = InfoNCELoss(temperature=0.07, positive_time_threshold=5.0)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # Training loop
    best_recall = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        
        # Validate every 5 epochs (or every epoch in debug mode)
        validate_freq = 1 if args.debug else 5
        if epoch % validate_freq == 0:
            print("\nValidating...")
            recall_1, recall_5, recall_10 = validate(model, val_loader, device)
            
            print(f"Recall@1:  {recall_1:.3f}")
            print(f"Recall@5:  {recall_5:.3f}")
            print(f"Recall@10: {recall_10:.3f}")
            
            # Save best model
            if recall_10 > best_recall:
                best_recall = recall_10
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_10': recall_10,
                    'recall_5': recall_5,
                    'recall_1': recall_1,
                }, checkpoint_path)
                print(f"💾 Saved best model (Recall@10: {recall_10:.3f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: epoch {epoch}")
        
        # Step scheduler after epoch
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # In debug mode, exit after 2 epochs
        if args.debug and epoch >= 2:
            print("\n✅ Debug mode: Training test completed successfully!")
            break
    
    print("\n🎉 Training completed!")
    print(f"Best Recall@10: {best_recall:.3f}")


if __name__ == "__main__":
    main()