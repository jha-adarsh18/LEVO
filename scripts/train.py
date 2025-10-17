import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
from tqdm import tqdm
import random

from utils.loadevents import event_extractor
from model import PEVSLAM


class SequenceAwareSampler(Sampler):
    """
    Smart sampler that creates batches with proper positives/negatives:
    - Samples full sequences (5-8 packets each)
    - Ensures temporal structure is preserved
    - Provides positives (1-5s apart) and negatives (>30s or different sequence)
    """
    
    def __init__(self, dataset, batch_size=128, samples_per_sequence=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_sequence = samples_per_sequence
        
        # Group indices by sequence
        self.sequence_groups = {}
        for idx, sample_info in enumerate(dataset.flat_samples):
            seq = sample_info['sequence']
            if seq not in self.sequence_groups:
                self.sequence_groups[seq] = []
            self.sequence_groups[seq].append(idx)
        
        self.sequences = list(self.sequence_groups.keys())
        print(f"Sampler: {len(self.sequences)} sequences, batch_size={batch_size}")
    
    def __iter__(self):
        # Shuffle sequences
        random.shuffle(self.sequences)
        
        batches = []
        current_batch = []
        
        for seq in self.sequences:
            indices = self.sequence_groups[seq]
            
            # Sample up to samples_per_sequence packets from this sequence
            if len(indices) > self.samples_per_sequence:
                sampled = random.sample(indices, self.samples_per_sequence)
            else:
                sampled = indices
            
            current_batch.extend(sampled)
            
            # When batch is full, yield it
            while len(current_batch) >= self.batch_size:
                batches.append(current_batch[:self.batch_size])
                current_batch = current_batch[self.batch_size:]
        
        # Add remaining samples
        if len(current_batch) > 0:
            batches.append(current_batch)
        
        # Shuffle batches
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        total_samples = sum(min(len(indices), self.samples_per_sequence) 
                          for indices in self.sequence_groups.values())
        return (total_samples + self.batch_size - 1) // self.batch_size


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
        
        # Vectorized loss computation
        # For each sample, compute: -log(sum(exp(pos_sim)) / (sum(exp(pos_sim)) + sum(exp(neg_sim))))
        
        # Mask out invalid entries with large negative values
        pos_sim = sim_matrix.clone()
        pos_sim[~positive_mask] = -1e9
        
        neg_sim = sim_matrix.clone()
        neg_sim[~negative_mask] = -1e9
        
        # Compute log-sum-exp for numerical stability
        pos_exp_sum = torch.logsumexp(pos_sim, dim=1)  # [B]
        neg_exp_sum = torch.logsumexp(neg_sim, dim=1)  # [B]
        all_exp_sum = torch.logsumexp(torch.stack([pos_exp_sum, neg_exp_sum], dim=1), dim=1)  # [B]
        
        # Loss per sample
        loss_per_sample = -(pos_exp_sum - all_exp_sum)  # [B]
        
        # Average over valid samples only
        total_loss = loss_per_sample[valid_samples].mean()
        
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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler):
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_positives = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_indices in pbar:
        # Get batch data
        batch_data = [dataloader.dataset[idx] for idx in batch_indices]
        
        events = torch.stack([d['events'] for d in batch_data]).to(device)
        masks = torch.stack([d['mask'] for d in batch_data]).to(device)
        metadata = [{'sequence': d['sequence'], 'time_mid': d['time_mid']} for d in batch_data]
        
        # Forward pass with mixed precision
        with autocast():
            descriptors = model(events, masks)
            loss, metrics = criterion(descriptors, metadata)
        
        if loss.item() > 0:  # Only backprop if valid loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            total_positives += metrics['num_positives']
            num_batches += 1
            
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
        for batch_indices in tqdm(dataloader, desc="Validation"):
            batch_data = [dataloader.dataset[idx] for idx in batch_indices]
            
            events = torch.stack([d['events'] for d in batch_data]).to(device)
            masks = torch.stack([d['mask'] for d in batch_data]).to(device)
            
            with autocast():
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
    # Configuration
    dataset_root = r"/workspace/PEVSLAM/npy_cache"
    batch_size = 128  # Increased from 64
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = event_extractor(dataset_root, N=1024, num_workers=4)  # Reduced from 16
    
    # Split into train/val sequences
    sequences = list(dataset.sequence_map.keys())
    random.shuffle(sequences)
    
    val_sequences = sequences[:2]  # 2 sequences for validation
    train_sequences = sequences[2:]  # Rest for training
    
    print(f"Training sequences: {train_sequences}")
    print(f"Validation sequences: {val_sequences}")
    
    # Create train/val splits
    train_indices = [i for i, s in enumerate(dataset.flat_samples) 
                    if s['sequence'] in train_sequences]
    val_indices = [i for i, s in enumerate(dataset.flat_samples) 
                  if s['sequence'] in val_sequences]
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create samplers and dataloaders
    train_sampler = SequenceAwareSampler(
        dataset, 
        batch_size=batch_size,
        samples_per_sequence=8
    )
    
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=8,  # Increased from 4
        pin_memory=True
    )
    
    val_sampler = SequenceAwareSampler(
        dataset,
        batch_size=batch_size,
        samples_per_sequence=100  # Use all samples for validation
    )
    
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        num_workers=8,  # Increased from 4
        pin_memory=True
    )
    
    # Create model
    model = PEVSLAM(
        base_channel=4,
        descriptor_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss function
    criterion = InfoNCELoss(temperature=0.07, positive_time_threshold=5.0)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_recall = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        
        # Validate every 10 epochs (reduced from 5)
        if epoch % 10 == 0:
            print("\nValidating...")
            recall_1, recall_5, recall_10 = validate(model, val_loader, device)
            
            print(f"Recall@1:  {recall_1:.3f}")
            print(f"Recall@5:  {recall_5:.3f}")
            print(f"Recall@10: {recall_10:.3f}")
            
            # Save best model
            if recall_10 > best_recall:
                best_recall = recall_10
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_10': recall_10,
                }, 'best_model.pth')
                print(f"Saved best model (Recall@10: {recall_10:.3f})")
        
        # Step scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\nTraining completed!")
    print(f"Best Recall@10: {best_recall:.3f}")


if __name__ == "__main__":
    main()