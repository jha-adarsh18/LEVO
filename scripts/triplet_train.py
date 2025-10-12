import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import time
from threading import Thread
from queue import Queue

from scripts.pevslam_net import PEVSLAM
from utils.loadevents import event_extractor
from utils.sample_and_mask import sample_and_mask

class TripletEventDataset(Dataset):
    """
    Optimized dataset for vast.ai with pre-cached triplet indices
    """

    def __init__(self, base_dataset, temporal_positive_range=3, temporal_negative_threshold=3):
        self.base_dataset = base_dataset
        self.pos_range = temporal_positive_range
        self.neg_theshold = temporal_negative_threshold

        # build sequence index for efficient sampling
        self.sequence_packets = self.build_sequence_index()

        # Filter flat_samples to only include valid indices
        valid_indices = set()
        for indices in self.sequence_packets.values():
            valid_indices.update(indices)
    
        self.valid_indices = sorted(list(valid_indices))
        
        # Pre-generate all triplet indices (1. Pre-caching)
        print("Pre-caching triplet indices...")
        self.triplets = self._generate_triplets()
        print(f"Pre-cached {len(self.triplets)} triplets")

    def build_sequence_index(self):
        """Group packets by sequence for efficient sampling - metadata only"""
        sequence_packets = {}
    
        for idx, sample in enumerate(tqdm(self.base_dataset.flat_samples, desc="Building sequence index")):
            seq_idx = sample['seq_idx']
        
            if seq_idx not in sequence_packets:
                sequence_packets[seq_idx] = []
        
            sequence_packets[seq_idx].append(idx)
    
        # Remove sequences with too few samples
        min_samples = max(self.pos_range + 1, self.neg_theshold + 1)
        sequence_packets = {
            k: v for k, v in sequence_packets.items() 
            if len(v) >= min_samples
        }
    
        print(f"Built sequence index with {len(sequence_packets)} sequences")
        return sequence_packets
    
    def _generate_triplets(self):
        """Pre-compute all triplet indices before training"""
        triplets = []
        
        for idx in tqdm(self.valid_indices, desc="Pre-generating triplets"):
            try:
                seq_idx = self.base_dataset.flat_samples[idx]['seq_idx']
                
                if seq_idx not in self.sequence_packets:
                    continue
                
                sequence_indices = self.sequence_packets[seq_idx]
                
                try:
                    anchor_position = sequence_indices.index(idx)
                except ValueError:
                    continue
                
                # Sample positive (temporal neighbor)
                pos_offset = np.random.randint(1, self.pos_range + 1)
                pos_idx_position = min(anchor_position + pos_offset, len(sequence_indices) - 1)
                pos_idx = sequence_indices[pos_idx_position]
                
                # Sample negative (far away or different sequence)
                valid_negatives = [
                    sequence_indices[i] for i in range(len(sequence_indices))
                    if abs(i - anchor_position) >= self.neg_theshold
                ]
                
                if len(valid_negatives) == 0:
                    other_sequences = [s for s in self.sequence_packets.keys() if s != seq_idx]
                    if len(other_sequences) > 0:
                        other_seq = np.random.choice(other_sequences)
                        neg_idx = np.random.choice(self.sequence_packets[other_seq])
                    else:
                        continue
                else:
                    neg_idx = np.random.choice(valid_negatives)
                
                triplets.append((idx, pos_idx, neg_idx))
            
            except Exception:
                continue
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """Simple, fast __getitem__ - no sampling or retry logic"""
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        
        try:
            # Load anchor
            anchor_packets = self.base_dataset[anchor_idx]
            if len(anchor_packets) == 0:
                # Fallback to zeros if empty
                raise ValueError("Empty anchor")
            anchor = anchor_packets[np.random.randint(0, len(anchor_packets))]
            
            # Load positive
            positive_packets = self.base_dataset[pos_idx]
            if len(positive_packets) == 0:
                raise ValueError("Empty positive")
            positive = positive_packets[np.random.randint(0, len(positive_packets))]
            
            # Load negative
            negative_packets = self.base_dataset[neg_idx]
            if len(negative_packets) == 0:
                raise ValueError("Empty negative")
            negative = negative_packets[np.random.randint(0, len(negative_packets))]
            
            # Extract events and sample/mask
            anchor_events, anchor_mask = sample_and_mask(anchor['left_events_strip'], N=1024)
            positive_events, positive_mask = sample_and_mask(positive['left_events_strip'], N=1024)
            negative_events, negative_mask = sample_and_mask(negative['left_events_strip'], N=1024)
            
            return {
                'anchor': anchor_events,
                'anchor_mask': anchor_mask,
                'positive': positive_events,
                'positive_mask': positive_mask,
                'negative': negative_events,
                'negative_mask': negative_mask
            }
        
        except Exception as e:
            # Fallback: return zeros (skip bad triplets)
            return {
                'anchor': np.zeros((1024, 4), dtype=np.float32),
                'anchor_mask': np.ones((1024,), dtype=np.float32),
                'positive': np.zeros((1024, 4), dtype=np.float32),
                'positive_mask': np.ones((1024,), dtype=np.float32),
                'negative': np.zeros((1024, 4), dtype=np.float32),
                'negative_mask': np.ones((1024,), dtype=np.float32)
            }
    
class TripletLoss(nn.Module):
    """
    Triplet loss with online hard mining
    """

    def __init__(self, margin=0.5, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        losses = F.relu(pos_dist - neg_dist + self.margin)

        if self.mining == 'hard':
            hard_triplets = losses > 0
            if hard_triplets.sum() > 0:
                return losses[hard_triplets].mean()
            else:
                return losses.mean()
        elif self.mining == 'semi-hard':
            semi_hard = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            if semi_hard.sum() > 0:
                return losses[semi_hard].mean()
            else:
                return losses.mean()  
        else:
            return losses.mean()
        
class DescriptorLoss(nn.Module):
    """
    Combined Loss for descriptor learning
    """

    def __init__(self, triplet_margin=0.5, sparsity_weight=0.1, uncertainty_weight=0.1):
        super(DescriptorLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=triplet_margin, mining='hard')
        self.sparsity_weight = sparsity_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(self, descriptors_a, keypoints_a, uncertainty_a, descriptors_p, keypoints_p, uncertainty_p, descriptors_n, keypoints_n, uncertainty_n):
        B, D, N = descriptors_a.shape

        weights_a = F.softmax(keypoints_a.squeeze(1), dim=1)
        weights_p = F.softmax(keypoints_p.squeeze(1), dim=1)
        weights_n = F.softmax(keypoints_n.squeeze(1), dim=1)

        desc_a = torch.sum(descriptors_a * weights_a.unsqueeze(1), dim=2)
        desc_p = torch.sum(descriptors_p * weights_p.unsqueeze(1), dim=2)
        desc_n = torch.sum(descriptors_n * weights_n.unsqueeze(1), dim=2)

        triplet = self.triplet_loss(desc_a, desc_p, desc_n)
        sparsity = (keypoints_a.mean() + keypoints_p.mean() + keypoints_n.mean()) / 3.0
        uncertainty_loss = uncertainty_a.mean() + uncertainty_p.mean() - uncertainty_n.mean()

        total_loss = triplet + self.sparsity_weight * sparsity + self.uncertainty_weight * uncertainty_loss

        return {
            'total': total_loss,
            'triplet': triplet.item(),
            'sparsity': sparsity.item(),
            'uncertainty': uncertainty_loss.item()
        }      
    
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, use_mixed_precision=True):
    """Train for one epoch with mixed precision (3. Mixed precision)"""
    model.train()

    losses = {'total': [], 'triplet': [], 'sparsity': [], 'uncertainty':[]}
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        anchor = batch['anchor'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive = batch['positive'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        negative = batch['negative'].to(device)
        negative_mask = batch['negative_mask'].to(device)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                desc_a, kp_a, unc_a = model(anchor, anchor_mask)
                desc_p, kp_p, unc_p = model(positive, positive_mask)
                desc_n, kp_n, unc_n = model(negative, negative_mask)
                loss_dict = criterion(desc_a, kp_a, unc_a, desc_p, kp_p, unc_p, desc_n, kp_n, unc_n)
            
            scaler.scale(loss_dict['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            desc_a, kp_a, unc_a = model(anchor, anchor_mask)
            desc_p, kp_p, unc_p = model(positive, positive_mask)
            desc_n, kp_n, unc_n = model(negative, negative_mask)
            loss_dict = criterion(desc_a, kp_a, unc_a, desc_p, kp_p, unc_p, desc_n, kp_n, unc_n)
            
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k, v in loss_dict.items():
            if k == 'total':
                losses[k].append(v.item())
            else:
                losses[k].append(v)
        
        pbar.set_postfix({
            'loss': f"{losses['total'][-1]:.4f}",
            'triplet': f"{losses['triplet'][-1]:.4f}"
        })

    return {k: np.mean(v) for k, v in losses.items()}

def validate(model, dataloader, criterion, device, use_mixed_precision=True):
    """Validate the model with mixed precision"""
    model.eval()
    losses = {'total': [], 'triplet': [], 'sparsity': [], 'uncertainty': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            anchor = batch['anchor'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            positive = batch['positive'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            negative = batch['negative'].to(device)
            negative_mask = batch['negative_mask'].to(device)

            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    desc_a, kp_a, unc_a = model(anchor, anchor_mask)
                    desc_p, kp_p, unc_p = model(positive, positive_mask)
                    desc_n, kp_n, unc_n = model(negative, negative_mask)
                    loss_dict = criterion(desc_a, kp_a, unc_a, desc_p, kp_p, unc_p, desc_n, kp_n, unc_n)
            else:
                desc_a, kp_a, unc_a = model(anchor, anchor_mask)
                desc_p, kp_p, unc_p = model(positive, positive_mask)
                desc_n, kp_n, unc_n = model(negative, negative_mask)
                loss_dict = criterion(desc_a, kp_a, unc_a, desc_p, kp_p, unc_p, desc_n, kp_n, unc_n)

            for k, v in loss_dict.items():
                if k == 'total':
                    losses[k].append(v.item())
                else:
                    losses[k].append(v)

    return {k: np.mean(v) for k, v in losses.items()}

def async_checkpoint_save(checkpoint_path, checkpoint_data, save_queue):
    """Save checkpoint in background thread (5. Async saving)"""
    def _save():
        try:
            torch.save(checkpoint_data, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    save_queue.put(_save)

def checkpoint_saver_worker(save_queue):
    """Worker thread for async checkpoint saving"""
    while True:
        try:
            save_fn = save_queue.get(timeout=1)
            if save_fn is None:
                break
            save_fn()
        except:
            continue

def train(dataset_root, output_dir='./checkpoints', epochs=50, batch_size=8, lr=0.001, use_mixed_precision=True):
    """Main training loop optimized for vast.ai"""

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_mixed_precision}")

    # Load dataset
    print("Loading dataset...")
    base_dataset = event_extractor(dataset_root)

    # Create single triplet dataset (pre-cached once)
    dataset = TripletEventDataset(base_dataset)

    # Split into train/val
    n_valid = len(dataset.triplets)
    n_train = int(0.9 * n_valid)
    indices = np.random.permutation(n_valid)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Dataloaders (2. Optimized for vast.ai)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=12,  # Increased for dedicated GPU
        pin_memory=True,
        prefetch_factor=4,  # More prefetching
        persistent_workers=True  # Avoid worker restart overhead
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"Using num_workers=6 (optimized for vast.ai RTX 3090)")

    # Model
    model = PEVSLAM().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = DescriptorLoss(triplet_margin=0.5, sparsity_weight=0.1, uncertainty_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup async checkpoint saving (5. Async saving)
    save_queue = Queue()
    saver_thread = Thread(target=checkpoint_saver_worker, args=(save_queue,), daemon=True)
    saver_thread.start()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, use_mixed_precision)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_mixed_precision)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"\nTrain Loss: {train_loss['total']:.4f} (triplet: {train_loss['triplet']:.4f})")
        print(f"Val Loss: {val_loss['total']:.4f} (triplet: {val_loss['triplet']:.4f})")

        # Checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        # Save latest (async)
        latest_path = os.path.join(output_dir, 'latest.pth')
        async_checkpoint_save(latest_path, checkpoint, save_queue)

        # Save best model (async)
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            best_path = os.path.join(output_dir, 'best.pth')
            async_checkpoint_save(best_path, checkpoint, save_queue)
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        # Save every 10 epochs (async)
        if epoch % 10 == 0:
            epoch_path = os.path.join(output_dir, f'epoch_{epoch}.pth')
            async_checkpoint_save(epoch_path, checkpoint, save_queue)

    # Cleanup
    save_queue.put(None)
    saver_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        "-d",
        default="/home/adarsh/PEVSLAM/dataset/dataset/train",
        type=str,
        help="The dataset root directory containing sequences"
    )

    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        type=str,
        help="Directory to store the checkpoints"
    )

    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs to train"
    )

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size for training"
    )

    parser.add_argument(
        "--learning-rate",
        "-lr",
        default=0.001,
        type=float,
        help="Learning rate"
    )

    parser.add_argument(
        "--mixed-precision",
        default=True,
        type=bool,
        help="Use mixed precision (FP16)"
    )

    args = parser.parse_args()

    train(
        dataset_root=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        use_mixed_precision=args.mixed_precision
    )