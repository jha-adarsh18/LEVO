import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import wandb

from scripts.pevslam_net import PEVSLAM
from utils.loadevents import event_extractor
from utils.sample_and_mask import sample_and_mask

class TripletEventDataset(Dataset):
    """
    Dataset that generates triplets (anchor, positive, negative) for descriptor learning
    
    Positives: Temporal neighbors (t, t+1, t+2)
    Negatives: Far away Packets (t, t+10+)
    """

    def __init__(self, base_dataset, temporal_positive_range=3, temporal_negative_threshold = 3):
        self.base_dataset = base_dataset
        self.pos_range = temporal_positive_range
        self.neg_theshold = temporal_negative_threshold

        # build sequence index for efficient sampling
        self.sequence_packets = self.build_sequence_index()

    def build_sequence_index(self):
        """Group packets by sequence for efficient triplet sampling"""
        sequence_packets = {}

        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset.flat_samples[idx]
            seq_idx = sample['seq_idx']

            if seq_idx not in sequence_packets:
                sequence_packets[seq_idx] = []

            sequence_packets[seq_idx].append(idx)

        return sequence_packets
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        anchor_packets = self.base_dataset[idx]

        # find sequence and position
        seq_idx, _, _ = self.base_dataset.time_samples[idx]
        sequence_indices = self.sequence_packets[seq_idx]
        anchor_position = sequence_indices.index(idx)

        # sample positive (temporal neighbor)
        pos_offset = np.random.randint(1, self.pos_range + 1)
        pos_idx_position = min(anchor_position + pos_offset, len(sequence_indices) - 1)
        pos_idx = sequence_indices[pos_idx_position]
        positive_packets = self.base_dataset[pos_idx]

        # sample negative (far away)
        valid_negatives = [
            sequence_indices[i] for i in range(len(sequence_indices))
            if abs(i - anchor_position) >= self.neg_theshold
        ]

        if len(valid_negatives) == 0:
            # fallback: sample from different sequence
            other_sequences = [s for s in self.sequence_packets.keys() if s != seq_idx]
            if len(other_sequences) > 0:
                other_seq = np.random.choice(other_sequences)
                neg_idx = np.random.choice(self.sequence_packets[other_seq])
            else:
                neg_idx = idx # edge case: use anchor itself
        else:
            neg_idx = np.random.choice(valid_negatives)

        negative_packets = self.base_dataset[neg_idx]

        # each packet list contains multiple event chunks, randomly select onoe from each
        anchor = np.random.choice(anchor_packets)
        positive = np.random.choice(positive_packets)
        negative = np.random.choice(negative_packets)

        # extract events and sample/mask
        anchor_events, anchor_mask = sample_and_mask(anchor['left_events_strip'], N=1024)
        positive_events, positive_mask = sample_and_mask(positive['left_events_strip'], N=1204)
        negative_events, negative_mask = sample_and_mask(negative['left_events_strip'], N=1024)

        return {
            'anchor': anchor_events,
            'anchor_mask': anchor_mask,
            'positive': positive_events,
            'positive_mask': positive_mask,
            'negative': negative_events,
            'negative_mask': negative_mask
        }
    
class TripletLoss(nn.Module):
    """
    Triplet loss with online hard mining
    L = max(0, ||f_a - f_p||^2 - ||f_a - f_n||^2 + margin)
    """

    def __init__(self, margin=0.5, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining # 'hard', 'semi-hard', or 'all'

    def forward(self, anchor, positive, negative):
        # compute distances 
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Basic triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)

        if self.mining == 'hard':
            # only backprop hardest triplets (positive but margin violated)
            hard_triplets = losses > 0
            if hard_triplets.sum() > 0:
                return losses[hard_triplets].mean()
            else:
                return losses.mean()
        elif self.mining == 'semi-hard':
            # semi-hard: pos_dist < neg_dist < pos_dist + margin
            semi_hard = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            if semi_hard.sum() > 0:
                return losses[semi_hard].mean()
            else:
                return losses.mean()  
        else:
            # all triplets
            return losses.mean()
        
class DescriptorLoss(nn.Module):
    """
    Combined Loss for descriptor learning:
    1. Triplet loss (main)
    2. Keypoint Sparsity Loss (encourage selecting fewer keypoints)
    3. Uncertainty calibration loss (match should be certain if correct)
    """

    def __init__(self, triplet_margin=0.5, sparsity_weight=0.1, uncertainty_weight=0.1):
        super(DescriptorLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=triplet_margin, mining='hard')
        self.sparsity_weight = sparsity_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(self, descriptors_a, keypoints_a, uncertainty_a, descriptors_p, keypoints_p, uncertainty_p, descriptors_n, keypoints_n, uncertainty_n):
        B, D, N = descriptors_a.shape

        # Global descriptors for each packet (max pooling over keypoints)
        # weighted by keypoint scores
        weights_a = F.softmax(keypoints_a.squeeze(1), dim=1) # [B, N]
        weights_p = F.softmax(keypoints_p.squeeze(1), dim=1)
        weights_n = F.softmax(keypoints_n.squeeze(1), dim=1)

        desc_a = torch.sum(descriptors_a * weights_a.unsqueeze(1), dim=2) # [B, D]
        desc_p = torch.sum(descriptors_p * weights_p.unsqueeze(1), dim=2)
        desc_n = torch.sum(descriptors_n * weights_n.unsqueeze(1), dim=2)

        # triplet loss
        triplet = self.triplet_loss(desc_a, desc_p, desc_n)

        # keypoint sparsity loss
        # L1 penalty on keypoint scores
        sparsity = (keypoints_a.mean() + keypoints_p.mean() + keypoints_n.mean()) / 3.0

        # uncertainty loss: positives should be certain, negatives uncertain
        # lower uncertainty for anchor-positive matches
        uncertainty_loss = uncertainty_a.mean() + uncertainty_p.mean() - uncertainty_n.mean()

        total_loss = triplet + self.sparsity_weight * sparsity + self.uncertainty_weight * uncertainty_loss

        return {
            'total': total_loss,
            'triplet': triplet.item(),
            'sparsity': sparsity.item(),
            'uncertainty': uncertainty_loss.item()
        }      
    
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """train for one epoch"""
    model.train()

    losses = {'total': [], 'triplet': [], 'sparsity': [], 'uncertainty':[]}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Move to device
        anchor = batch['anchor'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive = batch['positive'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        negative = batch['negative'].to(device)
        negative_mask = batch['negative_mask'].to(device)

        # forward pass
        desc_a, kp_a, unc_a = model(anchor, anchor_mask)
        desc_p, kp_p, unc_p = model(positive, positive_mask)
        desc_n, kp_n, unc_n = model(negative, negative_mask)

        # compute loss
        loss_dict = criterion(desc_a, kp_a, unc_a, desc_p, kp_p, unc_p, desc_n, kp_n, unc_n)

        # backward pass
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # log
        for k, v in loss_dict.items():
            if k == 'total':
                losses[k].append(v.item())
            else:
                losses[k].append(v)
        
        pbar.set_postfix({
            'loss': f"{losses['total'][-1]:.4f}",
            'triplet': f"{losses['triplet'][-1]:.4f}"
        })

    wandb.log({
        "train/total_loss": np.mean(losses['total']),
        "train/triplet_loss": np.mean(losses['triplet']),
        "train/sparsity_loss": np.mean(losses['sparsity']),
        "train/uncertainty_loss": np.mean(losses['uncertainty']),
        "epoch": epoch
    })

    return {k: np.mean(v) for k, v in losses.items()}

def validate(model, dataloader, criterion, device):
    """Validate the model"""

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

            desc_a, kp_a, unc_a = model(anchor, anchor_mask)
            desc_p, kp_p, unc_p = model(positive, positive_mask)
            desc_n, kp_n, unc_n = model(negative, negative_mask)

            loss_dict = criterion(desc_a, kp_a, unc_a, desc_p, kp_p, unc_p, desc_n, kp_n, unc_n)

            for k, v in loss_dict.items():
                if k == 'total':
                    losses[k].append(v.item())
                else:
                    losses[k].append(v)

    wandb.log({
        "val/total_loss": np.mean(losses['total']),
        "val/triplet_loss": np.mean(losses['triplet']),
        "val/sparsity_loss": np.mean(losses['sparsity']),
        "val/uncertainty_loss": np.mean(losses['uncertainty'])
    })

    return {k: np.mean(v) for k, v in losses.items()}

def train(dataset_root, output_dir='./checkpoints', epochs=50, batch_size=8, lr=0.001):
    """main training loop"""

    # setup
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    # load dataset
    print("Loading dataset...")
    base_dataset = event_extractor(dataset_root)

    # split into train/val (90/ 10 ) split
    n_samples = len(base_dataset)
    n_train = int(0.9 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # create triplet datasets
    train_dataset = TripletEventDataset(base_dataset)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    val_dataset = TripletEventDataset(base_dataset)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_indices)}, Val samples:{len(val_indices)}")

    # model
    model = PEVSLAM().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # loss and optimizer
    criterion = DescriptorLoss(triplet_margin=0.5, sparsity_weight=0.1, uncertainty_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Initialize wandb
    wandb.init(
        project="PEVSLAM",
        config={
            'epochs': epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "PEVSLAM",
            "train_samples": len(train_indices),
            "val_samples": len(val_indices)
        }
    )

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")

        # train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)

        # validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # scheduler step
        scheduler.step()

        # print summary
        print(f"\nTrain Loss: {train_loss['total']:.4f} (triplet: {train_loss['triplet']:.4f})")
        print(f"Val Loss: {val_loss['total']:.4f} (triplet: {val_loss['triplet']:.4f})")

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        } 

        torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))

        # save best model 
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save(checkpoint, os.path.join(output_dir, 'bets.pth'))
            print(f"Saved best Model (val_loss: {best_val_loss:.4f})")

        wandb.log({"best_val_loss": best_val_loss})

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch}.pth'))

        wandb.finish()

if __name__ == "__main__":
    # configuration
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        "--d",
        default="/home/adarsh/PEVSLAM/dataset/dataset/train",
        type=str,
        help="The dataset root directory containing sequences"
    )

    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        type=str,
        help="directory to store the checkpoints"
    )

    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs to train the model"
    )

    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="size of the batch for training"
    )

    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=0.001,
        type=int,
        help="learning rate for the model"
    )

    args = parser.parse_args()

    dataset_root = args.dataset
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # start training
    train(
        dataset_root=dataset_root,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate
    )