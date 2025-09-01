import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from typing import List, Dict, Tuple

# Import components
from loadeventdata import extract_events
from pointnet_events import EventPointNet
from Neural_ODE import EndToEndEventSLAM, compute_pose_loss, compute_ode_regularization

class SequentialEventDataset(Dataset):
    """
    Dataset that creates sequences of event chunks for ODE training
    """
    def __init__(self, dataset_root: str, sequence_length: int = 10, 
                 duration_us: int = 5, max_events: int = 2048, overlap: float = 0.5):
        self.sequence_length = sequence_length
        self.duration_us = duration_us
        self.max_events = max_events
        self.overlap = overlap
        
        # Load base dataset
        self.base_dataset = extract_events(dataset_root, duration_us=duration_us, max_events=max_events)
        
        # Create sequences by grouping consecutive chunks from same scenario
        self.sequences = self._create_sequences()
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def _create_sequences(self) -> List[Dict]:
        """Group dataset items into sequences"""
        sequences = []
        
        # Group by scenario for now (in practice, you'd load multiple chunks per sequence)
        scenario_groups = {}
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            scenario_key = f"{sample['sequence_info']['scenario']}_{sample['sequence_info']['sequence']}"
            
            if scenario_key not in scenario_groups:
                scenario_groups[scenario_key] = []
            scenario_groups[scenario_key].append(i)
        
        # Create sequences from each scenario group
        for scenario, indices in scenario_groups.items():
            if len(indices) >= self.sequence_length:
                # For now, just take the first sequence_length items
                # In practice, you'd extract multiple overlapping windows
                sequences.append({
                    'indices': indices[:self.sequence_length],
                    'scenario': scenario
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        indices = sequence_info['indices']
        
        # Collect sequence data
        left_events_seq = []
        right_events_seq = []
        poses_seq = []
        timestamps = []
        
        for i in indices:
            sample = self.base_dataset.get_pointnet_sample(i)
            
            left_events_seq.append(sample['left_tensor'])
            right_events_seq.append(sample['right_tensor'])
            poses_seq.append(sample['pose'])
            timestamps.append(sample['metadata']['t_start'])
        
        # Stack into tensors
        left_events_seq = torch.stack(left_events_seq)  # [seq_len, max_events, 4]
        right_events_seq = torch.stack(right_events_seq)  # [seq_len, max_events, 4]
        poses_seq = torch.stack(poses_seq)  # [seq_len, 7]
        timestamps = torch.tensor(timestamps)  # [seq_len]
        
        # Compute time deltas
        time_deltas = torch.zeros_like(timestamps)
        time_deltas[1:] = timestamps[1:] - timestamps[:-1]
        
        return {
            'left_events': left_events_seq,
            'right_events': right_events_seq,
            'poses': poses_seq,
            'time_deltas': time_deltas,
            'scenario': sequence_info['scenario']
        }

def collate_sequential_batch(batch):
    """Collate function for sequential data"""
    left_batch = torch.stack([item['left_events'] for item in batch])  # [B, seq_len, max_events, 4]
    right_batch = torch.stack([item['right_events'] for item in batch])  # [B, seq_len, max_events, 4]
    poses_batch = torch.stack([item['poses'] for item in batch])  # [B, seq_len, 7]
    time_deltas_batch = torch.stack([item['time_deltas'] for item in batch])  # [B, seq_len]
    
    return {
        'left_events': left_batch,
        'right_events': right_batch,
        'poses': poses_batch,
        'time_deltas': time_deltas_batch
    }

def train_end_to_end(dataset_root: str, model_save_path: str = 'event_slam_end2end.pth',
                    sequence_length: int = 8, latent_dim: int = 256, 
                    epochs: int = 50, batch_size: int = 4, lr: float = 1e-4):
    """
    Train the complete end-to-end Event SLAM system
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create sequential dataset
    dataset = SequentialEventDataset(
        dataset_root=dataset_root,
        sequence_length=sequence_length,
        duration_us=5,
        max_events=2048
    )
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequential_batch,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize components
    pointnet = EventPointNet(latent_dim=latent_dim, use_stereo=True)
    
    # Complete end-to-end model
    model = EndToEndEventSLAM(
        pointnet_model=pointnet,
        latent_dim=latent_dim,
        ode_hidden_dim=512
    ).to(device)
    
    # Optimizer with different learning rates for different components
    pointnet_params = list(model.pointnet.parameters())
    ode_params = list(model.latent_ode.parameters())
    pose_params = list(model.pose_regressor.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': pointnet_params, 'lr': lr * 0.5},  # Lower LR for PointNet++
        {'params': ode_params, 'lr': lr},             # Standard LR for ODE
        {'params': pose_params, 'lr': lr * 2.0}       # Higher LR for pose regressor
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Loss weights
    pose_weight = 1.0
    ode_reg_weight = 0.01
    consistency_weight = 0.1
    
    print(f"Training end-to-end model for {epochs} epochs...")
    print(f"Dataset size: {len(dataset)} sequences")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_pose_loss = 0.0
        total_ode_loss = 0.0
        total_consistency_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            left_events = batch['left_events'].to(device)    # [B, seq_len, max_events, 4]
            right_events = batch['right_events'].to(device)  # [B, seq_len, max_events, 4]
            true_poses = batch['poses'].to(device)           # [B, seq_len, 7]
            time_deltas = batch['time_deltas'].to(device)    # [B, seq_len]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_poses, latent_trajectory = model(left_events, right_events, time_deltas[0])
            
            # Compute losses
            
            # 1. Pose prediction loss
            pose_loss = compute_pose_loss(pred_poses, true_poses, 
                                        translation_weight=1.0, rotation_weight=0.5)
            
            # 2. ODE regularization (smoothness)
            ode_reg_loss = compute_ode_regularization(latent_trajectory, time_deltas[0])
            
            # 3. Consistency loss (predicted trajectory should match PointNet++ features)
            # Extract PointNet++ features directly for comparison
            pointnet_features = []
            batch_size, seq_len = left_events.shape[:2]
            
            for t in range(seq_len):
                with torch.no_grad():
                    feat_t = model.pointnet(left_events[:, t], right_events[:, t])
                    pointnet_features.append(feat_t)
            
            pointnet_sequence = torch.stack(pointnet_features, dim=1)  # [B, seq_len, latent_dim]
            consistency_loss = F.mse_loss(latent_trajectory, pointnet_sequence)
            
            # Total loss
            total_batch_loss = (pose_weight * pose_loss + 
                              ode_reg_weight * ode_reg_loss + 
                              consistency_weight * consistency_loss)
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_pose_loss += pose_loss.item()
            total_ode_loss += ode_reg_loss.item()
            total_consistency_loss += consistency_loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Total: {total_batch_loss.item():.6f}, '
                      f'Pose: {pose_loss.item():.6f}, '
                      f'ODE: {ode_reg_loss.item():.6f}, '
                      f'Consistency: {consistency_loss.item():.6f}')
        
        # Epoch statistics
        avg_total_loss = total_loss / num_batches
        avg_pose_loss = total_pose_loss / num_batches
        avg_ode_loss = total_ode_loss / num_batches
        avg_consistency_loss = total_consistency_loss / num_batches
        
        print(f'Epoch {epoch+1}/{epochs} Summary:')
        print(f'  Average Total Loss: {avg_total_loss:.6f}')
        print(f'  Average Pose Loss: {avg_pose_loss:.6f}')
        print(f'  Average ODE Loss: {avg_ode_loss:.6f}')
        print(f'  Average Consistency Loss: {avg_consistency_loss:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        scheduler.step()
        
        # Save best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'config': {
                    'latent_dim': latent_dim,
                    'sequence_length': sequence_length,
                    'ode_hidden_dim': 512
                }
            }, model_save_path)
            print(f'  New best model saved to {model_save_path}')
        
        print('-' * 80)
    
    print("Training completed!")
    return model

def evaluate_model(model_path: str, dataset_root: str, sequence_length: int = 8):
    """
    Evaluate the trained end-to-end model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate model
    pointnet = EventPointNet(latent_dim=config['latent_dim'], use_stereo=True)
    model = EndToEndEventSLAM(
        pointnet_model=pointnet,
        latent_dim=config['latent_dim'],
        ode_hidden_dim=config['ode_hidden_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best training loss: {checkpoint['loss']:.6f}")
    
    # Create test dataset
    test_dataset = SequentialEventDataset(
        dataset_root=dataset_root,
        sequence_length=sequence_length,
        duration_us=5,
        max_events=2048
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_sequential_batch
    )
    
    print(f"Evaluating on {len(test_dataset)} sequences...")
    
    total_pose_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            left_events = batch['left_events'].to(device)
            right_events = batch['right_events'].to(device)
            true_poses = batch['poses'].to(device)
            time_deltas = batch['time_deltas'].to(device)
            
            # Predict poses
            pred_poses, latent_trajectory = model(left_events, right_events, time_deltas[0])
            
            # Compute pose error
            pose_error = torch.mean(torch.norm(pred_poses - true_poses, dim=-1))
            total_pose_error += pose_error.item()
            total_samples += 1
            
            if batch_idx < 3:  # Print first few examples
                print(f"Sequence {batch_idx + 1}:")
                print(f"  Pose Error: {pose_error.item():.6f}")
                print(f"  True pose (first): {true_poses[0, 0].cpu().numpy()}")
                print(f"  Pred pose (first): {pred_poses[0, 0].cpu().numpy()}")
    
    avg_pose_error = total_pose_error / total_samples
    print(f"\nEvaluation Results:")
    print(f"Average Pose Error: {avg_pose_error:.6f}")
    
    return avg_pose_error

def test_future_prediction(model_path: str, dataset_root: str, 
                          history_length: int = 5, future_steps: int = 3):
    """
    Test the model's ability to predict future poses
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    pointnet = EventPointNet(latent_dim=config['latent_dim'], use_stereo=True)
    model = EndToEndEventSLAM(
        pointnet_model=pointnet,
        latent_dim=config['latent_dim'],
        ode_hidden_dim=config['ode_hidden_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset
    test_dataset = SequentialEventDataset(
        dataset_root=dataset_root,
        sequence_length=history_length + future_steps,
        duration_us=5,
        max_events=2048
    )
    
    print(f"Testing future prediction with {history_length} history → {future_steps} future")
    
    with torch.no_grad():
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            
            # Split into history and future
            left_history = sample['left_events'][:history_length].unsqueeze(0).to(device)
            right_history = sample['right_events'][:history_length].unsqueeze(0).to(device)
            time_deltas_history = sample['time_deltas'][:history_length]
            
            true_future_poses = sample['poses'][history_length:].to(device)
            
            # Create future time points (assuming constant 5μs intervals)
            future_times = torch.arange(1, future_steps + 1).float() * 5e-6  # 5μs intervals
            future_times = future_times.to(device)
            
            # Predict future poses
            pred_future_poses = model.predict_future_poses(
                left_history, right_history, time_deltas_history, future_times
            )
            
            print(f"Future prediction results:")
            for i in range(future_steps):
                true_pose = true_future_poses[i].cpu().numpy()
                pred_pose = pred_future_poses[0, i].cpu().numpy()
                error = np.linalg.norm(true_pose - pred_pose)
                
                print(f"  Step {i+1}: Error = {error:.6f}")
                print(f"    True: {true_pose}")
                print(f"    Pred: {pred_pose}")

def main():
    """
    Main training and evaluation pipeline
    """
    dataset_root = "/home/adarsh/Documents/SRM/dataset/train"
    model_save_path = "event_slam_end2end.pth"
    
    print("="*80)
    print("EVENT-BASED SLAM: END-TO-END TRAINING")
    print("PointNet++ → Neural ODE → Pose Regressor")
    print("="*80)
    
    # Training
    print("\n1. TRAINING PHASE")
    print("-" * 40)
    
    try:
        model = train_end_to_end(
            dataset_root=dataset_root,
            model_save_path=model_save_path,
            sequence_length=8,      # 8 frames per sequence
            latent_dim=256,         # 256-dim latent space
            epochs=30,              # 30 training epochs
            batch_size=2,           # Small batch due to memory constraints
            lr=1e-4                 # Learning rate
        )
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Evaluation
    print("\n2. EVALUATION PHASE")
    print("-" * 40)
    
    try:
        avg_error = evaluate_model(
            model_path=model_save_path,
            dataset_root=dataset_root,
            sequence_length=8
        )
        print(f"✓ Evaluation completed! Average error: {avg_error:.6f}")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
    
    # Future Prediction Test
    print("\n3. FUTURE PREDICTION TEST")
    print("-" * 40)
    
    try:
        test_future_prediction(
            model_path=model_save_path,
            dataset_root=dataset_root,
            history_length=5,
            future_steps=3
        )
        print("✓ Future prediction test completed!")
        
    except Exception as e:
        print(f"✗ Future prediction test failed: {e}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()