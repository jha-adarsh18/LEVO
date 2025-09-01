#!/usr/bin/env python3
"""
FIXED: Complete Training Script for Neural ODE Event Pose Prediction
This version fixes the infinite loss issues by properly handling data preparation and loss computation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import traceback

# Import your modules (make sure these are in your Python path)
from neural_ode_event_pose import (
    NeuralODEEventPosePredictor, 
    train_neural_ode_model,
    plot_training_history,
    save_model_for_inference,
    load_checkpoint,
)
from loadevents import event_extractor  # Your dataset
from eventpoints import EventPointNet, custom_collate_fn   # Your PointNet++

def setup_training_args():
    """Parse command line arguments for training"""
    parser = argparse.ArgumentParser(description='Train Neural ODE Event Pose Predictor')
    
    # Multi-GPU parameters
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                       help='Use multiple GPUs if available')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='GPU IDs to use (comma-separated)')
    
    # Dataset parameters
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to your event dataset root directory')
    parser.add_argument('--batch_size', type=int, default=2,  # Very small for tiny dataset
                       help='Batch size for training (per GPU)')
    parser.add_argument('--num_events', type=int, default=1024,
                       help='Number of events per sample (N parameter)')
    parser.add_argument('--num_workers', type=int, default=0,  # Set to 0 for debugging
                       help='Number of dataloader workers')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension size')
    parser.add_argument('--ode_hidden_dim', type=int, default=512,
                       help='Hidden dimension for ODE function')
    parser.add_argument('--use_stereo', action='store_true', default=True,
                       help='Use stereo event data')
    parser.add_argument('--rtol', type=float, default=1e-3,
                       help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=1e-4,
                       help='Absolute tolerance for ODE solver')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=3,  # Very small for debugging
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,  # FIXED: Lower learning rate
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--num_time_steps', type=int, default=10,
                       help='Number of time steps for ODE integration')
    
    # Loss weights - FIXED: Better balanced weights
    parser.add_argument('--position_weight', type=float, default=1.0,
                       help='Weight for position loss')
    parser.add_argument('--orientation_weight', type=float, default=0.1,
                       help='Weight for orientation loss')
    
    # Saving parameters
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/neural_ode_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=1,  # Save every epoch for debugging
                       help='Save checkpoint every N epochs')
    parser.add_argument('--save_best', action='store_true', default=True,
                       help='Save best model based on validation loss')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment (auto-generated if None)')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Validation
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--val_every', type=int, default=1,  # Validate every epoch for debugging
                       help='Run validation every N epochs')
    
    return parser.parse_args()

def setup_multi_gpu(args):
    """Setup multi-GPU configuration"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        return torch.device('cpu'), []
    
    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(i.strip()) for i in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    # Filter available GPUs
    available_gpus = []
    for gpu_id in gpu_ids:
        if gpu_id < torch.cuda.device_count():
            available_gpus.append(gpu_id)
            print(f"🚀 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            print(f"   Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    
    if len(available_gpus) == 0:
        print("⚠️  No available GPUs found, using CPU")
        return torch.device('cpu'), []
    
    # Set primary device
    primary_device = torch.device(f'cuda:{available_gpus[0]}')
    torch.cuda.set_device(primary_device)
    
    print(f"🎯 Primary device: {primary_device}")
    print(f"🔗 Using {len(available_gpus)} GPU(s): {available_gpus}")
    
    return primary_device, available_gpus

def create_data_loaders(args, num_gpus):
    """FIXED: Create training and validation data loaders with proper error handling"""
    print("🔄 Loading dataset...")
    
    try:
        # Load your event dataset
        full_dataset = event_extractor(args.dataset_root, N=args.num_events)
        print(f"📊 Total sequences found: {len(full_dataset)}")
        
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty! Check your dataset path and implementation.")
        
        # Test dataset by loading one sample
        print("🔬 Testing dataset loading...")
        test_sample = full_dataset[0]
        print(f"✅ Sample loaded successfully")
        print(f"   Left events shape: {test_sample['left_events_strip'].shape}")
        print(f"   Right events shape: {test_sample['right_events_strip'].shape}")
        print(f"   Left pose shape: {test_sample['left_pose'].shape}")
        print(f"   Right pose shape: {test_sample['right_pose'].shape}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        traceback.print_exc()
        raise
    
    # Split into train/validation
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Ensure at least one sample in each split
    if train_size == 0:
        train_size = 1
        val_size = len(full_dataset) - 1
    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"🚂 Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Adjust batch size for small dataset
    effective_batch_size = min(args.batch_size, len(train_dataset))
    if args.use_multi_gpu and num_gpus > 1:
        effective_batch_size = min(effective_batch_size * num_gpus, len(train_dataset))
    
    print(f"📦 Effective batch size: {effective_batch_size}")
    print(f"👷 Workers: {args.num_workers}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,  # FIXED: Use proper collate function
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False  # Don't drop incomplete batches with small dataset
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,  # FIXED: Use proper collate function
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"🔍 Train loader length: {len(train_loader)}")
    print(f"🔍 Val loader length: {len(val_loader)}")
    
    # Test data loader
    print("🔬 Testing data loader...")
    try:
        test_batch = next(iter(train_loader))
        print(f"✅ Batch loaded successfully")
        print(f"   Batch keys: {list(test_batch.keys())}")
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        traceback.print_exc()
        raise
    
    return train_loader, val_loader

def wrap_model_for_multi_gpu(model, gpu_ids):
    """Wrap model with DataParallel if multiple GPUs available"""
    if len(gpu_ids) > 1:
        print(f"🔗 Wrapping model with DataParallel for GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print("✅ Model wrapped with DataParallel")
    else:
        print("📱 Using single GPU - no DataParallel wrapping needed")
    
    return model

def validate_model(model, val_loader, device, args, is_multi_gpu=False):
    """FIXED: Run validation with proper error handling"""
    from neural_ode_event_pose import PoseLoss
    
    model.eval()
    pose_loss_fn = PoseLoss(
        position_weight=args.position_weight, 
        orientation_weight=args.orientation_weight
    )
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_orient_loss = 0.0
    num_batches = 0
    failed_batches = 0
    
    print(f"🔍 Validation: Processing {len(val_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Prepare batch with FIXED prepare_batch_for_ode
                prepared_batch = prepare_batch_for_ode_fixed(
                    batch, device, num_time_steps=args.num_time_steps
                )
                
                # Forward pass
                outputs = model(
                    prepared_batch['left_events'],
                    prepared_batch['time_steps'],
                    prepared_batch['right_events']
                )
                
                # Compute loss
                loss_dict = pose_loss_fn(outputs['poses'], prepared_batch['gt_poses'])
                
                # Check for valid loss
                if not (torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss'])):
                    total_loss += loss_dict['total_loss'].item()
                    total_pos_loss += loss_dict['position_loss'].item()
                    total_orient_loss += loss_dict['orientation_loss'].item()
                    num_batches += 1
                else:
                    print(f"⚠️  Invalid loss in validation batch {batch_idx}")
                    failed_batches += 1
                
            except Exception as e:
                print(f"❌ Validation batch {batch_idx} error: {e}")
                failed_batches += 1
                continue
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_pos = total_pos_loss / num_batches
        avg_orient = total_orient_loss / num_batches
        
        print(f"📊 Validation Results:")
        print(f"   Average Loss: {avg_loss:.6f}")
        print(f"   Position Loss: {avg_pos:.6f}")
        print(f"   Orientation Loss: {avg_orient:.6f}")
        print(f"   Successful batches: {num_batches}")
        print(f"   Failed batches: {failed_batches}")
        
        return avg_loss, avg_pos, avg_orient
    else:
        print("❌ No validation batches processed successfully!")
        return float('inf'), float('inf'), float('inf')

def prepare_batch_for_ode_fixed(batch, device, num_time_steps):
    """
    FIXED: Prepare batch data for Neural ODE with proper pose handling
    This fixes the main cause of infinite loss by ensuring proper pose format
    """
    print(f"🔧 prepare_batch_for_ode_fixed:")
    print(f"   Input batch keys: {list(batch.keys())}")
    
    # Move data to device
    left_events = batch['left_events'].to(device)  # [B, N, 4]
    right_events = batch['right_events'].to(device)  # [B, N, 4] 
    left_poses = batch['left_poses'].to(device)  # [B, 7]
    right_poses = batch['right_poses'].to(device)  # [B, 7]
    
    batch_size = left_events.shape[0]
    
    print(f"   left_events: {left_events.shape}")
    print(f"   right_events: {right_events.shape}")
    print(f"   left_poses: {left_poses.shape}")
    print(f"   right_poses: {right_poses.shape}")
    
    # Create time steps for ODE integration
    time_steps = torch.linspace(0, 1, num_time_steps, device=device)  # [T]
    time_steps = time_steps.unsqueeze(0).repeat(batch_size, 1)  # [B, T]
    
    # FIXED: Proper ground truth pose handling
    # Your dataset provides poses as [x, y, z, qx, qy, qz, qw] format
    # We need to create ground truth poses for each time step
    
    # For simplicity, we'll use left poses as primary ground truth
    # In a real scenario, you might want to interpolate poses across time steps
    gt_poses = left_poses.unsqueeze(1).repeat(1, num_time_steps, 1)  # [B, T, 7]
    
    print(f"   time_steps: {time_steps.shape}")
    print(f"   gt_poses: {gt_poses.shape}")
    
    # Validate poses - check for NaN/Inf and reasonable ranges
    if torch.isnan(gt_poses).any() or torch.isinf(gt_poses).any():
        print("⚠️  WARNING: NaN/Inf detected in ground truth poses!")
        print(f"   NaN count: {torch.isnan(gt_poses).sum()}")
        print(f"   Inf count: {torch.isinf(gt_poses).sum()}")
        
        # Replace NaN/Inf with reasonable defaults
        gt_poses = torch.where(torch.isnan(gt_poses) | torch.isinf(gt_poses), 
                              torch.zeros_like(gt_poses), gt_poses)
        # Set default quaternion to [0, 0, 0, 1] for invalid entries
        gt_poses[:, :, 6] = torch.where(gt_poses[:, :, 6] == 0, 
                                       torch.ones_like(gt_poses[:, :, 6]), gt_poses[:, :, 6])
    
    # Normalize quaternions to ensure they're valid
    quat_norm = torch.norm(gt_poses[:, :, 3:7], dim=-1, keepdim=True)  # [B, T, 1]
    quat_norm = torch.clamp(quat_norm, min=1e-6)  # Avoid division by zero
    gt_poses[:, :, 3:7] = gt_poses[:, :, 3:7] / quat_norm  # Normalize quaternions
    
    print(f"   Position range: [{gt_poses[:, :, :3].min():.3f}, {gt_poses[:, :, :3].max():.3f}]")
    print(f"   Quaternion norm range: [{torch.norm(gt_poses[:, :, 3:7], dim=-1).min():.3f}, {torch.norm(gt_poses[:, :, 3:7], dim=-1).max():.3f}]")
    
    prepared_batch = {
        'left_events': left_events,
        'right_events': right_events,
        'time_steps': time_steps,
        'gt_poses': gt_poses,
        'batch_size': batch_size
    }
    
    print(f"✅ Batch prepared successfully")
    return prepared_batch

def save_multi_gpu_checkpoint(model, optimizer, scheduler, epoch, loss, history, path, is_multi_gpu=False):
    """Save checkpoint handling DataParallel models"""
    # Extract the actual model from DataParallel wrapper if needed
    model_state_dict = model.module.state_dict() if is_multi_gpu else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'training_history': history,
        'is_multi_gpu': is_multi_gpu
    }
    
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")

def train_neural_ode_model_fixed(model, train_loader, val_loader, 
                                num_epochs, lr, device, save_dir,
                                save_every, save_best, val_every, args,
                                is_multi_gpu, num_gpus):
    """
    FIXED: Training function with comprehensive error handling and loss debugging
    """
    from neural_ode_event_pose import PoseLoss
    
    # Setup optimizer with gradient clipping
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    pose_loss_fn = PoseLoss(
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight
    )
    
    # Training tracking
    best_val_loss = float('inf')
    training_history = {
        'epoch': [], 'train_loss': [], 'train_pos_loss': [], 'train_orient_loss': [],
        'val_loss': [], 'val_pos_loss': [], 'val_orient_loss': [], 'lr': []
    }
    
    print(f"📈 Starting training with learning rate: {lr:.6f}")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_pos_loss = 0.0
        train_orient_loss = 0.0
        num_train_batches = 0
        failed_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Zero gradients
                optimizer.zero_grad()
                
                # Prepare batch with fixed function
                prepared_batch = prepare_batch_for_ode_fixed(batch, device, args.num_time_steps)
                
                # Validate input data
                for key, tensor in prepared_batch.items():
                    if isinstance(tensor, torch.Tensor):
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            print(f"⚠️  Invalid values in {key}, skipping batch {batch_idx}")
                            failed_batches += 1
                            continue
                
                # Forward pass
                try:
                    outputs = model(
                        prepared_batch['left_events'],
                        prepared_batch['time_steps'],
                        prepared_batch['right_events']
                    )
                except Exception as e:
                    print(f"❌ Forward pass error on batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue
                
                # Validate model outputs
                if 'poses' not in outputs:
                    print(f"⚠️  Model output missing 'poses' key for batch {batch_idx}")
                    failed_batches += 1
                    continue
                
                if torch.isnan(outputs['poses']).any() or torch.isinf(outputs['poses']).any():
                    print(f"⚠️  Invalid model outputs for batch {batch_idx}")
                    failed_batches += 1
                    continue
                
                # Compute loss
                try:
                    loss_dict = pose_loss_fn(outputs['poses'], prepared_batch['gt_poses'])
                    loss = loss_dict['total_loss']
                except Exception as e:
                    print(f"❌ Loss computation error on batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue
                
                # Validate loss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    print(f"⚠️  Invalid loss detected: {loss.item()}")
                    print(f"   Position loss: {loss_dict['position_loss'].item()}")
                    print(f"   Orientation loss: {loss_dict['orientation_loss'].item()}")
                    failed_batches += 1
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check gradients
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                if total_norm > 10:
                    print(f"⚠️  Large gradient norm: {total_norm:.2f}")
                
                optimizer.step()
                
                # Successfully processed batch
                train_loss += loss.item()
                train_pos_loss += loss_dict['position_loss'].item()
                train_orient_loss += loss_dict['orientation_loss'].item()
                num_train_batches += 1
                
                # Print progress
                if batch_idx % max(1, len(train_loader) // 3) == 0:
                    print(f"   Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.6f}")
                
            except Exception as e:
                print(f"❌ Unexpected error on training batch {batch_idx}: {e}")
                failed_batches += 1
                continue
        
        # Calculate epoch averages
        if num_train_batches > 0:
            avg_train_loss = train_loss / num_train_batches
            avg_train_pos = train_pos_loss / num_train_batches
            avg_train_orient = train_orient_loss / num_train_batches
        else:
            print("🛑 No training batches succeeded - stopping training")
            break
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Training Summary for Epoch {epoch+1}:")
        print(f"   Successful batches: {num_train_batches}/{len(train_loader)}")
        print(f"   Failed batches: {failed_batches}")
        print(f"   Average train loss: {avg_train_loss:.6f}")
        print(f"   Average position loss: {avg_train_pos:.6f}")
        print(f"   Average orientation loss: {avg_train_orient:.6f}")
        
        # Validation phase
        avg_val_loss = avg_val_pos = avg_val_orient = float('inf')
        if (epoch + 1) % val_every == 0:
            print("🔍 Running validation...")
            avg_val_loss, avg_val_pos, avg_val_orient = validate_model(
                model, val_loader, device, args, is_multi_gpu
            )
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Update training history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_pos_loss'].append(avg_train_pos)
        training_history['train_orient_loss'].append(avg_train_orient)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_pos_loss'].append(avg_val_pos)
        training_history['val_orient_loss'].append(avg_val_orient)
        training_history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.6f}")
        print(f"   Val Loss: {avg_val_loss:.6f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if save_best and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_multi_gpu_checkpoint(model, optimizer, scheduler, epoch + 1, avg_val_loss,
                                    training_history, best_model_path, is_multi_gpu)
            print(f"✅ New best model saved! Val Loss: {best_val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_multi_gpu_checkpoint(model, optimizer, scheduler, epoch + 1, avg_train_loss,
                                    training_history, checkpoint_path, is_multi_gpu)
    
    return training_history

def main_with_debugging():
    """FIXED: Main function with comprehensive error handling"""
    args = setup_training_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Setup multi-GPU
    device, gpu_ids = setup_multi_gpu(args)
    num_gpus = len(gpu_ids)
    is_multi_gpu = num_gpus > 1 and args.use_multi_gpu
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_suffix = f"_{num_gpus}gpu" if is_multi_gpu else "_1gpu"
        args.experiment_name = f"neural_ode_experiment_{timestamp}{gpu_suffix}"
    
    print(f"🧪 Experiment: {args.experiment_name}")
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Save directory created: {save_dir}")
    except OSError as e:
        print(f"❌ Error creating save directory: {e}")
        alt_save_dir = f"/tmp/neural_ode_checkpoints/{args.experiment_name}"
        try:
            os.makedirs(alt_save_dir, exist_ok=True)
            save_dir = alt_save_dir
            print(f"📁 Using alternative directory: {save_dir}")
        except OSError:
            raise RuntimeError("Cannot create writable directory for checkpoints")
    
    # Save training configuration
    import json
    training_config = vars(args).copy()
    training_config['gpu_ids_used'] = gpu_ids
    training_config['num_gpus_used'] = num_gpus
    training_config['is_multi_gpu'] = is_multi_gpu
    training_config['actual_save_dir'] = save_dir
    
    with open(os.path.join(save_dir, 'training_args.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(args, num_gpus)
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        traceback.print_exc()
        return
    
    # Initialize model
    print("🔧 Initializing model...")
    model = NeuralODEEventPosePredictor(
        latent_dim=args.latent_dim,
        ode_hidden_dim=args.ode_hidden_dim,
        use_stereo=args.use_stereo,
        rtol=args.rtol,
        atol=args.atol
    )
    
    # Move to device and wrap for multi-GPU
    model = model.to(device)
    if is_multi_gpu:
        model = wrap_model_for_multi_gpu(model, gpu_ids)
    
    # Count parameters
    param_model = model.module if is_multi_gpu else model
    total_params = sum(p.numel() for p in param_model.parameters())
    trainable_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    if is_multi_gpu:
        print(f"   Parameters replicated across {num_gpus} GPUs")
    
    # Start training
    print("🚂 Starting training...")
    print("=" * 60)
    
    try:
        training_history = train_neural_ode_model_fixed(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            save_dir=save_dir,
            save_every=args.save_every,
            save_best=args.save_best,
            val_every=args.val_every,
            args=args,
            is_multi_gpu=is_multi_gpu,
            num_gpus=num_gpus
        )
        
        # Plot and save training history
        print("📊 Plotting training history...")
        plot_path = os.path.join(save_dir, 'training_history.png')
        plot_training_history(training_history, save_path=plot_path)
        
        # Save final model for inference
        print("💾 Saving final model...")
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        final_model = model.module if is_multi_gpu else model
        save_model_for_inference(final_model, final_model_path)
        
        print("🎉 Training completed successfully!")
        print(f"📁 Results saved in: {save_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()

def debug_data_sample(dataset, device):
    """
    FIXED: Debug function to analyze a single data sample in detail
    """
    print("🔍 DEBUGGING SINGLE DATA SAMPLE:")
    
    try:
        # Get a sample
        sample = dataset[0]
        print(f"✅ Sample loaded successfully")
        print(f"   Sample keys: {list(sample.keys())}")
        
        # Analyze event data
        left_events = sample['left_events_strip']
        right_events = sample['right_events_strip']
        
        print(f"\n📊 Left Events Analysis:")
        print(f"   Shape: {left_events.shape}")
        print(f"   Dtype: {left_events.dtype}")
        print(f"   X range: [{left_events['x'].min():.3f}, {left_events['x'].max():.3f}]")
        print(f"   Y range: [{left_events['y'].min():.3f}, {left_events['y'].max():.3f}]")
        print(f"   T range: [{left_events['t'].min():.3f}, {left_events['t'].max():.3f}]")
        print(f"   P values: {np.unique(left_events['p'])}")
        
        # Count valid vs padding events
        valid_mask = left_events['x'] >= 0
        valid_count = np.sum(valid_mask)
        padding_count = len(left_events) - valid_count
        print(f"   Valid events: {valid_count}")
        print(f"   Padding events: {padding_count}")
        
        # Analyze poses
        left_pose = sample['left_pose']
        right_pose = sample['right_pose']
        
        print(f"\n📊 Pose Analysis:")
        print(f"   Left pose shape: {left_pose.shape}")
        print(f"   Left pose: {left_pose}")
        print(f"   Position range: [{left_pose[:3].min():.3f}, {left_pose[:3].max():.3f}]")
        print(f"   Quaternion: {left_pose[3:7]}")
        print(f"   Quaternion norm: {np.linalg.norm(left_pose[3:7]):.6f}")
        
        # Test conversion to tensor and collate function
        print(f"\n🔧 Testing collate function:")
        from eventpoints import custom_collate_fn
        batch = custom_collate_fn([sample])
        
        print(f"   Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        
    except Exception as e:
        print(f"❌ Error in data sample debugging: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main_with_debugging()