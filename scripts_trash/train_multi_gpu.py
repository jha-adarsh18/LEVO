#!/usr/bin/env python3
"""
Complete Training Script for Neural ODE Event Pose Prediction with Multi-GPU Support
Run this script to train your model with your event dataset using DataParallel
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
    custom_collate_fn
)
from loadevents import event_extractor  # Your dataset
from eventpoints import EventPointNet   # Your PointNet++

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
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--num_time_steps', type=int, default=10,
                       help='Number of time steps for ODE integration')
    
    # Loss weights
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
    """Create training and validation data loaders"""
    print("🔄 Loading dataset...")
    
    # Load your event dataset
    full_dataset = event_extractor(args.dataset_root, N=args.num_events)
    print(f"📊 Total sequences found: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty! Check your dataset path and implementation.")
    
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
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False  # Don't drop incomplete batches with small dataset
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"🔍 Train loader length: {len(train_loader)}")
    print(f"🔍 Val loader length: {len(val_loader)}")
    
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
    """Run validation and return average loss"""
    from neural_ode_event_pose import PoseLoss, prepare_batch_for_ode
    
    model.eval()
    pose_loss_fn = PoseLoss(
        position_weight=args.position_weight, 
        orientation_weight=args.orientation_weight
    )
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_orient_loss = 0.0
    num_batches = 0
    
    print(f"🔍 Validation: Processing {len(val_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                print(f"🔍 Validation batch {batch_idx}...")
                
                # Prepare batch
                prepared_batch = prepare_batch_for_ode(
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
                
                total_loss += loss_dict['total_loss'].item()
                total_pos_loss += loss_dict['position_loss'].item()
                total_orient_loss += loss_dict['orientation_loss'].item()
                num_batches += 1
                
                print(f"✅ Validation batch {batch_idx} completed")
                
            except Exception as e:
                print(f"❌ Validation batch {batch_idx} error: {e}")
                print(f"Full traceback:")
                traceback.print_exc()
                continue
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_pos = total_pos_loss / num_batches
        avg_orient = total_orient_loss / num_batches
        
        print(f"📊 Validation Results:")
        print(f"   Average Loss: {avg_loss:.6f}")
        print(f"   Position Loss: {avg_pos:.6f}")
        print(f"   Orientation Loss: {avg_orient:.6f}")
        print(f"   Batches processed: {num_batches}")
        
        return avg_loss, avg_pos, avg_orient
    else:
        print("❌ No validation batches processed successfully!")
        return float('inf'), float('inf'), float('inf')

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

def main_with_debugging():
    """Modified main function with enhanced debugging"""
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
    
    # Create save directory with error handling
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Save directory created: {save_dir}")
    except OSError as e:
        print(f"❌ Error creating save directory: {e}")
        # Try alternative directory
        alt_save_dir = f"/tmp/neural_ode_checkpoints/{args.experiment_name}"
        try:
            os.makedirs(alt_save_dir, exist_ok=True)
            save_dir = alt_save_dir
            print(f"📁 Using alternative directory: {save_dir}")
        except OSError:
            raise RuntimeError("Cannot create writable directory for checkpoints")
    
    # Save training arguments
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
    
    # Move to device first, then wrap with DataParallel
    model = model.to(device)
    if is_multi_gpu:
        model = wrap_model_for_multi_gpu(model, gpu_ids)
    else:
        print("📱 Using single GPU - no DataParallel wrapping needed")
    
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
    
    print("🔍 DEBUGGING DATA LOADING BEFORE TRAINING:")
    debug_data_loading(train_loader, device)
    # Training
    try:
        training_history = train_neural_ode_model_with_better_debugging(
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

# Additional debugging function to test data loading
def debug_data_loading(train_loader, device):
    """Debug function to test data loading and preparation"""
    print("🔍 DEBUGGING DATA LOADING:")
    
    try:
        batch = next(iter(train_loader))
        print(f"✅ Successfully loaded a batch")
        print(f"   Batch keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}:")
                print(f"     Shape: {value.shape}")
                print(f"     Dtype: {value.dtype}")
                print(f"     Device: {value.device}")
                print(f"     Range: [{value.min().item():.6f}, {value.max().item():.6f}]")
                print(f"     Contains NaN: {torch.isnan(value).any()}")
                print(f"     Contains Inf: {torch.isinf(value).any()}")
        
        # Test prepare_batch_for_ode
        from neural_ode_event_pose import prepare_batch_for_ode
        print("\n🔍 Testing prepare_batch_for_ode...")
        try:
            prepared_batch = prepare_batch_for_ode(batch, device, num_time_steps=10)
            print("✅ prepare_batch_for_ode successful")
            print(f"   Prepared batch keys: {list(prepared_batch.keys())}")
            
            for key, value in prepared_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: shape={value.shape}, device={value.device}")
        
        except Exception as e:
            print(f"❌ prepare_batch_for_ode failed: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()



def train_neural_ode_model_with_better_debugging(model, train_loader, val_loader, 
                                               num_epochs, lr, device, save_dir,
                                               save_every, save_best, val_every, args,
                                               is_multi_gpu, num_gpus):
    """
    FIXED: Training function with better debugging to resolve infinite loss issue
    """
    from neural_ode_event_pose import PoseLoss, prepare_batch_for_ode
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
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
    
    print(f"📈 Learning rate scaled to: {lr:.6f} (base: {lr:.6f})")
    
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
                # Check if batch is valid
                if batch is None:
                    print(f"⚠️  Batch {batch_idx} is None, skipping...")
                    failed_batches += 1
                    continue
                
                # Debug batch size and content
                if 'left_events' not in batch or 'right_events' not in batch:
                    print(f"⚠️  Batch {batch_idx} missing required keys, skipping...")
                    failed_batches += 1
                    continue
                
                batch_size = batch['left_events'].shape[0]
                if batch_size == 0:
                    print(f"⚠️  Batch {batch_idx} has zero size, skipping...")
                    failed_batches += 1
                    continue
                
                # Prepare batch with detailed error checking
                try:
                    prepared_batch = prepare_batch_for_ode(batch, device, args.num_time_steps)
                except Exception as e:
                    print(f"❌ Error preparing batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue
                
                # Check prepared batch validity
                if any(torch.isnan(v).any() for k, v in prepared_batch.items() 
                       if isinstance(v, torch.Tensor)):
                    print(f"⚠️  NaN detected in prepared batch {batch_idx}, skipping...")
                    failed_batches += 1
                    continue
                
                # Forward pass with gradient checking
                optimizer.zero_grad()
                
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
                
                # Check outputs validity
                if 'poses' not in outputs:
                    print(f"⚠️  Model output missing 'poses' key for batch {batch_idx}")
                    failed_batches += 1
                    continue
                
                if torch.isnan(outputs['poses']).any() or torch.isinf(outputs['poses']).any():
                    print(f"⚠️  NaN/Inf in model outputs for batch {batch_idx}")
                    print(f"   Output poses shape: {outputs['poses'].shape}")
                    print(f"   Contains NaN: {torch.isnan(outputs['poses']).any()}")
                    print(f"   Contains Inf: {torch.isinf(outputs['poses']).any()}")
                    failed_batches += 1
                    continue
                
                # Compute loss with validation
                try:
                    loss_dict = pose_loss_fn(outputs['poses'], prepared_batch['gt_poses'])
                    loss = loss_dict['total_loss']
                except Exception as e:
                    print(f"❌ Loss computation error on batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue
                
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    print(f"⚠️  Invalid loss detected on batch {batch_idx}: {loss.item()}")
                    print(f"   Position loss: {loss_dict['position_loss'].item()}")
                    print(f"   Orientation loss: {loss_dict['orientation_loss'].item()}")
                    print(f"   GT poses shape: {prepared_batch['gt_poses'].shape}")
                    print(f"   Pred poses shape: {outputs['poses'].shape}")
                    failed_batches += 1
                    continue
                
                # Backward pass with gradient clipping
                try:
                    loss.backward()
                    
                    # Check for gradient explosion
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    
                    if total_norm > 100:  # Gradient explosion threshold
                        print(f"⚠️  Large gradient norm detected: {total_norm:.2f}")
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                except Exception as e:
                    print(f"❌ Backward pass error on batch {batch_idx}: {e}")
                    failed_batches += 1
                    continue
                
                # Successfully processed batch
                train_loss += loss.item()
                train_pos_loss += loss_dict['position_loss'].item()
                train_orient_loss += loss_dict['orientation_loss'].item()
                num_train_batches += 1
                
                # Print progress every few batches
                if batch_idx % max(1, len(train_loader) // 5) == 0:
                    print(f"   Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.6f}")
                
            except Exception as e:
                print(f"❌ Unexpected error on training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                failed_batches += 1
                continue
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch averages
        if num_train_batches > 0:
            avg_train_loss = train_loss / num_train_batches
            avg_train_pos = train_pos_loss / num_train_batches
            avg_train_orient = train_orient_loss / num_train_batches
        else:
            avg_train_loss = float('inf')
            avg_train_pos = float('inf')
            avg_train_orient = float('inf')
        
        current_lr = scheduler.get_last_lr()[0]
        
        # Print training summary
        print(f"\n📊 Training Summary for Epoch {epoch+1}:")
        print(f"   Successful batches: {num_train_batches}/{len(train_loader)}")
        print(f"   Failed batches: {failed_batches}")
        print(f"   Average train loss: {avg_train_loss:.6f}")
        
        # If no batches succeeded, investigate further
        if num_train_batches == 0:
            print("\n🔍 DEBUGGING: No training batches succeeded!")
            print("Let's investigate the first batch in detail...")
            
            # Try to process first batch with maximum debugging
            try:
                first_batch = next(iter(train_loader))
                print(f"   First batch keys: {list(first_batch.keys())}")
                for key, value in first_batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                        print(f"   {key} range: [{value.min().item():.6f}, {value.max().item():.6f}]")
                        print(f"   {key} contains NaN: {torch.isnan(value).any()}")
                        print(f"   {key} contains Inf: {torch.isinf(value).any()}")
            except Exception as debug_e:
                print(f"   Error investigating first batch: {debug_e}")
        
        # Validation phase
        avg_val_loss = avg_val_pos = avg_val_orient = float('inf')
        if (epoch + 1) % val_every == 0:
            print("🔍 Running validation...")
            avg_val_loss, avg_val_pos, avg_val_orient = validate_model(
                model, val_loader, device, args, is_multi_gpu
            )
        
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
            print(f"💾 Checkpoint saved: {best_model_path}")
            print(f"   ✅ New best model saved! Val Loss: {best_val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_multi_gpu_checkpoint(model, optimizer, scheduler, epoch + 1, avg_train_loss,
                                    training_history, checkpoint_path, is_multi_gpu)
        
        # Early stopping condition
        if num_train_batches == 0:
            print("🛑 Stopping training - no batches processed successfully")
            break
    
    return training_history



if __name__ == "__main__":
    main_w()