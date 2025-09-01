import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import argparse
from pathlib import Path
import json
import numpy as np

# Import custom modules (assuming they're in the same directory)
from neural_ode_dataset import EventODEDataset, collate_fn
from neural_ode_model import NeuralODEEventSLAM, PoseLoss, train_epoch, validate_epoch, get_cosine_schedule_with_warmup


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save epoch-specific checkpoint
    epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    print(f"Loaded checkpoint from epoch {epoch}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
    
    return epoch, train_loss, val_loss


def create_data_loaders(config):
    """Create training and validation data loaders"""
    
    # Training dataset
    train_dataset = EventODEDataset(
        dataset_root=config['train_dataset_root'],
        window_duration=config['window_duration'],
        slice_duration=config['slice_duration'],
        overlap=config['overlap']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Validation dataset
    if config.get('val_dataset_root'):
        val_dataset = EventODEDataset(
            dataset_root=config['val_dataset_root'],
            window_duration=config['window_duration'],
            slice_duration=config['slice_duration'],
            overlap=config['overlap']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True if config['device'] == 'cuda' else False
        )
    else:
        # Use a split from training data if no separate validation set
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True if config['device'] == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True if config['device'] == 'cuda' else False
        )
    
    return train_loader, val_loader


def train_neural_ode_slam():
    """Main training function"""
    
    # Configuration
    config = {
        'train_dataset_root': "//media/adarsh/One Touch/EventSLAM/dataset/train",
        'val_dataset_root': "//media/adarsh/One Touch/EventSLAM/dataset/val",  # Optional
        
        # Dataset parameters
        'window_duration': 1000,    # 1 second windows
        'slice_duration': 5,        # 5ms slices
        'overlap': 200,             # 200ms overlap between windows
        
        # Model parameters
        'hidden_dim': 256,
        'event_dim': 64,
        'pose_dim': 7,
        
        # Training parameters
        'batch_size': 8,            # Reduced batch size due to memory constraints
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lambda_t': 1.0,            # Translation loss weight
        'lambda_r': 1.0,            # Rotation loss weight
        
        # Training settings
        'num_workers': 2,           # Reduced due to memory loading
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 15,
        'grad_clip_norm': 1.0,
        
        # Checkpoints and logging
        'checkpoint_dir': './checkpoints_neural_ode',
        'log_interval': 10,         # Log every 10 batches
        'save_interval': 5,         # Save checkpoint every 5 epochs
        
        # Resume training
        'resume_from_checkpoint': None,  # Path to checkpoint to resume from
    }
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("Initializing model...")
    model = NeuralODEEventSLAM(
        hidden_dim=config['hidden_dim'],
        event_dim=config['event_dim'],
        pose_dim=config['pose_dim']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = PoseLoss(
        lambda_t=config['lambda_t'],
        lambda_r=config['lambda_r']
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if config['resume_from_checkpoint']:
        start_epoch, train_loss, val_loss = load_checkpoint(
            config['resume_from_checkpoint'], 
            model, optimizer, scheduler
        )
        best_val_loss = val_loss
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    print("Starting training...")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Training
        train_loss, train_t_loss, train_r_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validation
        val_loss, val_t_loss, val_r_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.6f} (T: {train_t_loss:.6f}, R: {train_r_loss:.6f})")
        print(f"Val Loss:   {val_loss:.6f} (T: {val_t_loss:.6f}, R: {val_r_loss:.6f})")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                train_loss, val_loss, config, config['checkpoint_dir']
            )
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            torch.save(best_checkpoint, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f"🏆 New best model saved! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⏰ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
        
        print(f"Patience: {patience_counter}/{config['early_stopping_patience']}")
    
    # Training completed
    total_training_time = time.time() - training_start_time
    print(f"\n✅ Training completed!")
    print(f"Total training time: {total_training_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, epoch + 1,
        train_loss, val_loss, config, config['checkpoint_dir']
    )


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Train Neural ODE Event SLAM')
    
    parser.add_argument('--train_data', type=str, 
                       default="//media/adarsh/One Touch/EventSLAM/dataset/train",
                       help='Path to training dataset')
    parser.add_argument('--val_data', type=str, 
                       default="//media/adarsh/One Touch/EventSLAM/dataset/val",
                       help='Path to validation dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_neural_ode',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    # You can modify the config dictionary in train_neural_ode_slam() 
    # to use these arguments
    
    train_neural_ode_slam()


if __name__ == "__main__":
    main()