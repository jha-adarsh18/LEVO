import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import wandb
from collections import defaultdict

from dataset import EventVODataset, create_dataloader, collate_fn, worker_init_fn
from model import EventVO
from losses import VOLoss


def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch):
    model.train()
    metrics = {'loss': [], 'pose_loss': [], 'rot_loss': [], 'trans_loss': [], 
               'match_loss': [], 'epi_loss': []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        events1 = batch['events1'].to(device)
        mask1 = batch['mask1'].to(device)
        events2 = batch['events2'].to(device)
        mask2 = batch['mask2'].to(device)
        R_gt = batch['R_gt'].to(device)
        t_gt = batch['t_gt'].to(device)
        K = batch['K'].to(device)
        resolution = batch['resolution'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            predictions = model(events1, mask1, events2, mask2)
            if batch_idx % 10 == 0:
                print(f"\nMatch max: {predictions['matches'].max():.6f}, mean: {predictions['matches'].mean():.6f}")
            targets = {'R_gt': R_gt, 't_gt': t_gt, 'K': K}
            loss, stats = loss_fn(predictions, targets)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nNaN/Inf loss detected at batch {batch_idx}!")
            print(f"Stats: {stats}")
            print(f"R_pred range: [{predictions['R_pred'].min():.3f}, {predictions['R_pred'].max():.3f}]")
            print(f"t_pred: {predictions['t_pred'][0]}")
            print(f"matches sum: {predictions['matches'].sum()}")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        
        for key in metrics.keys():
            if key in stats:
                metrics[key].append(stats[key])
        
        pbar.set_postfix({
            'loss': f"{stats['loss']}",
            'rot': f"{stats['rot_loss']}",
            'trans': f"{stats['trans_loss']}",
            'match': f"{stats['match_loss']}"
        })
    
    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    rot_errors = []
    trans_errors = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        events1 = batch['events1'].to(device)
        mask1 = batch['mask1'].to(device)
        events2 = batch['events2'].to(device)
        mask2 = batch['mask2'].to(device)
        R_gt = batch['R_gt'].to(device)
        t_gt = batch['t_gt'].to(device)
        
        predictions = model(events1, mask1, events2, mask2)
        R_pred = predictions['R_pred']
        t_pred = predictions['t_pred']
        
        t_gt_norm = F.normalize(t_gt, p=2, dim=1)
        
        for i in range(R_pred.shape[0]):
            trace = torch.trace(R_pred[i] @ R_gt[i].T)
            rot_error = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
            rot_errors.append(rot_error.item() * 180 / np.pi)
            
            cos_sim = (t_pred[i] * t_gt_norm[i]).sum()
            trans_error = torch.acos(torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7))
            trans_errors.append(trans_error.item() * 180 / np.pi)
    
    return {
        'rot_error_mean': np.mean(rot_errors),
        'rot_error_median': np.median(rot_errors),
        'trans_error_mean': np.mean(trans_errors),
        'trans_error_median': np.median(trans_errors)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./checkpoints_vo')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--camera', type=str, default='left')
    parser.add_argument('--dt-range', type=int, nargs=2, default=[50, 200])
    parser.add_argument('--n-events', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--d-desc', type=int, default=256)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--intrinsics-config', type=str, default='/home/adarsh/PEVSLAM/configs/config.yaml')
    parser.add_argument('--wandb-project', type=str, default='event-vo')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=500)
    
    args = parser.parse_args()
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Creating dataset (loading to RAM once)...")
    full_dataset = EventVODataset(
        args.data_root,
        camera=args.camera,
        dt_range=tuple(args.dt_range),
        n_events=args.n_events,
        augment=True,
        intrinsics_config=args.intrinsics_config
    )

    # Sequence-level split for true generalization
    train_seqs = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 
                  'outdoor_day1', 'outdoor_night1', 'outdoor_night2', 'outdoor_night3']
    val_seqs = ['indoor_flying4', 'outdoor_day2']

    print(f"Training sequences: {train_seqs}")
    print(f"Validation sequences: {val_seqs}")

    train_indices = [i for i, (seq, _, _) in enumerate(full_dataset.pairs) if seq in train_seqs]
    val_indices = [i for i, (seq, _, _) in enumerate(full_dataset.pairs) if seq in val_seqs]

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Filter out empty samples from validation - vectorized version
    print("Filtering validation dataset...")
    valid_val_indices = []
    window_us = full_dataset.event_window_ms * 1000
    
    # Group validation indices by sequence for batch processing
    seq_to_local_indices = defaultdict(list)
    seq_to_pair_info = defaultdict(list)
    
    for local_idx in range(len(val_dataset)):
        global_idx = val_indices[local_idx]
        seq_name, i, j = full_dataset.pairs[global_idx]
        seq_to_local_indices[seq_name].append(local_idx)
        seq_to_pair_info[seq_name].append((i, j))
    
    # Process each sequence in batch
    for seq_name in tqdm(seq_to_local_indices.keys(), desc="Filtering sequences"):
        seq = full_dataset.sequences[seq_name]
        local_indices = seq_to_local_indices[seq_name]
        pair_info = seq_to_pair_info[seq_name]
        
        events_t = seq['events_t']
        timestamps = seq['timestamps']
        
        # Vectorize timestamp extraction
        i_indices = np.array([p[0] for p in pair_info])
        j_indices = np.array([p[1] for p in pair_info])
        
        t1_batch = timestamps[i_indices]
        t2_batch = timestamps[j_indices]
        
        # Vectorize window calculations
        t1_start = (t1_batch - window_us / 2e6) * 1e6
        t1_end = (t1_batch + window_us / 2e6) * 1e6
        t2_start = (t2_batch - window_us / 2e6) * 1e6
        t2_end = (t2_batch + window_us / 2e6) * 1e6
        
        # Batch searchsorted
        t1_start_idx = np.searchsorted(events_t, t1_start)
        t1_end_idx = np.searchsorted(events_t, t1_end)
        t2_start_idx = np.searchsorted(events_t, t2_start)
        t2_end_idx = np.searchsorted(events_t, t2_end)
        
        # Check valid samples
        n_events1 = t1_end_idx - t1_start_idx
        n_events2 = t2_end_idx - t2_start_idx
        
        valid_mask = (n_events1 > 0) & (n_events2 > 0)
        valid_val_indices.extend([local_indices[i] for i in np.where(valid_mask)[0]])
    
    valid_val_indices = sorted(valid_val_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, valid_val_indices)
    print(f"Validation: {len(val_dataset)} samples (after filtering)")

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers // 2, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    model = EventVO(d_model=args.d_model, num_samples=args.num_samples).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    wandb.watch(model, log='all', log_freq=100)
    
    loss_fn = VOLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    start_epoch = 0
    best_rot_error = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_rot_error = checkpoint.get('best_rot_error', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        loss_fn.update_weights(epoch, args.epochs)
        
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, epoch)
        scheduler.step()
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/rot_loss': train_metrics['rot_loss'],
            'train/trans_loss': train_metrics['trans_loss'],
            'train/match_loss': train_metrics['match_loss'],
            'train/epi_loss': train_metrics['epi_loss'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"\nEpoch {epoch}: Loss={train_metrics['loss']}, "
              f"Rot={train_metrics['rot_loss']}, Trans={train_metrics['trans_loss']}, "
              f"Match={train_metrics['match_loss']}, Epi={train_metrics['epi_loss']}")
        
        if (epoch + 1) % args.validate_every == 0:
            val_metrics = validate(model, val_loader, device)
            
            wandb.log({
                'epoch': epoch,
                'val/rot_error_mean': val_metrics['rot_error_mean'],
                'val/rot_error_median': val_metrics['rot_error_median'],
                'val/trans_error_mean': val_metrics['trans_error_mean'],
                'val/trans_error_median': val_metrics['trans_error_median']
            })
            
            print(f"Val - Rot error: {val_metrics['rot_error_mean']}° (median: {val_metrics['rot_error_median']}°)")
            print(f"Val - Trans error: {val_metrics['trans_error_mean']}° (median: {val_metrics['trans_error_median']}°)")
            
            if val_metrics['rot_error_mean'] < best_rot_error:
                best_rot_error = val_metrics['rot_error_mean']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'metrics': val_metrics,
                    'best_rot_error': best_rot_error,
                }, output_dir / 'best_model.pth')
                print(f"✓ Best model saved: Rot={best_rot_error:.2f}°")
                wandb.save(str(output_dir / 'best_model.pth'))
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_rot_error': best_rot_error,
            }, output_dir / f'checkpoint_{epoch:03d}.pth')
    
    print(f"\n✓ Training complete! Best rot error: {best_rot_error:.2f}°")
    wandb.finish()


if __name__ == '__main__':
    main()