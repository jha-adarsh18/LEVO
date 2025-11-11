import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, scheduler, device, epoch):
    model.train()
    metrics = {'loss': [], 'pose_loss': [], 'rot_loss': [], 'trans_loss': [], 
               'match_loss': [], 'epi_loss': [], 'contrastive_loss': []}
    
    skipped_batches = 0
    total_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        total_batches += 1
        
        events1 = batch['events1'].to(device, non_blocking=True)
        mask1 = batch['mask1'].to(device, non_blocking=True)
        events2 = batch['events2'].to(device, non_blocking=True)
        mask2 = batch['mask2'].to(device, non_blocking=True)
        R_gt = batch['R_gt'].to(device, non_blocking=True)
        t_gt = batch['t_gt'].to(device, non_blocking=True)
        K = batch['K'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            predictions = model(events1, mask1, events2, mask2, K)
        
        targets = {'R_gt': R_gt, 't_gt': t_gt, 'K': K}
        loss, stats = loss_fn(predictions, targets)
        
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_batches += 1
            pbar.set_postfix({
                'loss': 'NaN/Inf (skipped)',
                'skipped': f"{skipped_batches}/{total_batches}"
            })
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if batch_idx % 10 == 0:
            for key in metrics.keys():
                if key in stats:
                    metrics[key].append(stats[key])
        
        pbar.set_postfix({
            'loss': f"{stats['loss']:.3f}",
            'match': f"{stats['match_loss']:.3f}",
            'epi': f"{stats['epi_loss']:.3f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            'skipped': f"{skipped_batches}/{total_batches}"
        })
    
    if skipped_batches > 0:
        print(f"\n⚠ Skipped {skipped_batches}/{total_batches} batches due to NaN/Inf losses")
    
    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    rot_errors = []
    trans_errors = []
    match_scores = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        events1 = batch['events1'].to(device, non_blocking=True)
        mask1 = batch['mask1'].to(device, non_blocking=True)
        events2 = batch['events2'].to(device, non_blocking=True)
        mask2 = batch['mask2'].to(device, non_blocking=True)
        R_gt = batch['R_gt'].to(device, non_blocking=True)
        t_gt = batch['t_gt'].to(device, non_blocking=True)
        K = batch['K'].to(device, non_blocking=True)
        
        if mask1.sum() == 0 or mask2.sum() == 0:
            continue
        
        predictions = model(events1, mask1, events2, mask2, K)
        R_pred = predictions['R_pred']
        t_pred = predictions['t_pred']
        matches = predictions['matches']
        
        match_scores.extend(matches.max(dim=2)[0].mean(dim=1).cpu().numpy().tolist())
        
        t_pred_norm = F.normalize(t_pred, p=2, dim=1)
        t_gt_norm = F.normalize(t_gt, p=2, dim=1)
        
        for i in range(R_pred.shape[0]):
            R_pred_i = R_pred[i]
            R_gt_i = R_gt[i]
            
            if torch.isnan(R_pred_i).any() or torch.isnan(R_gt_i).any():
                continue
            
            if torch.isinf(R_pred_i).any() or torch.isinf(R_gt_i).any():
                continue
            
            trace = torch.trace(R_pred_i @ R_gt_i.T)
            
            if torch.isnan(trace):
                continue
            
            if trace < -1.1 or trace > 3.1:
                continue
            
            trace_val = (trace - 1) / 2
            trace_clamped = torch.clamp(trace_val, -0.9999, 0.9999)
            rot_error = torch.acos(trace_clamped)
            
            if torch.isnan(rot_error):
                continue
            
            rot_errors.append(rot_error.item() * 180 / np.pi)
            
            cos_sim = (t_pred_norm[i] * t_gt_norm[i]).sum()
            
            if torch.isnan(cos_sim):
                continue
            
            cos_clamped = torch.clamp(cos_sim, -0.9999, 0.9999)
            trans_error = torch.acos(cos_clamped)
            
            if torch.isnan(trans_error):
                continue
            
            trans_errors.append(trans_error.item() * 180 / np.pi)
    
    return {
        'rot_error_mean': np.mean(rot_errors) if len(rot_errors) > 0 else float('nan'),
        'rot_error_median': np.median(rot_errors) if len(rot_errors) > 0 else float('nan'),
        'trans_error_mean': np.mean(trans_errors) if len(trans_errors) > 0 else float('nan'),
        'trans_error_median': np.median(trans_errors) if len(trans_errors) > 0 else float('nan'),
        'match_score_mean': np.mean(match_scores) if len(match_scores) > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/workspace/dataset')
    parser.add_argument('--output-dir', type=str, default='./checkpoints_vo')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--camera', type=str, default='left')
    parser.add_argument('--dt-range', type=int, nargs=2, default=[50, 200])
    parser.add_argument('--n-events', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--validate-every', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--intrinsics-config', type=str, default='/workspace/PEVSLAM/configs/config.yaml')
    parser.add_argument('--wandb-project', type=str, default='event-vo')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--stratified-sampling', action='store_true')
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--training-stage', type=str, default='matching', choices=['matching', 'joint'])
    parser.add_argument('--stage1-epochs', type=int, default=20)
    
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Creating dataset...")
    full_dataset = EventVODataset(
        args.data_root,
        camera=args.camera,
        dt_range=tuple(args.dt_range),
        n_events=args.n_events,
        augment=True,
        intrinsics_config=args.intrinsics_config
    )

    train_seqs = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 
                  'outdoor_day1', 'outdoor_night1', 'outdoor_night2', 'outdoor_night3']
    val_seqs = ['indoor_flying4', 'outdoor_day2', 'mocap-desk']
    
    group_e_seqs = [
        'mocap-1d-trans', 'mocap-3d-trans', 'mocap-6dof', 'mocap-shake', 'mocap-shake2',
        'office-maze', 'running-easy', 'running-hard', 'skate-easy', 'skate-hard',
        'floor2-dark', 'slide', 'bike-easy', 'bike-hard', 'bike-dark'
    ]
    train_seqs.extend(group_e_seqs)
    
    train_indices = []
    val_indices = []
    
    for i, (seq, _, _) in enumerate(full_dataset.pairs):
        seq_lower = seq.lower().replace('_', '-')
        is_train = any(train_seq.lower().replace('_', '-') in seq_lower for train_seq in train_seqs)
        is_val = any(val_seq.lower().replace('_', '-') in seq_lower for val_seq in val_seqs)
        
        if is_val:
            val_indices.append(i)
        elif is_train:
            train_indices.append(i)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")
    
    sampler = None
    shuffle = True
    
    if args.stratified_sampling:
        seq_counts = defaultdict(int)
        for idx in train_indices:
            seq_name, _, _ = full_dataset.pairs[idx]
            seq_counts[seq_name] += 1
        
        weights = []
        for idx in train_indices:
            seq_name, _, _ = full_dataset.pairs[idx]
            weights.append(1.0 / seq_counts[seq_name])
        
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers, 
        collate_fn=collate_fn,
        pin_memory=True, 
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        prefetch_factor=args.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2
    )

    model = EventVO(d_model=args.d_model, num_samples=args.num_samples).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    if args.training_stage == 'matching':
        print("\n" + "="*60)
        print("STAGE 1: MATCHING-ONLY TRAINING")
        print("Pose losses disabled - focusing on correspondence quality")
        print("="*60 + "\n")
        loss_fn = VOLoss(
            w_rot=0.0,
            w_trans=0.0,
            w_match=10.0,
            w_epipolar=5.0,
            w_contrastive=2.0
        )
    else:
        print("\n" + "="*60)
        print("STAGE 2: JOINT TRAINING")
        print("All losses enabled with matching-dominant weighting")
        print("="*60 + "\n")
        loss_fn = VOLoss(
            w_rot=0.5,
            w_trans=1.0,
            w_match=10.0,
            w_epipolar=5.0,
            w_contrastive=2.0
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        total_steps = args.epochs * len(train_loader)
        progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    
    start_epoch = 0
    best_metric = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('best_metric', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, scaler, scheduler, device, epoch)
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/rot_loss': train_metrics['rot_loss'],
            'train/trans_loss': train_metrics['trans_loss'],
            'train/match_loss': train_metrics['match_loss'],
            'train/epi_loss': train_metrics['epi_loss'],
            'train/contrastive_loss': train_metrics['contrastive_loss'],
            'lr': optimizer.param_groups[0]['lr'],
            'w_rot': loss_fn.w_rot,
            'w_trans': loss_fn.w_trans,
            'w_match': loss_fn.w_match,
            'w_epipolar': loss_fn.w_epipolar,
            'w_contrastive': loss_fn.w_contrastive
        })
        
        print(f"\nEpoch {epoch}: Loss={train_metrics['loss']:.3f}, "
              f"Match={train_metrics['match_loss']:.3f}, Epi={train_metrics['epi_loss']:.3f}")
        
        if (epoch + 1) % args.validate_every == 0:
            val_metrics = validate(model, val_loader, device)
            
            wandb.log({
                'epoch': epoch,
                'val/rot_error_mean': val_metrics['rot_error_mean'],
                'val/rot_error_median': val_metrics['rot_error_median'],
                'val/trans_error_mean': val_metrics['trans_error_mean'],
                'val/trans_error_median': val_metrics['trans_error_median'],
                'val/match_score_mean': val_metrics['match_score_mean']
            })
            
            print(f"Val - Match Score: {val_metrics['match_score_mean']:.4f}")
            if not np.isnan(val_metrics['rot_error_mean']):
                print(f"Val - Rot: {val_metrics['rot_error_mean']:.2f}° (median: {val_metrics['rot_error_median']:.2f}°)")
                print(f"Val - Trans: {val_metrics['trans_error_mean']:.2f}° (median: {val_metrics['trans_error_median']:.2f}°)")
            
            if args.training_stage == 'matching':
                current_metric = -val_metrics['match_score_mean']
            else:
                current_metric = val_metrics['rot_error_median']
            
            if current_metric < best_metric:
                best_metric = current_metric
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'metrics': val_metrics,
                    'best_metric': best_metric,
                }, output_dir / 'best_model.pth')
                if args.training_stage == 'matching':
                    print(f"✓ Best model saved: Match Score={val_metrics['match_score_mean']:.4f}")
                else:
                    print(f"✓ Best model saved: Rot={val_metrics['rot_error_median']:.2f}°")
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_metric': best_metric,
            }, output_dir / f'checkpoint_{epoch:03d}.pth')
    
    print(f"\n✓ Training complete!")
    wandb.finish()


if __name__ == '__main__':
    main()