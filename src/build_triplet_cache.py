#!/usr/bin/env python3
"""
Build triplet cache for metric learning from pose data (GPU-accelerated).

Input:
  - poses.h5 (from pose converter script)
  - *-events_left.h5/*-events_right.h5 files (for sequence boundaries)

Output: triplet_cache.h5
  /triplets/indices [N, 6] # (anchor_seq_idx, anchor_time_idx, pos_seq_idx, pos_time_idx, neg_seq_idx, neg_time_idx)
  /triplets/metadata
  /sequence_names [list of sequence names]

Triplet Selection Logic:
- Positives: same location (±0.5m), different view (15-60° rotation), time gap >1s
- Hard Negatives: far location (>3m), or visually similar but different place
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import torch


def quat_to_rotation_matrix_batch(quats):
    """
    Convert batch of quaternions to rotation matrices on GPU.
    quats: [N, 4] tensor (x, y, z, w)
    Returns: [N, 3, 3] rotation matrices
    """
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    # Normalize
    norm = torch.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Build rotation matrices
    R = torch.zeros(len(quats), 3, 3, device=quats.device, dtype=quats.dtype)
    
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    
    return R


def rotation_angles_batch(q1, quats_batch):
    """
    Calculate rotation angles between one quaternion and a batch.
    q1: [4] single quaternion (x, y, z, w)
    quats_batch: [N, 4] batch of quaternions
    Returns: [N] angles in degrees
    """
    device = quats_batch.device
    
    # Convert to rotation matrices
    R1 = quat_to_rotation_matrix_batch(q1.unsqueeze(0))[0]  # [3, 3]
    R2_batch = quat_to_rotation_matrix_batch(quats_batch)  # [N, 3, 3]
    
    # Relative rotation: R1^T @ R2
    R_rel = torch.matmul(R1.T.unsqueeze(0), R2_batch)  # [N, 3, 3]
    
    # Angle from trace: theta = arccos((trace(R) - 1) / 2)
    traces = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # [N]
    
    # Clamp for numerical stability
    cos_angle = (traces - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    angles = torch.acos(cos_angle) * 180 / np.pi  # [N] in degrees
    
    return angles


def load_poses(poses_h5_path):
    """Load all poses from poses.h5 and return structured data."""
    sequences = {}
    sequence_names = []
    
    with h5py.File(poses_h5_path, 'r') as f:
        for seq_name in f.keys():
            timestamps = f[seq_name]['timestamps'][:]
            positions = f[seq_name]['positions'][:]
            quaternions = f[seq_name]['quaternions'][:]
            
            sequences[seq_name] = {
                'timestamps': timestamps,
                'positions': positions,
                'quaternions': quaternions,
                'seq_idx': len(sequence_names)
            }
            sequence_names.append(seq_name)
    
    return sequences, sequence_names


def find_positives_gpu(anchor_pos, anchor_quat, anchor_time, anchor_seq_idx,
                       target_positions, target_quaternions, target_times, target_seq_idx,
                       pos_dist_thresh=0.5, min_angle=15, max_angle=60, min_time_gap=1.0,
                       device='cuda'):
    """
    Find positive samples using GPU acceleration.
    
    Returns: numpy array of valid positive indices
    """
    # Convert to tensors
    anchor_pos_t = torch.from_numpy(anchor_pos).float().to(device)
    anchor_quat_t = torch.from_numpy(anchor_quat).float().to(device)
    target_pos_t = torch.from_numpy(target_positions).float().to(device)
    target_quat_t = torch.from_numpy(target_quaternions).float().to(device)
    
    # Calculate distances (GPU)
    distances = torch.norm(target_pos_t - anchor_pos_t, dim=1)
    location_mask = distances <= pos_dist_thresh
    
    # Early exit if no candidates
    candidate_indices = torch.where(location_mask)[0]
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int64)
    
    # Calculate rotation angles only for distance-valid candidates (GPU)
    candidate_quats = target_quat_t[candidate_indices]
    angles_candidates = rotation_angles_batch(anchor_quat_t, candidate_quats)
    
    # Build full angle array
    angles = torch.zeros(len(target_quaternions), device=device)
    angles[candidate_indices] = angles_candidates
    
    # Calculate time gaps
    time_gaps = np.abs(target_times - anchor_time)
    time_gaps_t = torch.from_numpy(time_gaps).float().to(device)
    
    # Apply all constraints
    angle_mask = (angles >= min_angle) & (angles <= max_angle)
    time_mask = time_gaps_t >= min_time_gap
    
    # If same sequence, enforce time gap; if different sequence, no time constraint
    if anchor_seq_idx == target_seq_idx:
        valid_mask = location_mask & angle_mask & time_mask
    else:
        valid_mask = location_mask & angle_mask
    
    valid_indices = torch.where(valid_mask)[0].cpu().numpy()
    return valid_indices


def find_hard_negatives_gpu(anchor_pos, anchor_quat,
                            target_positions, target_quaternions,
                            neg_dist_thresh=3.0, similar_angle_thresh=10,
                            device='cuda'):
    """
    Find hard negative samples using GPU acceleration.
    
    Returns: numpy array of valid negative indices
    """
    # Convert to tensors
    anchor_pos_t = torch.from_numpy(anchor_pos).float().to(device)
    anchor_quat_t = torch.from_numpy(anchor_quat).float().to(device)
    target_pos_t = torch.from_numpy(target_positions).float().to(device)
    target_quat_t = torch.from_numpy(target_quaternions).float().to(device)
    
    # Calculate distances (GPU)
    distances = torch.norm(target_pos_t - anchor_pos_t, dim=1)
    
    # Far negatives (don't need angle check)
    far_mask = distances > neg_dist_thresh
    
    # For "similar view different place", check middle distance range
    middle_dist_mask = (distances > 0.5) & (distances <= neg_dist_thresh)
    middle_dist_indices = torch.where(middle_dist_mask)[0]
    
    # Only compute angles for middle distance candidates
    angles = torch.zeros(len(target_quaternions), device=device)
    if len(middle_dist_indices) > 0:
        middle_quats = target_quat_t[middle_dist_indices]
        angles_middle = rotation_angles_batch(anchor_quat_t, middle_quats)
        angles[middle_dist_indices] = angles_middle
    
    # Similar view but different place
    similar_view_diff_place = middle_dist_mask & (angles < similar_angle_thresh)
    
    valid_mask = far_mask | similar_view_diff_place
    valid_indices = torch.where(valid_mask)[0].cpu().numpy()
    
    return valid_indices


def build_triplets(sequences, sequence_names, 
                   samples_per_anchor=5,
                   pos_dist_thresh=0.5,
                   min_angle=15,
                   max_angle=60,
                   min_time_gap=1.0,
                   neg_dist_thresh=3.0,
                   device='cuda'):
    """
    Build triplet indices for all sequences using GPU acceleration.
    
    Returns: 
        triplets: array of shape [N, 6] with (anchor_seq, anchor_idx, pos_seq, pos_idx, neg_seq, neg_idx)
        metadata: dict with statistics
    """
    triplets = []
    stats = {
        'total_anchors': 0,
        'anchors_with_positives': 0,
        'total_triplets': 0,
        'per_sequence_stats': {}
    }
    
    print(f"Building triplets on {device}...")
    
    for anchor_seq_name in tqdm(sequence_names, desc="Processing sequences"):
        anchor_data = sequences[anchor_seq_name]
        anchor_seq_idx = anchor_data['seq_idx']
        
        seq_stats = {
            'total_poses': len(anchor_data['timestamps']),
            'triplets_generated': 0
        }
        
        n_poses = len(anchor_data['timestamps'])
        anchor_indices = np.arange(n_poses)
        
        for anchor_idx in tqdm(anchor_indices, desc=f"  {anchor_seq_name}", leave=False):
            anchor_pos = anchor_data['positions'][anchor_idx]
            anchor_quat = anchor_data['quaternions'][anchor_idx]
            anchor_time = anchor_data['timestamps'][anchor_idx]
            
            stats['total_anchors'] += 1
            
            # Collect all positive candidates from all sequences
            all_positives = []
            
            for target_seq_name in sequence_names:
                target_data = sequences[target_seq_name]
                target_seq_idx = target_data['seq_idx']
                
                pos_indices = find_positives_gpu(
                    anchor_pos, anchor_quat, anchor_time, anchor_seq_idx,
                    target_data['positions'],
                    target_data['quaternions'],
                    target_data['timestamps'],
                    target_seq_idx,
                    pos_dist_thresh, min_angle, max_angle, min_time_gap,
                    device
                )
                
                for pos_idx in pos_indices:
                    all_positives.append((target_seq_idx, pos_idx))
            
            if len(all_positives) == 0:
                continue
            
            stats['anchors_with_positives'] += 1
            
            # Collect all negative candidates from all sequences
            all_negatives = []
            
            for target_seq_name in sequence_names:
                target_data = sequences[target_seq_name]
                target_seq_idx = target_data['seq_idx']
                
                neg_indices = find_hard_negatives_gpu(
                    anchor_pos, anchor_quat,
                    target_data['positions'],
                    target_data['quaternions'],
                    neg_dist_thresh,
                    device=device
                )
                
                for neg_idx in neg_indices:
                    all_negatives.append((target_seq_idx, neg_idx))
            
            if len(all_negatives) == 0:
                continue
            
            # Sample triplets
            n_triplets = min(samples_per_anchor, len(all_positives))
            
            for _ in range(n_triplets):
                # Randomly select positive
                pos_seq_idx, pos_idx = all_positives[np.random.randint(len(all_positives))]
                
                # Randomly select negative
                neg_seq_idx, neg_idx = all_negatives[np.random.randint(len(all_negatives))]
                
                triplets.append([
                    anchor_seq_idx, anchor_idx,
                    pos_seq_idx, pos_idx,
                    neg_seq_idx, neg_idx
                ])
                
                seq_stats['triplets_generated'] += 1
                stats['total_triplets'] += 1
        
        stats['per_sequence_stats'][anchor_seq_name] = seq_stats
    
    triplets = np.array(triplets, dtype=np.int32)
    
    print(f"\nTriplet Generation Complete:")
    print(f"  Total anchors: {stats['total_anchors']}")
    print(f"  Anchors with positives: {stats['anchors_with_positives']}")
    print(f"  Total triplets: {stats['total_triplets']}")
    
    return triplets, stats


def save_triplet_cache(output_path, triplets, metadata, sequence_names):
    """Save triplet cache to HDF5 file."""
    with h5py.File(output_path, 'w') as f:
        # Save triplets
        triplet_grp = f.create_group('triplets')
        triplet_grp.create_dataset('indices', data=triplets, compression='gzip')
        
        # Save metadata as JSON string
        triplet_grp.attrs['metadata'] = json.dumps(metadata)
        
        # Save sequence names
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('sequence_names', data=sequence_names, dtype=dt)
        
        # Save config for reference
        config = {
            'n_triplets': len(triplets),
            'n_sequences': len(sequence_names)
        }
        f.attrs['config'] = json.dumps(config)
    
    print(f"\nSaved triplet cache to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build triplet cache for metric learning from pose data (GPU-accelerated)'
    )
    parser.add_argument('poses_h5', type=str, help='Path to poses.h5 file')
    parser.add_argument('--output', type=str, default='triplet_cache.h5', 
                        help='Output cache file (default: triplet_cache.h5)')
    parser.add_argument('--samples-per-anchor', type=int, default=5,
                        help='Number of triplets per anchor (default: 5)')
    parser.add_argument('--pos-dist', type=float, default=0.5,
                        help='Positive distance threshold in meters (default: 0.5)')
    parser.add_argument('--min-angle', type=float, default=15,
                        help='Minimum rotation angle for positives in degrees (default: 15)')
    parser.add_argument('--max-angle', type=float, default=60,
                        help='Maximum rotation angle for positives in degrees (default: 60)')
    parser.add_argument('--min-time-gap', type=float, default=1.0,
                        help='Minimum time gap for positives in seconds (default: 1.0)')
    parser.add_argument('--neg-dist', type=float, default=3.0,
                        help='Negative distance threshold in meters (default: 3.0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    poses_path = Path(args.poses_h5)
    if not poses_path.exists():
        print(f"Error: poses.h5 file not found at {poses_path}")
        return
    
    # Load poses
    print(f"Loading poses from {poses_path}...")
    sequences, sequence_names = load_poses(poses_path)
    print(f"Loaded {len(sequences)} sequences")
    
    # Build triplets
    triplets, metadata = build_triplets(
        sequences, sequence_names,
        samples_per_anchor=args.samples_per_anchor,
        pos_dist_thresh=args.pos_dist,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        min_time_gap=args.min_time_gap,
        neg_dist_thresh=args.neg_dist,
        device=args.device
    )
    
    # Save cache
    save_triplet_cache(args.output, triplets, metadata, sequence_names)
    
    print("\nDone!")


if __name__ == '__main__':
    main()