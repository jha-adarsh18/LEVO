#!/usr/bin/env python3
"""
Build triplet cache for metric learning from pose data.

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
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json


def rotation_angle_between_quaternions(q1, q2):
    """
    Calculate rotation angle in degrees between two quaternions.
    q1, q2: quaternions in (x, y, z, w) format
    """
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    relative_rotation = r1.inv() * r2
    angle = relative_rotation.magnitude() * 180 / np.pi
    return angle


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


def find_positives(anchor_pos, anchor_quat, anchor_time, anchor_seq_idx,
                   target_positions, target_quaternions, target_times, target_seq_idx,
                   pos_dist_thresh=0.5, min_angle=15, max_angle=60, min_time_gap=1.0):
    """
    Find positive samples: same location, different viewpoint, time gap.
    
    Returns: list of valid positive indices
    """
    # Calculate distances
    distances = np.linalg.norm(target_positions - anchor_pos, axis=1)
    
    # Calculate rotation angles
    angles = np.array([
        rotation_angle_between_quaternions(anchor_quat, q) 
        for q in target_quaternions
    ])
    
    # Calculate time gaps
    time_gaps = np.abs(target_times - anchor_time)
    
    # Apply constraints
    location_mask = distances <= pos_dist_thresh
    angle_mask = (angles >= min_angle) & (angles <= max_angle)
    time_mask = time_gaps >= min_time_gap
    
    # If same sequence, enforce time gap; if different sequence, no time constraint needed
    if anchor_seq_idx == target_seq_idx:
        valid_mask = location_mask & angle_mask & time_mask
    else:
        valid_mask = location_mask & angle_mask
    
    valid_indices = np.where(valid_mask)[0]
    return valid_indices


def find_hard_negatives(anchor_pos, anchor_quat, 
                        target_positions, target_quaternions,
                        neg_dist_thresh=3.0, similar_angle_thresh=10):
    """
    Find hard negative samples:
    1. Far location (>3m)
    2. Similar viewpoint but different location (visually similar but wrong place)
    
    Returns: list of valid negative indices
    """
    # Calculate distances
    distances = np.linalg.norm(target_positions - anchor_pos, axis=1)
    
    # Calculate rotation angles
    angles = np.array([
        rotation_angle_between_quaternions(anchor_quat, q) 
        for q in target_quaternions
    ])
    
    # Hard negatives: far away
    far_mask = distances > neg_dist_thresh
    
    # Hard negatives: similar angle but different location (0.5m to 3m away with similar view)
    similar_view_diff_place = (distances > 0.5) & (distances <= neg_dist_thresh) & (angles < similar_angle_thresh)
    
    valid_mask = far_mask | similar_view_diff_place
    valid_indices = np.where(valid_mask)[0]
    
    return valid_indices


def build_triplets(sequences, sequence_names, 
                   samples_per_anchor=5,
                   pos_dist_thresh=0.5,
                   min_angle=15,
                   max_angle=60,
                   min_time_gap=1.0,
                   neg_dist_thresh=3.0):
    """
    Build triplet indices for all sequences.
    
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
    
    print("Building triplets...")
    
    for anchor_seq_name in tqdm(sequence_names, desc="Processing sequences"):
        anchor_data = sequences[anchor_seq_name]
        anchor_seq_idx = anchor_data['seq_idx']
        
        seq_stats = {
            'total_poses': len(anchor_data['timestamps']),
            'triplets_generated': 0
        }
        
        # Sample anchors (can use all or subsample)
        n_poses = len(anchor_data['timestamps'])
        anchor_indices = np.arange(n_poses)
        
        for anchor_idx in anchor_indices:
            anchor_pos = anchor_data['positions'][anchor_idx]
            anchor_quat = anchor_data['quaternions'][anchor_idx]
            anchor_time = anchor_data['timestamps'][anchor_idx]
            
            stats['total_anchors'] += 1
            
            # Collect all positive candidates from all sequences
            all_positives = []
            
            for target_seq_name in sequence_names:
                target_data = sequences[target_seq_name]
                target_seq_idx = target_data['seq_idx']
                
                pos_indices = find_positives(
                    anchor_pos, anchor_quat, anchor_time, anchor_seq_idx,
                    target_data['positions'],
                    target_data['quaternions'],
                    target_data['timestamps'],
                    target_seq_idx,
                    pos_dist_thresh, min_angle, max_angle, min_time_gap
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
                
                neg_indices = find_hard_negatives(
                    anchor_pos, anchor_quat,
                    target_data['positions'],
                    target_data['quaternions'],
                    neg_dist_thresh
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
        description='Build triplet cache for metric learning from pose data'
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
    
    args = parser.parse_args()
    
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
        neg_dist_thresh=args.neg_dist
    )
    
    # Save cache
    save_triplet_cache(args.output, triplets, metadata, sequence_names)
    
    print("\nDone!")


if __name__ == '__main__':
    main()