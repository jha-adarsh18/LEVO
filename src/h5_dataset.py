#!/usr/bin/env python3
"""
H5TripletDataset for efficient event-based metric learning.

Features:
- Opens H5 files in worker_init_fn (persistent across batches)
- Loads pre-cached triplet indices
- Samples events from H5 using binary search (searchsorted)
- Returns normalized events [1024, 4] with masks
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class H5TripletDataset(Dataset):
    """
    Dataset for event-based triplet learning.
    
    Each sample returns a triplet of (anchor, positive, negative) events.
    Events are sampled around timestamps from the triplet cache.
    """
    
    def __init__(self, 
                 triplet_cache_path,
                 data_root,
                 poses_h5_path,
                 event_window_ms=50,
                 n_events=1024,
                 camera='left'):
        """
        Args:
            triplet_cache_path: Path to triplet_cache.h5
            data_root: Root directory containing sequence folders
            poses_h5_path: Path to poses.h5
            event_window_ms: Time window in milliseconds to sample events
            n_events: Target number of events (pad/sample to this)
            camera: 'left' or 'right'
        
        Note: Image resolution is automatically determined per sequence:
            - MVSEC (indoor/outdoor): 346 x 260
            - TUM-VIE (others): 1280 x 720
        """
        self.data_root = Path(data_root)
        self.event_window_ms = event_window_ms
        self.n_events = n_events
        self.camera = camera
        
        # Resolution per sequence (will be populated)
        self.sequence_resolutions = {}
        
        # Load triplet cache
        print(f"Loading triplet cache from {triplet_cache_path}...")
        with h5py.File(triplet_cache_path, 'r') as f:
            self.triplet_indices = f['triplets/indices'][:]
            self.sequence_names = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                   for s in f['sequence_names'][:]]
            
            # Load metadata
            if 'metadata' in f['triplets'].attrs:
                self.metadata = json.loads(f['triplets'].attrs['metadata'])
            else:
                self.metadata = {}
        
        print(f"Loaded {len(self.triplet_indices)} triplets from {len(self.sequence_names)} sequences")
        
        # Determine resolution for each sequence
        self._setup_sequence_resolutions()
        
        # Load poses for timestamp lookup
        print(f"Loading poses from {poses_h5_path}...")
        self.poses = {}
        with h5py.File(poses_h5_path, 'r') as f:
            for seq_name in self.sequence_names:
                self.poses[seq_name] = {
                    'timestamps': f[seq_name]['timestamps'][:],
                    'positions': f[seq_name]['positions'][:],
                    'quaternions': f[seq_name]['quaternions'][:]
                }
        
        # Build H5 file paths
        self.h5_paths = {}
        for seq_name in self.sequence_names:
            seq_dir = self.data_root / seq_name
            # Find event files with pattern *-events_left.h5 or *-events_right.h5
            event_files = list(seq_dir.glob(f"*-events_{camera}.h5"))
            if not event_files:
                # Try alternative naming
                event_files = list(seq_dir.glob(f"*_{camera}_events.h5"))
            if not event_files:
                event_files = list(seq_dir.glob(f"*{camera}*.h5"))
            
            if event_files:
                self.h5_paths[seq_name] = str(event_files[0])
            else:
                raise FileNotFoundError(f"No {camera} event file found in {seq_dir}")
        
        # H5 file handles (opened in worker_init_fn)
        self.h5_files = {}
        self.h5_data_cache = {}
    
    def _setup_sequence_resolutions(self):
        """Determine resolution for each sequence based on dataset type."""
        for seq_name in self.sequence_names:
            seq_lower = seq_name.lower()
            
            # MVSEC sequences contain 'indoor' or 'outdoor'
            if 'indoor' in seq_lower or 'outdoor' in seq_lower:
                self.sequence_resolutions[seq_name] = (346, 260)  # (width, height)
            else:
                # TUM-VIE sequences
                self.sequence_resolutions[seq_name] = (1280, 720)  # (width, height)
        
        print(f"Sequence resolutions:")
        for seq_name, (w, h) in self.sequence_resolutions.items():
            dataset_type = "MVSEC" if w == 346 else "TUM-VIE"
            print(f"  {seq_name}: {w}x{h} ({dataset_type})")
    
    def _get_resolution(self, seq_name):
        """Get image resolution for a sequence."""
        return self.sequence_resolutions[seq_name]
    
    def __len__(self):
        return len(self.triplet_indices)
    
    def _open_h5_files(self):
        """Open all H5 files. Called in worker_init_fn for persistent handles."""
        for seq_name, h5_path in self.h5_paths.items():
            if seq_name not in self.h5_files:
                self.h5_files[seq_name] = h5py.File(h5_path, 'r')
                # Cache dataset references and metadata
                self._cache_h5_structure(seq_name)
    
    def _cache_h5_structure(self, seq_name):
        """Cache H5 structure for faster access."""
        f = self.h5_files[seq_name]
        
        # Try to find event data with common keys
        data_keys = ['events/x', 'events/y', 'events/t', 'events/p',
                     'events/ts', 'x', 'y', 't', 'p', 'ts']
        
        cache = {}
        
        # Find datasets
        for base_key in ['events/', '']:
            for coord in ['x', 'y', 't', 'ts', 'p', 'pol', 'polarity']:
                key = f"{base_key}{coord}"
                if key in f:
                    if coord in ['t', 'ts']:
                        cache['t'] = f[key]
                    elif coord == 'x':
                        cache['x'] = f[key]
                    elif coord == 'y':
                        cache['y'] = f[key]
                    elif coord in ['p', 'pol', 'polarity']:
                        cache['p'] = f[key]
        
        if 't' not in cache:
            raise KeyError(f"Could not find timestamp data in {self.h5_paths[seq_name]}")
        
        self.h5_data_cache[seq_name] = cache
    
    def _close_h5_files(self):
        """Close all H5 files."""
        for f in self.h5_files.values():
            f.close()
        self.h5_files = {}
        self.h5_data_cache = {}
    
    def _load_events_at_timestamp(self, seq_name, timestamp_us):
        """
        Load events around a timestamp.
        
        Args:
            seq_name: Sequence name
            timestamp_us: Center timestamp in microseconds
            
        Returns:
            events: [N, 4] array of (x, y, t, p)
        """
        cache = self.h5_data_cache[seq_name]
        timestamps = cache['t']
        
        # Convert window from ms to us
        window_us = self.event_window_ms * 1000
        t_start = timestamp_us - window_us / 2
        t_end = timestamp_us + window_us / 2
        
        # Use searchsorted for efficient lookup
        start_idx = np.searchsorted(timestamps, t_start, side='left')
        end_idx = np.searchsorted(timestamps, t_end, side='right')
        
        # Extract events
        x = cache['x'][start_idx:end_idx]
        y = cache['y'][start_idx:end_idx]
        t = timestamps[start_idx:end_idx]
        p = cache['p'][start_idx:end_idx] if 'p' in cache else np.ones(len(x))
        
        # Stack into [N, 4]
        events = np.stack([x, y, t, p], axis=1).astype(np.float32)
        
        return events
    
    def _process_events(self, events, seq_name):
        """
        Process events: sample/pad to n_events, normalize coordinates.
        
        Args:
            events: [N, 4] array of (x, y, t, p)
            seq_name: Sequence name for resolution lookup
            
        Returns:
            processed_events: [n_events, 4] normalized events
            mask: [n_events] binary mask (1 for valid events, 0 for padding)
        """
        n = len(events)
        
        if n == 0:
            # No events: return zeros with mask all False
            return np.zeros((self.n_events, 4), dtype=np.float32), np.zeros(self.n_events, dtype=np.float32)
        
        # Sample or pad
        if n > self.n_events:
            # Random sampling
            indices = np.random.choice(n, self.n_events, replace=False)
            indices = np.sort(indices)  # Keep temporal order
            sampled_events = events[indices]
            mask = np.ones(self.n_events, dtype=np.float32)
        else:
            # Pad with zeros
            sampled_events = np.zeros((self.n_events, 4), dtype=np.float32)
            sampled_events[:n] = events
            mask = np.zeros(self.n_events, dtype=np.float32)
            mask[:n] = 1.0
        
        # Get resolution for this sequence
        image_width, image_height = self._get_resolution(seq_name)
        
        # Normalize coordinates
        sampled_events[:, 0] /= image_width   # x
        sampled_events[:, 1] /= image_height  # y
        
        # Normalize time within the window
        if n > 0:
            t_min = sampled_events[:n, 2].min()
            t_max = sampled_events[:n, 2].max()
            if t_max > t_min:
                sampled_events[:n, 2] = (sampled_events[:n, 2] - t_min) / (t_max - t_min)
            else:
                sampled_events[:n, 2] = 0.5  # All events at same time
        
        # Polarity is already 0/1, keep as is
        
        return sampled_events, mask
    
    def __getitem__(self, idx):
        """
        Get triplet of (anchor, positive, negative) events.
        
        Returns:
            dict with keys:
                'anchor_events': [n_events, 4]
                'anchor_mask': [n_events]
                'positive_events': [n_events, 4]
                'positive_mask': [n_events]
                'negative_events': [n_events, 4]
                'negative_mask': [n_events]
                'anchor_pose': [7] (position + quaternion)
                'positive_pose': [7]
                'negative_pose': [7]
        """
        # Get triplet indices
        anchor_seq_idx, anchor_time_idx, pos_seq_idx, pos_time_idx, neg_seq_idx, neg_time_idx = self.triplet_indices[idx]
        
        # Get sequence names
        anchor_seq = self.sequence_names[anchor_seq_idx]
        pos_seq = self.sequence_names[pos_seq_idx]
        neg_seq = self.sequence_names[neg_seq_idx]
        
        # Get timestamps (convert to microseconds if needed)
        anchor_timestamp = self.poses[anchor_seq]['timestamps'][anchor_time_idx]
        pos_timestamp = self.poses[pos_seq]['timestamps'][pos_time_idx]
        neg_timestamp = self.poses[neg_seq]['timestamps'][neg_time_idx]
        
        # Ensure timestamps are in microseconds (common event camera format)
        # If timestamps are in seconds, convert to microseconds
        if anchor_timestamp < 1e9:  # Likely in seconds
            anchor_timestamp *= 1e6
            pos_timestamp *= 1e6
            neg_timestamp *= 1e6
        
        # Load events
        anchor_events = self._load_events_at_timestamp(anchor_seq, anchor_timestamp)
        pos_events = self._load_events_at_timestamp(pos_seq, pos_timestamp)
        neg_events = self._load_events_at_timestamp(neg_seq, neg_timestamp)
        
        # Process events (with sequence-specific resolution)
        anchor_events, anchor_mask = self._process_events(anchor_events, anchor_seq)
        pos_events, pos_mask = self._process_events(pos_events, pos_seq)
        neg_events, neg_mask = self._process_events(neg_events, neg_seq)
        
        # Get poses
        anchor_pose = np.concatenate([
            self.poses[anchor_seq]['positions'][anchor_time_idx],
            self.poses[anchor_seq]['quaternions'][anchor_time_idx]
        ])
        pos_pose = np.concatenate([
            self.poses[pos_seq]['positions'][pos_time_idx],
            self.poses[pos_seq]['quaternions'][pos_time_idx]
        ])
        neg_pose = np.concatenate([
            self.poses[neg_seq]['positions'][neg_time_idx],
            self.poses[neg_seq]['quaternions'][neg_time_idx]
        ])
        
        return {
            'anchor_events': torch.from_numpy(anchor_events),
            'anchor_mask': torch.from_numpy(anchor_mask),
            'positive_events': torch.from_numpy(pos_events),
            'positive_mask': torch.from_numpy(pos_mask),
            'negative_events': torch.from_numpy(neg_events),
            'negative_mask': torch.from_numpy(neg_mask),
            'anchor_pose': torch.from_numpy(anchor_pose).float(),
            'positive_pose': torch.from_numpy(pos_pose).float(),
            'negative_pose': torch.from_numpy(neg_pose).float(),
        }


def worker_init_fn(worker_id):
    """Initialize worker with persistent H5 file handles."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._open_h5_files()


def collate_fn(batch):
    """Custom collate function to stack batch items."""
    return {
        'anchor_events': torch.stack([item['anchor_events'] for item in batch]),
        'anchor_mask': torch.stack([item['anchor_mask'] for item in batch]),
        'positive_events': torch.stack([item['positive_events'] for item in batch]),
        'positive_mask': torch.stack([item['positive_mask'] for item in batch]),
        'negative_events': torch.stack([item['negative_events'] for item in batch]),
        'negative_mask': torch.stack([item['negative_mask'] for item in batch]),
        'anchor_pose': torch.stack([item['anchor_pose'] for item in batch]),
        'positive_pose': torch.stack([item['positive_pose'] for item in batch]),
        'negative_pose': torch.stack([item['negative_pose'] for item in batch]),
    }


def create_dataloader(triplet_cache_path,
                     data_root,
                     poses_h5_path,
                     batch_size=32,
                     num_workers=4,
                     **dataset_kwargs):
    """
    Create DataLoader with proper worker initialization.
    
    Example:
        train_loader = create_dataloader(
            'triplet_cache.h5',
            '/path/to/sequences',
            'poses.h5',
            batch_size=32,
            num_workers=4,
            event_window_ms=50,
            n_events=1024,
            camera='left'
        )
    
    Note: Image resolution is automatically determined per sequence.
    """
    dataset = H5TripletDataset(
        triplet_cache_path,
        data_root,
        poses_h5_path,
        **dataset_kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test H5TripletDataset')
    parser.add_argument('triplet_cache', type=str, help='Path to triplet_cache.h5')
    parser.add_argument('data_root', type=str, help='Root directory with sequences')
    parser.add_argument('poses_h5', type=str, help='Path to poses.h5')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    
    args = parser.parse_args()
    
    # Create dataloader
    print("Creating dataloader...")
    loader = create_dataloader(
        args.triplet_cache,
        args.data_root,
        args.poses_h5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        event_window_ms=50,
        n_events=1024,
        camera='left'
    )
    
    # Test loading
    print(f"\nDataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    print("\nLoading first batch...")
    batch = next(iter(loader))
    
    print("\nBatch contents:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}, dtype={value.dtype}")
    
    print("\nEvent statistics (first sample):")
    print(f"  Anchor events range: x=[{batch['anchor_events'][0, :, 0].min():.3f}, {batch['anchor_events'][0, :, 0].max():.3f}], "
          f"y=[{batch['anchor_events'][0, :, 1].min():.3f}, {batch['anchor_events'][0, :, 1].max():.3f}]")
    print(f"  Anchor mask sum: {batch['anchor_mask'][0].sum().item()}/{batch['anchor_mask'][0].shape[0]} valid events")
    
    print("\nTest successful!")