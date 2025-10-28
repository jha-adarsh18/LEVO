#!/usr/bin/env python3
"""
H5TripletDataset for efficient event-based metric learning.

Features:
- Supports both cached (events_cache.h5) and on-the-fly loading modes
- Cached mode: Direct loading from pre-computed events (fast)
- On-the-fly mode: Original behavior with H5 event files (backward compatible)
- Subset sampling: Train on a fraction of data for faster iteration
- Returns normalized events [1024, 4] with masks
"""

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class H5TripletDataset(Dataset):
    """
    Dataset for event-based triplet learning.
    
    Each sample returns a triplet of (anchor, positive, negative) events.
    Supports two modes:
    1. Cached mode: Load from pre-computed events_cache.h5 (fast)
    2. On-the-fly mode: Load from raw H5 event files (original behavior)
    """
    
    def __init__(self, 
                 triplet_cache_path,
                 data_root=None,
                 poses_h5_path=None,
                 event_window_ms=50,
                 n_events=1024,
                 camera='left',
                 use_cached_events=None,
                 subset_fraction=1.0):
        """
        Args:
            triplet_cache_path: Path to triplet_cache.h5 OR events_cache.h5
            data_root: Root directory containing sequence folders (only for on-the-fly mode)
            poses_h5_path: Path to poses.h5 (only for on-the-fly mode)
            event_window_ms: Time window in milliseconds to sample events
            n_events: Target number of events (pad/sample to this)
            camera: 'left' or 'right'
            use_cached_events: If None, auto-detect. If True, use cached mode. If False, use on-the-fly mode.
            subset_fraction: Fraction of dataset to use (0.0-1.0). Default 1.0 uses all data.
        """
        self.event_window_ms = event_window_ms
        self.n_events = n_events
        self.camera = camera
        self.subset_fraction = subset_fraction
        
        # Auto-detect mode if not specified
        if use_cached_events is None:
            use_cached_events = self._is_events_cache(triplet_cache_path)
        
        self.use_cached_events = use_cached_events
        
        if self.use_cached_events:
            # Cached mode: Load from events_cache.h5
            print(f"Using CACHED events mode from {triplet_cache_path}")
            self._init_cached_mode(triplet_cache_path)
        else:
            # On-the-fly mode: Original behavior
            print(f"Using ON-THE-FLY events mode")
            if data_root is None or poses_h5_path is None:
                raise ValueError("data_root and poses_h5_path required for on-the-fly mode")
            self.data_root = Path(data_root)
            self._init_onthefly_mode(triplet_cache_path, poses_h5_path)
        
        # Apply subset sampling if requested
        if subset_fraction < 1.0:
            self._create_subset()
    
    def _is_events_cache(self, path):
        """Detect if this is an events_cache.h5 file."""
        try:
            with h5py.File(path, 'r') as f:
                # Check for cached event datasets
                has_anchor_events = 'anchor_events' in f
                has_positive_events = 'positive_events' in f
                has_negative_events = 'negative_events' in f
                return has_anchor_events and has_positive_events and has_negative_events
        except:
            return False
    
    def _init_cached_mode(self, cache_path):
        """Initialize for cached events mode."""
        self.cache_path = cache_path
        self.cache_file = None  # Opened in worker_init_fn or lazily
        
        # Get dataset length
        with h5py.File(cache_path, 'r') as f:
            self.n_samples = f['anchor_events'].shape[0]
        
        print(f"Loaded {self.n_samples} cached triplets")
        
        # Initialize valid_indices to all samples (will be subsampled if needed)
        self.valid_indices = None
    
    def _init_onthefly_mode(self, triplet_cache_path, poses_h5_path):
        """Initialize for on-the-fly loading mode (original behavior)."""
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
        
        # Store poses file path (will be opened per-worker)
        self.poses_h5_path = poses_h5_path
        self.poses_file = None  # Opened in worker_init_fn or lazily
        
        # Build H5 file paths
        self.h5_paths = {}
        for seq_name in self.sequence_names:
            seq_dir = self.data_root / seq_name
            # Find event files with pattern *-events_left.h5 or *-events_right.h5
            event_files = list(seq_dir.glob(f"*-events_{self.camera}.h5"))
            if not event_files:
                # Try alternative naming
                event_files = list(seq_dir.glob(f"*_{self.camera}_events.h5"))
            if not event_files:
                event_files = list(seq_dir.glob(f"*{self.camera}*.h5"))
            
            if event_files:
                self.h5_paths[seq_name] = str(event_files[0])
            else:
                raise FileNotFoundError(f"No {self.camera} event file found in {seq_dir}")
        
        # H5 file handles (opened in worker_init_fn)
        self.h5_files = {}
        self.h5_data_cache = {}
        
        # MVSEC timestamp cache (loaded once per worker)
        self.mvsec_timestamps = {}
        
        # Pose cache (loaded once per worker)
        self.poses_cache = {}
    
    def _create_subset(self):
        """Create a random subset of the dataset."""
        if self.use_cached_events:
            # For cached mode
            n_subset = int(self.n_samples * self.subset_fraction)
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            self.valid_indices = np.sort(rng.choice(self.n_samples, n_subset, replace=False))
            print(f"Using subset: {len(self.valid_indices)}/{self.n_samples} samples ({self.subset_fraction:.1%})")
        else:
            # For on-the-fly mode
            n_total = len(self.triplet_indices)
            n_subset = int(n_total * self.subset_fraction)
            rng = np.random.RandomState(42)
            subset_idx = rng.choice(n_total, n_subset, replace=False)
            self.triplet_indices = self.triplet_indices[subset_idx]
            print(f"Using subset: {len(self.triplet_indices)}/{n_total} samples ({self.subset_fraction:.1%})")
    
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
    
    def _is_mvsec(self, seq_name):
        """Check if sequence is MVSEC (has indoor/outdoor in name)."""
        seq_lower = seq_name.lower()
        return 'indoor' in seq_lower or 'outdoor' in seq_lower
    
    def __len__(self):
        if self.use_cached_events:
            if self.valid_indices is not None:
                return len(self.valid_indices)
            return self.n_samples
        else:
            return len(self.triplet_indices)
    
    def _open_h5_files(self):
        """Open all H5 files. Called in worker_init_fn for persistent handles."""
        if self.use_cached_events:
            # Open cache file
            if self.cache_file is None:
                self.cache_file = h5py.File(self.cache_path, 'r')
        else:
            # Open event files
            for seq_name, h5_path in self.h5_paths.items():
                if seq_name not in self.h5_files:
                    self.h5_files[seq_name] = h5py.File(h5_path, 'r')
                    # Cache dataset references and metadata
                    self._cache_h5_structure(seq_name)
    
    def _cache_h5_structure(self, seq_name):
        """Cache H5 structure for faster access."""
        f = self.h5_files[seq_name]
        cache = {}
        
        # Find event datasets
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
        
        # For MVSEC: Load timestamps into RAM once per worker (small size)
        if self._is_mvsec(seq_name):
            print(f"Loading MVSEC timestamps for {seq_name} into RAM...")
            self.mvsec_timestamps[seq_name] = cache['t'][:]
    
    def _cache_poses(self):
        """Cache all poses/timestamps into RAM to avoid repeated H5 reads."""
        if self.poses_cache:  # Already cached
            return
        
        print("Caching poses and timestamps into RAM...")
        for seq_name in self.sequence_names:
            self.poses_cache[seq_name] = {
                'timestamps': self.poses_file[seq_name]['timestamps'][:],
                'positions': self.poses_file[seq_name]['positions'][:],
                'quaternions': self.poses_file[seq_name]['quaternions'][:]
            }
        print(f"Cached poses for {len(self.sequence_names)} sequences")
    
    def _close_h5_files(self):
        """Close all H5 files."""
        if self.use_cached_events:
            if self.cache_file is not None:
                self.cache_file.close()
                self.cache_file = None
        else:
            for f in self.h5_files.values():
                f.close()
            self.h5_files = {}
            self.h5_data_cache = {}
            self.mvsec_timestamps = {}
            self.poses_cache = {}
            if self.poses_file is not None:
                self.poses_file.close()
                self.poses_file = None
    
    def _load_events_at_timestamp(self, seq_name, timestamp_us):
        """
        Load events around a timestamp.
        Uses ms_to_idx for TUM-VIE (fast), searchsorted for MVSEC.
        
        Args:
            seq_name: Sequence name
            timestamp_us: Center timestamp in microseconds
            
        Returns:
            events: [N, 4] array of (x, y, t, p)
        """
        cache = self.h5_data_cache[seq_name]
        h5file = self.h5_files[seq_name]
        
        window_us = self.event_window_ms * 1000
        t_start = timestamp_us - window_us / 2
        t_end = timestamp_us + window_us / 2
        
        # Determine lookup method based on dataset type
        if self._is_mvsec(seq_name):
            # MVSEC: Use searchsorted on pre-loaded timestamps
            timestamps = self.mvsec_timestamps[seq_name]
            start_idx = np.searchsorted(timestamps, t_start, side='left')
            end_idx = np.searchsorted(timestamps, t_end, side='right')
        else:
            # TUM-VIE: Use ms_to_idx for fast lookup (no RAM loading)
            if 'ms_to_idx' not in h5file:
                raise KeyError(f"ms_to_idx not found in TUM-VIE file: {self.h5_paths[seq_name]}")
            
            ms_to_idx = h5file['ms_to_idx']
            ms_start = int(t_start / 1000)
            ms_end = int(t_end / 1000)
            
            # Bounds check
            if ms_start >= len(ms_to_idx):
                return np.array([], dtype=np.float32).reshape(0, 4)
            if ms_end >= len(ms_to_idx):
                ms_end = len(ms_to_idx) - 1
            
            start_idx = int(ms_to_idx[ms_start])
            end_idx = int(ms_to_idx[ms_end])
        
        # Bounds check
        if start_idx >= end_idx:
            return np.array([], dtype=np.float32).reshape(0, 4)
        
        # Extract events (only the slice we need)
        x = cache['x'][start_idx:end_idx]
        y = cache['y'][start_idx:end_idx]
        t = cache['t'][start_idx:end_idx]
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
        if self.use_cached_events:
            return self._getitem_cached(idx)
        else:
            return self._getitem_onthefly(idx)
    
    def _getitem_cached(self, idx):
        """Load from pre-computed events_cache.h5"""
        # Lazy initialization for single-process mode (num_workers=0)
        if self.cache_file is None:
            self.cache_file = h5py.File(self.cache_path, 'r')
        
        # Map to actual index if using subset
        if self.valid_indices is not None:
            actual_idx = self.valid_indices[idx]
        else:
            actual_idx = idx
        
        # Direct loading from cache (already normalized and processed)
        return {
            'anchor_events': torch.from_numpy(self.cache_file['anchor_events'][actual_idx]),
            'anchor_mask': torch.from_numpy(self.cache_file['anchor_mask'][actual_idx]),
            'positive_events': torch.from_numpy(self.cache_file['positive_events'][actual_idx]),
            'positive_mask': torch.from_numpy(self.cache_file['positive_mask'][actual_idx]),
            'negative_events': torch.from_numpy(self.cache_file['negative_events'][actual_idx]),
            'negative_mask': torch.from_numpy(self.cache_file['negative_mask'][actual_idx]),
            'anchor_pose': torch.from_numpy(self.cache_file['anchor_pose'][actual_idx]),
            'positive_pose': torch.from_numpy(self.cache_file['positive_pose'][actual_idx]),
            'negative_pose': torch.from_numpy(self.cache_file['negative_pose'][actual_idx]),
        }
    
    def _getitem_onthefly(self, idx):
        """Load events on-the-fly from raw H5 files (original behavior)"""
        # Lazy initialization for single-process mode (num_workers=0)
        if not self.h5_files:
            self._open_h5_files()
        if self.poses_file is None:
            self.poses_file = h5py.File(self.poses_h5_path, 'r')
            self._cache_poses()
        
        # Get triplet indices
        anchor_seq_idx, anchor_time_idx, pos_seq_idx, pos_time_idx, neg_seq_idx, neg_time_idx = self.triplet_indices[idx]
        
        # Get sequence names
        anchor_seq = self.sequence_names[anchor_seq_idx]
        pos_seq = self.sequence_names[pos_seq_idx]
        neg_seq = self.sequence_names[neg_seq_idx]
        
        # Get timestamps from cache (fast RAM access)
        anchor_timestamp = self.poses_cache[anchor_seq]['timestamps'][anchor_time_idx]
        pos_timestamp = self.poses_cache[pos_seq]['timestamps'][pos_time_idx]
        neg_timestamp = self.poses_cache[neg_seq]['timestamps'][neg_time_idx]
        
        # FIXED: Timestamps are already in microseconds - no conversion needed
        # The poses.h5 timestamps are already in microseconds to match event data
        
        # Load events
        anchor_events = self._load_events_at_timestamp(anchor_seq, anchor_timestamp)
        pos_events = self._load_events_at_timestamp(pos_seq, pos_timestamp)
        neg_events = self._load_events_at_timestamp(neg_seq, neg_timestamp)
        
        # Process events (with sequence-specific resolution)
        anchor_events, anchor_mask = self._process_events(anchor_events, anchor_seq)
        pos_events, pos_mask = self._process_events(pos_events, pos_seq)
        neg_events, neg_mask = self._process_events(neg_events, neg_seq)
        
        # Get poses from cache (fast RAM access)
        anchor_pose = np.concatenate([
            self.poses_cache[anchor_seq]['positions'][anchor_time_idx],
            self.poses_cache[anchor_seq]['quaternions'][anchor_time_idx]
        ])
        pos_pose = np.concatenate([
            self.poses_cache[pos_seq]['positions'][pos_time_idx],
            self.poses_cache[pos_seq]['quaternions'][pos_time_idx]
        ])
        neg_pose = np.concatenate([
            self.poses_cache[neg_seq]['positions'][neg_time_idx],
            self.poses_cache[neg_seq]['quaternions'][neg_time_idx]
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
        
        # For on-the-fly mode, also open poses file
        if not dataset.use_cached_events:
            dataset.poses_file = h5py.File(dataset.poses_h5_path, 'r')
            dataset._cache_poses()


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
                     data_root=None,
                     poses_h5_path=None,
                     batch_size=32,
                     num_workers=4,
                     use_cached_events=None,
                     **dataset_kwargs):
    """
    Create DataLoader with proper worker initialization.
    
    Example (cached mode with subset):
        train_loader = create_dataloader(
            'events_cache.h5',
            batch_size=32,
            num_workers=4,
            subset_fraction=0.3  # Use 30% of data
        )
    
    Example (on-the-fly mode):
        train_loader = create_dataloader(
            'triplet_cache.h5',
            data_root='/path/to/sequences',
            poses_h5_path='poses.h5',
            batch_size=32,
            num_workers=4,
            event_window_ms=50,
            n_events=1024,
            camera='left',
            subset_fraction=0.3  # Use 30% of data
        )
    """
    dataset = H5TripletDataset(
        triplet_cache_path,
        data_root=data_root,
        poses_h5_path=poses_h5_path,
        use_cached_events=use_cached_events,
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
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=8 if num_workers > 0 else None
    )
    
    return dataloader


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test H5TripletDataset')
    parser.add_argument('cache_path', type=str, help='Path to events_cache.h5 or triplet_cache.h5')
    parser.add_argument('--data-root', type=str, help='Root directory with sequences (for on-the-fly mode)')
    parser.add_argument('--poses-h5', type=str, help='Path to poses.h5 (for on-the-fly mode)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--mode', type=str, choices=['cached', 'onthefly', 'auto'], default='auto',
                       help='Mode: cached (use events_cache.h5), onthefly (load from raw H5), auto (detect)')
    parser.add_argument('--subset-fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (0.0-1.0, default: 1.0)')
    
    args = parser.parse_args()
    
    # Determine mode
    use_cached = None if args.mode == 'auto' else (args.mode == 'cached')
    
    # Create dataloader
    print("Creating dataloader...")
    loader = create_dataloader(
        args.cache_path,
        data_root=args.data_root,
        poses_h5_path=args.poses_h5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cached_events=use_cached,
        event_window_ms=50,
        n_events=1024,
        camera='left',
        subset_fraction=args.subset_fraction
    )
    
    # Test loading
    print(f"\nDataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")
    print(f"Mode: {'CACHED' if loader.dataset.use_cached_events else 'ON-THE-FLY'}")
    
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