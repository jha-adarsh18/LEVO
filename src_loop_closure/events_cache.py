#!/usr/bin/env python3
"""Pre-cache all event windows into sequential H5 for fast training (Multi-process)"""
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import os

# Import your current dataset
from h5_dataset import H5TripletDataset

# Global variables for worker processes
_dataset = None
_triplet_cache_path = None
_data_root = None
_poses_h5_path = None
_event_window_ms = None
_n_events = None
_camera = None


def init_worker(triplet_cache_path, data_root, poses_h5_path, event_window_ms, n_events, camera):
    """Initialize worker process with dataset"""
    global _dataset, _triplet_cache_path, _data_root, _poses_h5_path
    global _event_window_ms, _n_events, _camera
    
    _triplet_cache_path = triplet_cache_path
    _data_root = data_root
    _poses_h5_path = poses_h5_path
    _event_window_ms = event_window_ms
    _n_events = n_events
    _camera = camera
    
    # Create dataset in worker
    _dataset = H5TripletDataset(triplet_cache_path, data_root, poses_h5_path,
                                event_window_ms=event_window_ms,
                                n_events=n_events,
                                camera=camera)
    _dataset._open_h5_files()
    _dataset.poses_file = h5py.File(poses_h5_path, 'r')
    _dataset._cache_poses()


def process_sample(i):
    """Process a single sample - called by worker"""
    try:
        sample = _dataset[i]
        return i, {
            'anchor_events': sample['anchor_events'].numpy(),
            'anchor_mask': sample['anchor_mask'].numpy(),
            'positive_events': sample['positive_events'].numpy(),
            'positive_mask': sample['positive_mask'].numpy(),
            'negative_events': sample['negative_events'].numpy(),
            'negative_mask': sample['negative_mask'].numpy(),
            'anchor_pose': sample['anchor_pose'].numpy(),
            'positive_pose': sample['positive_pose'].numpy(),
            'negative_pose': sample['negative_pose'].numpy(),
        }
    except Exception as e:
        return i, None


def create_sequential_cache(triplet_cache_path, data_root, poses_h5_path, output_path, 
                           event_window_ms=50, n_events=1024, camera='left', num_workers=None):
    """Pre-load all event windows into one sequential file with multi-processing"""
    
    if num_workers is None:
        num_workers = min(32, cpu_count())
    
    print(f"Using {num_workers} workers for parallel processing")
    
    print("Loading dataset (main process)...")
    dataset = H5TripletDataset(triplet_cache_path, data_root, poses_h5_path,
                               event_window_ms=event_window_ms, 
                               n_events=n_events, 
                               camera=camera)
    
    n = len(dataset)
    print(f"Caching {n} triplets to {output_path}...")
    
    # Create output file with pre-allocated arrays
    print("Creating output file...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('anchor_events', (n, n_events, 4), dtype='float32', 
                        chunks=(1, n_events, 4), compression='gzip', compression_opts=4)
        f.create_dataset('anchor_mask', (n, n_events), dtype='float32', 
                        chunks=(1, n_events), compression='gzip', compression_opts=4)
        f.create_dataset('positive_events', (n, n_events, 4), dtype='float32', 
                        chunks=(1, n_events, 4), compression='gzip', compression_opts=4)
        f.create_dataset('positive_mask', (n, n_events), dtype='float32', 
                        chunks=(1, n_events), compression='gzip', compression_opts=4)
        f.create_dataset('negative_events', (n, n_events, 4), dtype='float32', 
                        chunks=(1, n_events, 4), compression='gzip', compression_opts=4)
        f.create_dataset('negative_mask', (n, n_events), dtype='float32', 
                        chunks=(1, n_events), compression='gzip', compression_opts=4)
        f.create_dataset('anchor_pose', (n, 7), dtype='float32')
        f.create_dataset('positive_pose', (n, 7), dtype='float32')
        f.create_dataset('negative_pose', (n, 7), dtype='float32')
    
    # Process in parallel
    print("Processing events in parallel...")
    with Pool(processes=num_workers, 
              initializer=init_worker,
              initargs=(triplet_cache_path, data_root, poses_h5_path, 
                       event_window_ms, n_events, camera)) as pool:
        
        # Process with progress bar
        results = []
        with tqdm(total=n) as pbar:
            for result in pool.imap_unordered(process_sample, range(n), chunksize=100):
                results.append(result)
                pbar.update(1)
    
    # Write results to file
    print("\nWriting results to file...")
    failed = []
    with h5py.File(output_path, 'r+') as f:
        for i, data in tqdm(results, desc="Writing"):
            if data is None:
                failed.append(i)
                continue
            f['anchor_events'][i] = data['anchor_events']
            f['anchor_mask'][i] = data['anchor_mask']
            f['positive_events'][i] = data['positive_events']
            f['positive_mask'][i] = data['positive_mask']
            f['negative_events'][i] = data['negative_events']
            f['negative_mask'][i] = data['negative_mask']
            f['anchor_pose'][i] = data['anchor_pose']
            f['positive_pose'][i] = data['positive_pose']
            f['negative_pose'][i] = data['negative_pose']
    
    if failed:
        print(f"\nWarning: {len(failed)} samples failed to process")
        print(f"Failed indices: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    print(f"\n✓ Successfully cached {n - len(failed)}/{n} triplets to {output_path}")
    
    # Print file size
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"✓ Cache file size: {size_gb:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-cache event windows for fast training (parallel)')
    parser.add_argument('--triplet-cache', type=str, required=True,
                        help='Path to triplet_cache.h5')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing sequence folders')
    parser.add_argument('--poses-h5', type=str, required=True,
                        help='Path to poses.h5')
    parser.add_argument('--output', type=str, default='events_cache.h5',
                        help='Output cache file (default: events_cache.h5)')
    parser.add_argument('--camera', type=str, default='left',
                        choices=['left', 'right'],
                        help='Camera to use (default: left)')
    parser.add_argument('--event-window-ms', type=int, default=50,
                        help='Event window in milliseconds (default: 50)')
    parser.add_argument('--n-events', type=int, default=1024,
                        help='Number of events per sample (default: 1024)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: min(32, cpu_count))')
    
    args = parser.parse_args()
    
    create_sequential_cache(
        args.triplet_cache,
        args.data_root,
        args.poses_h5,
        args.output,
        event_window_ms=args.event_window_ms,
        n_events=args.n_events,
        camera=args.camera,
        num_workers=args.num_workers
    )