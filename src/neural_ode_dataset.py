import numpy as np
import os
from bisect import bisect_left
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset

class EventODEDataset(Dataset):
    """
    Dataset for Neural ODE-based event SLAM that loads data in 5ms time slices.
    Only loads one batch/window into RAM at a time for memory efficiency.
    """
    def __init__(self, dataset_root, window_duration=1000, slice_duration=5, overlap=200):
        """
        Args:
            dataset_root: Path to dataset directory
            window_duration: Duration of each training window in milliseconds (default 1000ms = 1s)
            slice_duration: Duration of each ODE time slice in milliseconds (default 5ms)
            overlap: Overlap between consecutive windows in milliseconds (default 200ms)
        """
        self.dataset_root = dataset_root
        self.window_duration = window_duration
        self.slice_duration = slice_duration
        self.overlap = overlap
        self.stride = window_duration - overlap  # Stride between window starts
        
        self.sequences = []
        self.windows = []  # List of (sequence_idx, window_start_time) tuples
        
        self.discover_sequences()
        self.cache_metadata()
        self.generate_windows()
    
    def discover_sequences(self):
        """Discover all sequences in the dataset directory"""
        for sequence in os.listdir(self.dataset_root):
            sequence_path = os.path.join(self.dataset_root, sequence)
            if not os.path.isdir(sequence_path):
                continue
            
            left_path = None
            right_path = None
            pose_path = None
            
            for item in os.listdir(sequence_path):
                if "left" in item.lower():
                    left_path = os.path.join(sequence_path, item)
                elif "right" in item.lower():
                    right_path = os.path.join(sequence_path, item)
                else:
                    pose_path = os.path.join(sequence_path, item)
            
            if left_path and right_path and pose_path:
                self.sequences.append({
                    'sequence': sequence,
                    'path': sequence_path,
                    'left_events_path': left_path,  
                    'right_events_path': right_path,
                    'pose_path': pose_path
                })
    
    def cache_metadata(self):
        """Cache metadata for all sequences"""
        for seq_info in self.sequences:
            # Set camera parameters based on sequence type
            if "indoor" in seq_info['sequence'].lower() or "outdoor" in seq_info['sequence'].lower():
                seq_info['height'] = 260
                seq_info['width'] = 346
                seq_info['use_ms_idx'] = False
            else:
                seq_info['height'] = 720
                seq_info['width'] = 1280
                seq_info['use_ms_idx'] = True
            
            # Load poses
            poses = np.loadtxt(seq_info['pose_path'])
            
            # Get time range from events and poses
            with h5py.File(seq_info['left_events_path'], "r") as e_left, \
                 h5py.File(seq_info['right_events_path'], "r") as e_right:
                
                left_t_start = e_left["events/t"][0]
                left_t_end = e_left["events/t"][-1]
                right_t_start = e_right["events/t"][0]
                right_t_end = e_right["events/t"][-1]
                
                t0 = min(left_t_start, right_t_start, poses[0, 0])
                t1 = max(left_t_end, right_t_end, poses[-1, 0])
                
                seq_info['t0'] = t0
                seq_info['t1'] = t1
                seq_info['time_range'] = t1 - t0
                seq_info['poses'] = poses
    
    def generate_windows(self):
        """Generate all training windows across all sequences"""
        self.windows = []
        
        for seq_idx, seq_info in enumerate(self.sequences):
            # Generate windows with stride
            window_start = seq_info['t0']
            while window_start + self.window_duration <= seq_info['t1']:
                self.windows.append((seq_idx, window_start))
                window_start += self.stride
            
        print(f"Generated {len(self.windows)} windows from {len(self.sequences)} sequences")
    
    def slice_events_in_window(self, events, window_start, seq_info):
        """
        Slice events into 5ms bins within a window
        Returns list of event slices, each representing events in a 5ms interval
        """
        window_end = window_start + self.window_duration
        num_slices = self.window_duration // self.slice_duration
        
        slices = []
        
        for slice_idx in range(num_slices):
            slice_start = window_start + slice_idx * self.slice_duration
            slice_end = slice_start + self.slice_duration
            
            # Find events in this slice
            if seq_info['use_ms_idx']:
                # Use millisecond indexing for faster lookup
                with h5py.File(seq_info['left_events_path'], "r") as f:
                    ms_to_idx = f["ms_to_idx"][:]
                    
                ms_start = int(slice_start / 1000)
                ms_end = int(slice_end / 1000)
                
                if ms_start < len(ms_to_idx) and ms_end < len(ms_to_idx):
                    start_idx = ms_to_idx[ms_start]
                    end_idx = ms_to_idx[ms_end] if ms_end < len(ms_to_idx) else len(events)
                else:
                    start_idx = end_idx = len(events)
            else:
                # Use binary search for normalized timestamps
                slice_start_norm = (slice_start - seq_info['t0']) / seq_info['time_range']
                slice_end_norm = (slice_end - seq_info['t0']) / seq_info['time_range']
                
                start_idx = np.searchsorted(events['t'], slice_start_norm, side='left')
                end_idx = np.searchsorted(events['t'], slice_end_norm, side='right')
            
            # Extract events in this slice
            if start_idx < end_idx:
                slice_events = events[start_idx:end_idx]
                
                # Convert to tensor format [x, y, t_rel, polarity]
                if len(slice_events) > 0:
                    # Relative time within the slice (0 to slice_duration)
                    t_rel = (slice_events['t'] - slice_start_norm) * seq_info['time_range'] if not seq_info['use_ms_idx'] else (slice_events['t'] - slice_start)
                    t_rel = np.clip(t_rel, 0, self.slice_duration)  # Ensure within slice bounds
                    
                    event_tensor = np.column_stack([
                        slice_events['x'],
                        slice_events['y'], 
                        t_rel,
                        slice_events['p'].astype(np.float32)
                    ])
                else:
                    event_tensor = np.empty((0, 4), dtype=np.float32)
            else:
                event_tensor = np.empty((0, 4), dtype=np.float32)
            
            slices.append({
                'events': torch.FloatTensor(event_tensor),
                'slice_start': slice_start,
                'slice_end': slice_end,
                'slice_idx': slice_idx
            })
        
        return slices
    
    def get_poses_for_window(self, seq_info, window_start):
        """Get pose trajectory for the window duration"""
        window_end = window_start + self.window_duration
        
        # Find poses within the window
        poses = seq_info['poses']
        start_idx = np.searchsorted(poses[:, 0], window_start, side='left')
        end_idx = np.searchsorted(poses[:, 0], window_end, side='right')
        
        window_poses = poses[start_idx:end_idx]
        
        if len(window_poses) == 0:
            # No poses in window, use nearest pose
            nearest_idx = np.argmin(np.abs(poses[:, 0] - (window_start + self.window_duration/2)))
            window_poses = poses[nearest_idx:nearest_idx+1]
        
        # Normalize timestamps to be relative to window start
        window_poses = window_poses.copy()
        window_poses[:, 0] = window_poses[:, 0] - window_start
        
        return torch.FloatTensor(window_poses)
    
    def load_window_data(self, seq_idx, window_start):
        """Load and process data for a single window. Only loads what's needed into RAM."""
        seq_info = self.sequences[seq_idx]
        window_end = window_start + self.window_duration
        
        # Load events for this window only
        with h5py.File(seq_info['left_events_path'], "r") as e_left, \
             h5py.File(seq_info['right_events_path'], "r") as e_right:
            
            # Determine time range indices
            if seq_info['use_ms_idx']:
                ms_to_idx_left = e_left["ms_to_idx"][:]
                ms_to_idx_right = e_right["ms_to_idx"][:]
                
                ms_start = int(window_start / 1000)
                ms_end = int(window_end / 1000)
                
                left_start_idx = ms_to_idx_left[ms_start] if ms_start < len(ms_to_idx_left) else len(e_left["events/t"])-1
                left_end_idx = ms_to_idx_left[ms_end] if ms_end < len(ms_to_idx_left) else len(e_left["events/t"])
                right_start_idx = ms_to_idx_right[ms_start] if ms_start < len(ms_to_idx_right) else len(e_right["events/t"])-1
                right_end_idx = ms_to_idx_right[ms_end] if ms_end < len(ms_to_idx_right) else len(e_right["events/t"])
            else:
                window_start_norm = (window_start - seq_info['t0']) / seq_info['time_range']
                window_end_norm = (window_end - seq_info['t0']) / seq_info['time_range']
                
                left_start_idx = np.searchsorted(e_left["events/t"], window_start_norm, side='left')
                left_end_idx = np.searchsorted(e_left["events/t"], window_end_norm, side='right')
                right_start_idx = np.searchsorted(e_right["events/t"], window_start_norm, side='left')
                right_end_idx = np.searchsorted(e_right["events/t"], window_end_norm, side='right')
            
            # Load only the events in this window
            def load_events_slice(event_file, start_idx, end_idx):
                if start_idx >= end_idx:
                    return np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
                
                x = event_file["events/x"][start_idx:end_idx]
                y = event_file["events/y"][start_idx:end_idx]
                t = event_file["events/t"][start_idx:end_idx]
                p = event_file["events/p"][start_idx:end_idx]
                
                events = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
                events['x'] = x / seq_info['width']  # Normalize coordinates
                events['y'] = y / seq_info['height']
                events['t'] = t if seq_info['use_ms_idx'] else (t - seq_info['t0']) / seq_info['time_range']
                events['p'] = p.astype(np.uint8)
                
                return events
            
            left_events = load_events_slice(e_left, left_start_idx, left_end_idx)
            right_events = load_events_slice(e_right, right_start_idx, right_end_idx)
        
        # Slice events into 5ms bins
        left_slices = self.slice_events_in_window(left_events, window_start, seq_info)
        right_slices = self.slice_events_in_window(right_events, window_start, seq_info)
        
        # Get pose trajectory for this window
        poses = self.get_poses_for_window(seq_info, window_start)
        
        return {
            'left_slices': left_slices,
            'right_slices': right_slices,
            'poses': poses,
            'window_start': window_start,
            'window_duration': self.window_duration,
            'slice_duration': self.slice_duration,
            'sequence_info': {
                'sequence': seq_info['sequence'],
                'width': seq_info['width'],
                'height': seq_info['height']
            }
        }
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get a single training window with 5ms event slices"""
        seq_idx, window_start = self.windows[idx]
        return self.load_window_data(seq_idx, window_start)


def collate_fn(batch):
    """
    Custom collate function for Neural ODE event data
    Handles variable-length sequences and slice structures
    """
    batch_data = {
        'left_slices': [],
        'right_slices': [],
        'poses': [],
        'window_info': []
    }
    
    max_slices = 0
    
    # Find maximum number of slices across batch
    for sample in batch:
        max_slices = max(max_slices, len(sample['left_slices']))
    
    # Process each sample in batch
    for sample in batch:
        left_slices = sample['left_slices']
        right_slices = sample['right_slices']
        
        # Pad sequences to max_slices if needed
        while len(left_slices) < max_slices:
            left_slices.append({
                'events': torch.FloatTensor(np.empty((0, 4))),
                'slice_start': 0,
                'slice_end': 0,
                'slice_idx': len(left_slices)
            })
        
        while len(right_slices) < max_slices:
            right_slices.append({
                'events': torch.FloatTensor(np.empty((0, 4))),
                'slice_start': 0,
                'slice_end': 0,  
                'slice_idx': len(right_slices)
            })
        
        batch_data['left_slices'].append(left_slices)
        batch_data['right_slices'].append(right_slices)
        batch_data['poses'].append(sample['poses'])
        batch_data['window_info'].append(sample['sequence_info'])
    
    return batch_data


# Usage example
if __name__ == "__main__":
    dataset_root = "//media/adarsh/One Touch/EventSLAM/dataset/train"
    
    # Create dataset with 1 second windows, 5ms slices, 200ms overlap
    dataset = EventODEDataset(
        dataset_root=dataset_root,
        window_duration=1000,  # 1 second windows
        slice_duration=5,      # 5ms slices  
        overlap=200           # 200ms overlap between windows
    )
    
    print(f"Dataset contains {len(dataset)} training windows")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"Window duration: {sample['window_duration']}ms")
    print(f"Number of left slices: {len(sample['left_slices'])}")
    print(f"Number of right slices: {len(sample['right_slices'])}")
    print(f"Poses shape: {sample['poses'].shape}")
    
    # Check first slice
    if len(sample['left_slices'][0]['events']) > 0:
        print(f"First left slice has {len(sample['left_slices'][0]['events'])} events")
        print(f"First event: {sample['left_slices'][0]['events'][0]}")