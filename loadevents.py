import numpy as np
import os
from bisect import bisect_left
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset

class EventExtractionDataset(Dataset):
    def __init__(self, dataset_root, window_duration=5000, stride=2500, max_events_per_strip=4096):
        """
        Args:
            dataset_root: Path to dataset
            window_duration: Duration of each window in microseconds
            stride: Stride between windows in microseconds
            max_events_per_strip: Maximum number of events per strip (for batching)
        """
        self.window_duration = window_duration
        self.stride = stride
        self.max_events_per_strip = max_events_per_strip
        self.sequences = []
        self.windows = []  # List of (seq_idx, window_start_time, window_end_time)
        
        self.discover_sequences(dataset_root)
        self.cache_metadata()
        self.compute_windows()
    
    def discover_sequences(self, dataset_root):
        for sequence in os.listdir(dataset_root):
            sequence_path = os.path.join(dataset_root, sequence)
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
        for seq_info in self.sequences:
            # Feature detection instead of string matching
            try:
                with h5py.File(seq_info['left_events_path'], "r") as e_left:
                    if 'ms_to_idx' in e_left:
                        seq_info['use_ms_idx'] = True
                    else:
                        seq_info['use_ms_idx'] = False
                    
                    # Try to get dimensions from HDF5 attributes or detect from data
                    if 'width' in e_left.attrs and 'height' in e_left.attrs:
                        seq_info['width'] = int(e_left.attrs['width'])
                        seq_info['height'] = int(e_left.attrs['height'])
                    else:
                        # Fallback to heuristic detection
                        max_x = np.max(e_left["events/x"][:1000])  # Sample first 1000 events
                        max_y = np.max(e_left["events/y"][:1000])
                        if max_x < 400 and max_y < 300:
                            seq_info['height'] = 260
                            seq_info['width'] = 346
                        else:
                            seq_info['height'] = 720
                            seq_info['width'] = 1280
                    
                    # Get dataset length for efficient indexing
                    seq_info['left_length'] = len(e_left["events/t"])
                    
                with h5py.File(seq_info['right_events_path'], "r") as e_right:
                    seq_info['right_length'] = len(e_right["events/t"])
                    
            except Exception as e:
                print(f"Warning: Could not read metadata for {seq_info['sequence']}: {e}")
                # Default values
                seq_info['use_ms_idx'] = False
                seq_info['height'] = 720
                seq_info['width'] = 1280
                seq_info['left_length'] = 1000000  # Default fallback
                seq_info['right_length'] = 1000000
            
            # Load and process poses
            poses_raw = np.loadtxt(seq_info['pose_path'])
            
            # Get time bounds from events and poses
            with h5py.File(seq_info['left_events_path'], "r") as e_left, \
                 h5py.File(seq_info['right_events_path'], "r") as e_right:
                
                # Read just first and last timestamps efficiently
                left_t_start = e_left["events/t"][0]
                left_t_end = e_left["events/t"][-1]
                right_t_start = e_right["events/t"][0]
                right_t_end = e_right["events/t"][-1]
                
                t0 = min(left_t_start, right_t_start, poses_raw[0, 0])
                t1 = max(left_t_end, right_t_end, poses_raw[-1, 0])
                
                seq_info['t0'] = t0
                seq_info['t1'] = t1
                seq_info['time_range'] = t1 - t0
                
                # Keep raw poses for time lookup
                seq_info['poses_raw'] = poses_raw.copy()
                
                # Create normalized poses for the network
                poses_norm = poses_raw.copy()
                poses_norm[:, 0] = (poses_raw[:, 0] - t0) / (t1 - t0)
                
                # Normalize quaternions to unit length
                quats = poses_norm[:, 4:8]
                quats_norm = quats / np.linalg.norm(quats, axis=1, keepdims=True)
                poses_norm[:, 4:8] = quats_norm
                
                seq_info['poses_norm'] = poses_norm

    def compute_windows(self):
        """Compute all valid windows across all sequences"""
        self.windows = []
        
        for seq_idx, seq_info in enumerate(self.sequences):
            t0 = seq_info['t0']
            t1 = seq_info['t1']
            
            # Compute number of windows for this sequence
            current_time = t0
            while current_time + self.window_duration <= t1:
                window_start = current_time
                window_end = current_time + self.window_duration
                self.windows.append((seq_idx, window_start, window_end))
                current_time += self.stride
            
            print(f"Sequence {seq_info['sequence']}: {len([w for w in self.windows if w[0] == seq_idx])} windows")

    def __len__(self):
        return len(self.windows)
    
    def binary_search_timestamps(self, h5_dataset, target_time, side='left'):
        """
        Binary search for timestamp without loading entire array into memory
        """
        dataset_length = len(h5_dataset)
        
        if dataset_length == 0:
            return 0
        
        # Check bounds first
        first_time = h5_dataset[0]
        last_time = h5_dataset[dataset_length - 1]
        
        if target_time <= first_time:
            return 0 if side == 'left' else 1
        if target_time >= last_time:
            return dataset_length - 1 if side == 'left' else dataset_length
        
        # Binary search
        left = 0
        right = dataset_length - 1
        
        while left < right:
            mid = (left + right) // 2
            mid_time = h5_dataset[mid]
            
            if mid_time < target_time:
                left = mid + 1
            else:
                right = mid
        
        if side == 'left':
            return left
        else:
            # For 'right' side, we want the first position where value > target_time
            if left < dataset_length and h5_dataset[left] == target_time:
                # Find the rightmost occurrence of target_time
                while left < dataset_length - 1 and h5_dataset[left + 1] == target_time:
                    left += 1
                return left + 1
            return left
    
    def load_and_slice_events(self, seq_info, t_start_raw, t_end_raw):
        """Load and slice events using raw timestamps"""
        with h5py.File(seq_info['left_events_path'], "r") as e_left, \
             h5py.File(seq_info['right_events_path'], "r") as e_right:
            
            # Use memory-efficient binary search instead of np.searchsorted
            left_start_idx = self.binary_search_timestamps(e_left["events/t"], t_start_raw, side='left')
            left_end_idx = self.binary_search_timestamps(e_left["events/t"], t_end_raw, side='right')
            right_start_idx = self.binary_search_timestamps(e_right["events/t"], t_start_raw, side='left')
            right_end_idx = self.binary_search_timestamps(e_right["events/t"], t_end_raw, side='right')
            
            # Extract left events
            if left_start_idx >= left_end_idx:
                left_events = np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
            else:
                left_x = e_left["events/x"][left_start_idx:left_end_idx]
                left_y = e_left["events/y"][left_start_idx:left_end_idx]
                left_t = e_left["events/t"][left_start_idx:left_end_idx]
                left_p = e_left["events/p"][left_start_idx:left_end_idx]
                
                # Create normalized events
                left_events = np.zeros(len(left_x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
                left_events['x'] = left_x / seq_info['width']
                left_events['y'] = left_y / seq_info['height']
                # Normalize time to [0, 1] within the window
                left_events['t'] = (left_t - t_start_raw) / (t_end_raw - t_start_raw)
                # Convert polarity to [-1, 1]
                left_events['p'] = left_p.astype(np.float32) * 2.0 - 1.0
            
            # Extract right events
            if right_start_idx >= right_end_idx:
                right_events = np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
            else:
                right_x = e_right["events/x"][right_start_idx:right_end_idx]
                right_y = e_right["events/y"][right_start_idx:right_end_idx]
                right_t = e_right["events/t"][right_start_idx:right_end_idx]
                right_p = e_right["events/p"][right_start_idx:right_end_idx]
                
                right_events = np.zeros(len(right_x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
                right_events['x'] = right_x / seq_info['width']
                right_events['y'] = right_y / seq_info['height']
                right_events['t'] = (right_t - t_start_raw) / (t_end_raw - t_start_raw)
                right_events['p'] = right_p.astype(np.float32) * 2.0 - 1.0
        
        return left_events, right_events
    
    def sample_or_pad_events(self, events):
        """Sample or pad events to fixed size"""
        n_events = len(events)
        
        if n_events == 0:
            # Return empty events with padding
            padded_events = np.zeros((self.max_events_per_strip, 4), dtype=np.float32)
            mask = np.zeros(self.max_events_per_strip, dtype=bool)
            return padded_events, mask
        
        if n_events >= self.max_events_per_strip:
            # Randomly sample max_events_per_strip events
            indices = np.random.choice(n_events, self.max_events_per_strip, replace=False)
            indices = np.sort(indices)  # Keep temporal order
            sampled_events = events[indices]
            
            # Convert to array format
            event_array = np.column_stack([
                sampled_events['x'], sampled_events['y'],
                sampled_events['t'], sampled_events['p']
            ]).astype(np.float32)
            
            mask = np.ones(self.max_events_per_strip, dtype=bool)
            
        else:
            # Pad with zeros
            event_array = np.column_stack([
                events['x'], events['y'],
                events['t'], events['p']
            ]).astype(np.float32)
            
            # Pad with zeros
            padded_events = np.zeros((self.max_events_per_strip, 4), dtype=np.float32)
            padded_events[:n_events] = event_array
            
            mask = np.zeros(self.max_events_per_strip, dtype=bool)
            mask[:n_events] = True
            
            event_array = padded_events
        
        return event_array, mask
    
    def find_pose(self, poses_raw, target_time_raw):
        """Find pose at target time using raw timestamps"""
        if len(poses_raw) == 0:
            return np.zeros(7, dtype=np.float32)
        
        # Use raw timestamps for lookup
        idx = bisect_left(poses_raw[:, 0], target_time_raw)
        
        if idx == 0:
            pose = poses_raw[0, 1:8].copy()  # tx, ty, tz, qx, qy, qz, qw
        elif idx >= len(poses_raw):
            pose = poses_raw[-1, 1:8].copy()
        else:
            pose_before = poses_raw[idx - 1]
            pose_after = poses_raw[idx]
            
            # Choose closest pose
            if abs(pose_before[0] - target_time_raw) < abs(pose_after[0] - target_time_raw):
                pose = pose_before[1:8].copy()
            else:
                pose = pose_after[1:8].copy()
        
        # Normalize quaternion
        quat = pose[3:7]  # qx, qy, qz, qw
        quat_norm = quat / np.linalg.norm(quat)
        pose[3:7] = quat_norm
        
        return pose.astype(np.float32)
            
    def __getitem__(self, idx):
        seq_idx, window_start, window_end = self.windows[idx]
        seq_info = self.sequences[seq_idx]
        
        # Load and slice events with raw timestamps
        left_events, right_events = self.load_and_slice_events(seq_info, window_start, window_end)
        
        # Sample or pad events to fixed size
        left_event_array, left_mask = self.sample_or_pad_events(left_events)
        right_event_array, right_mask = self.sample_or_pad_events(right_events)
        
        # Find poses at middle of window using raw timestamps
        mid_time_raw = (window_start + window_end) / 2.0
        left_pose = self.find_pose(seq_info['poses_raw'], mid_time_raw)
        right_pose = self.find_pose(seq_info['poses_raw'], mid_time_raw)  # Same for stereo
        
        # Debug info
        n_left_events = np.sum(left_mask)
        n_right_events = np.sum(right_mask)
        
        return {
            'left_events': left_event_array,  # (max_events_per_strip, 4)
            'right_events': right_event_array,  # (max_events_per_strip, 4)
            'left_mask': left_mask,  # (max_events_per_strip,)
            'right_mask': right_mask,  # (max_events_per_strip,)
            'left_pose': left_pose,  # (7,) [tx, ty, tz, qx, qy, qz, qw]
            'right_pose': right_pose,  # (7,)
            'n_left_events': n_left_events,
            'n_right_events': n_right_events,
            'window_start': window_start,
            'window_end': window_end,
            'sequence_info': {
                'sequence': seq_info['sequence'],
                'seq_idx': seq_idx
            }
        }

def event_extractor(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)

if __name__ == "__main__":
    dataset_root = r"//media/adarsh/One Touch/EventSLAM/dataset/train"
    dataset = event_extractor(dataset_root, window_duration=5000, stride=2500, max_events_per_strip=2048)
    print(f"Found {len(dataset)} windows across {len(dataset.sequences)} sequences")
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Sequence: {sample['sequence_info']['sequence']}")
        print(f"  Left events: {sample['n_left_events']}")
        print(f"  Right events: {sample['n_right_events']}")
        print(f"  Left pose: {sample['left_pose']}")
        print(f"  Time window: {sample['window_start']:.0f} - {sample['window_end']:.0f}")
        
        # Check event data structure
        print(f"  Left events shape: {sample['left_events'].shape}")
        print(f"  Left mask shape: {sample['left_mask'].shape}")
        
        # Check some events
        if sample['n_left_events'] > 0:
            print(f"  First few left events:")
            valid_events = sample['left_events'][sample['left_mask']]
            for j in range(min(5, len(valid_events))):
                x, y, t, p = valid_events[j]
                print(f"    Event {j}: x={x:.3f}, y={y:.3f}, t={t:.3f}, p={p:.1f}")