import numpy as np
import os
from bisect import bisect_left
import torch
from torch.utils.data import Dataset

class EventExtractionDataset(Dataset): #Extracts events for a specific duration from stereo event Cameras
    def __init__(self, dataset_root, width=346, height=260, duration_us=5, max_events=2048):
        self.width = width 
        self.height = height
        self.duration_us = duration_us
        self.max_events = max_events

        self.sequences = []
        self._discover_sequences(dataset_root)

    def _discover_sequences(self, dataset_root):
        for scenario in os.listdir(dataset_root):
            scenario_path = os.path.join(dataset_root, scenario)
            if not os.path.isdir(scenario_path):
                continue

            for item in os.listdir(scenario_path):
                item_path = os.path.join(scenario_path, item)
                if os.path.isdir(item_path) and item != 'calibration':
                    required_files = [
                    'left_Events.npz', 'right_Events.npz', 'timestamps.txt', 'pose.txt'
                    ]
                if all(os.path.exists(os.path.join(item_path, f)) for f in required_files):
                    self.sequences.append({
                        'scenario': scenario,
                        'sequence': item,
                        'path': item_path,
                        'left_events_path': os.path.join(item_path, 'left_Events.npz'),
                        'right_events_path': os.path.join(item_path, 'right_Events.npz'),
                        'timestamps_path': os.path.join(item_path, 'timestamps.txt'),
                        'pose_path': os.path.join(item_path, 'pose.txt')
                    })

    def __len__(self):
        return len(self.sequences)
    
    def _load_and_align_data(self, seq_info):
        left_events = np.load(seq_info['left_events_path'], allow_pickle=True)['events']
        right_events = np.load(seq_info['right_events_path'], allow_pickle=True)['events']
        timestamps = np.loadtxt(seq_info['timestamps_path'])
        poses = np.loadtxt(seq_info['pose_path'])

        t0 = min(
            left_events['t'][0], right_events['t'][0], timestamps[0], poses[0, 0]
        )

        left_events = left_events.copy()
        right_events = right_events.copy()
        left_events['t'] -= t0
        right_events['t'] -= t0
        timestamps -= t0
        poses = poses.copy()
        poses[:, 0] -= t0

        return left_events, right_events, timestamps, poses
    
    def get_event_slice(self, events, duration_us=None):
        if duration_us is None:
            duration_us = self.duration_us
            
        if len(events) == 0:
            return events, 0.0, 0.0
        
        start_time = events['t'][0]
        end_time = start_time + duration_us / 1000000.0  # Convert µs to seconds
        
        # Select events within the window
        mask = (events['t'] >= start_time) & (events['t'] < end_time)
        selected = events[mask]
        
        return selected, start_time, end_time
        
    def _normalize_events(self, events, t_start, t_end):
        if len(events) == 0:
            return events
        
        dt = t_end - t_start
        if dt == 0:
            dt = 1.0

        norm_events = np.zeros(events.shape, dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
        norm_events['x'] = events['x'] / self.width
        norm_events['y'] = events['y'] / self.height
        norm_events['t'] = (events['t'] - t_start) / dt
        norm_events['p'] = events['p'].astype(np.float32)

        return norm_events
    
    def _find_pose_at_time(self, poses, target_time): #find closest pose
        if len(poses) == 0:
            return np.zeros(7, dtype=np.float32)

        idx = bisect_left(poses[:, 0], target_time)

        if idx == 0:
            return poses[0, 1:].astype(np.float32)
        elif idx >= len(poses):
            return poses[-1, 1:].astype(np.float32)
        else:
            before = poses[idx - 1]
            after = poses[idx]
            if abs(before[0] - target_time) < abs(after[0] - target_time):
                return before[1:].astype(np.float32)
            else:
                return after[1:].astype(np.float32)
    
    def events_to_tensor(self, events_array, max_events=None):
        """
        Convert numpy structured array to tensor format for PointNet++
        Args:
            events_array: numpy structured array with fields ['x', 'y', 't', 'p']
            max_events: maximum number of events to include (uses self.max_events if None)
        Returns:
            tensor: [max_events, 4] tensor ready for PointNet++
        """
        if max_events is None:
            max_events = self.max_events
            
        if len(events_array) == 0:
            return torch.zeros(max_events, 4, dtype=torch.float32)
        
        x = events_array['x'].astype(np.float32)
        y = events_array['y'].astype(np.float32) 
        t = events_array['t'].astype(np.float32)
        p = events_array['p'].astype(np.float32)
        
        events_tensor = np.stack([x, y, t, p], axis=1)  # [N, 4]
        
        if len(events_tensor) < max_events:
            padding = np.zeros((max_events - len(events_tensor), 4), dtype=np.float32)
            events_tensor = np.concatenate([events_tensor, padding], axis=0)
        elif len(events_tensor) > max_events:
            indices = np.random.choice(len(events_tensor), max_events, replace=False)
            events_tensor = events_tensor[indices]
        
        return torch.from_numpy(events_tensor)  # [max_events, 4]
    
    def to_pointnet_format(self):
        """
        Convert current sample to PointNet++ tensor format
        Returns:
            dict with 'left_tensor', 'right_tensor', 'pose', 'metadata'
        """
        # Get the current sample (need to call __getitem__ first)
        sample = self.__getitem__(0) if hasattr(self, '_current_sample') else None
        if sample is None:
            raise ValueError("Call __getitem__ first or use get_pointnet_sample(idx)")
        
        left_tensor = self.events_to_tensor(sample['left'])
        right_tensor = self.events_to_tensor(sample['right'])
        
        return {
            'left_tensor': left_tensor,
            'right_tensor': right_tensor, 
            'pose': torch.from_numpy(sample['pose']).float(),
            'metadata': {
                't_start': sample['t_start'],
                't_end': sample['t_end'],
                'sequence_info': sample['sequence_info']
            }
        }
    
    def get_pointnet_sample(self, idx):
        """
        Get a sample and convert it to PointNet++ format in one call
        Args:
            idx: sequence index
        Returns:
            dict with PointNet++ tensors and metadata
        """
        sample = self.__getitem__(idx)
        
        left_tensor = self.events_to_tensor(sample['left'])
        right_tensor = self.events_to_tensor(sample['right'])
        
        return {
            'left_tensor': left_tensor,      # [max_events, 4]
            'right_tensor': right_tensor,    # [max_events, 4] 
            'pose': torch.from_numpy(sample['pose']).float(),  # [7]
            'metadata': {
                't_start': sample['t_start'],
                't_end': sample['t_end'],
                'sequence_info': sample['sequence_info'],
                'left_event_count': len(sample['left']),
                'right_event_count': len(sample['right'])
            }
        }
            
    def __getitem__(self, idx): #extract initial events and return a dictionary with keys - left, right, pose, t_start, t_end, sequence_info
        seq_info = self.sequences[idx]

        left_events, right_events, timestamps, poses = self._load_and_align_data(seq_info)
        left_extracted, left_t_start, left_t_end = self.get_event_slice(left_events, self.duration_us)
        right_extracted, right_t_start, right_t_end = self.get_event_slice(right_events, self.duration_us)

        t_start = min(left_t_start, right_t_start)
        t_end = max(left_t_end, right_t_end)
        t_mid = (t_start +t_end) / 2.0

        left_norm = self._normalize_events(left_extracted, t_start, t_end)
        right_norm = self._normalize_events(right_extracted, t_start, t_end)

        pose = self._find_pose_at_time(poses, t_mid)

        return{
            'left': left_norm,
            'right': right_norm,
            'pose': pose,
            't_start': float(t_start),
            't_end': float(t_end),
            'sequence_info': {
                'scenario': seq_info['scenario'],
                'sequence': seq_info['sequence']
            }
        }

def extract_events(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)

def collate_pointnet_batch(batch_list):
    """
    Custom collate function for batching PointNet++ samples
    Args:
        batch_list: list of samples from get_pointnet_sample()
    Returns:
        batched tensors ready for PointNet++
    """
    left_tensors = torch.stack([sample['left_tensor'] for sample in batch_list])  # [B, max_events, 4]
    right_tensors = torch.stack([sample['right_tensor'] for sample in batch_list])  # [B, max_events, 4]
    poses = torch.stack([sample['pose'] for sample in batch_list])  # [B, 7]
    
    # Collect metadata
    metadata = {
        't_starts': [sample['metadata']['t_start'] for sample in batch_list],
        't_ends': [sample['metadata']['t_end'] for sample in batch_list],
        'sequence_infos': [sample['metadata']['sequence_info'] for sample in batch_list],
        'left_counts': [sample['metadata']['left_event_count'] for sample in batch_list],
        'right_counts': [sample['metadata']['right_event_count'] for sample in batch_list]
    }
    
    return {
        'left_events': left_tensors,
        'right_events': right_tensors,
        'poses': poses,
        'metadata': metadata
    }

if __name__ == "__main__":
    dataset_root = r"/home/adarsh/Documents/SRM/dataset/train" #change with actual path if required
    
    # Test with 5 microsecond chunks
    dataset = extract_events(dataset_root, duration_us=5, max_events=2048)

    print(f"found {len(dataset)} sequences")

    if len(dataset) > 0:
        # Test original format
        sample = dataset[0]
        print(f"Original sample from {sample['sequence_info']['scenario']}/{sample['sequence_info']['sequence']}:")
        print(f"left events: {len(sample['left'])}")
        print(f"right events: {len(sample['right'])}")
        print(f"Time Window: {sample['t_start']:.6f}s to {sample['t_end']:.6f}s ({(sample['t_end']-sample['t_start'])*1000000:.2f}μs)")
        print(f"Pose Shape: {sample['pose'].shape}")

        if len(sample['left']) > 0:
            print(f"left events - X range: [{sample['left']['x'].min():.3f}, {sample['left']['x'].max():.3f}]")
            print(f"left events - T range: [{sample['left']['t'].min():.3f}, {sample['left']['t'].max():.3f}]")
        
        print("\n" + "="*50)
        
        # Test PointNet++ format
        pointnet_sample = dataset.get_pointnet_sample(0)
        print(f"PointNet++ format:")
        print(f"Left tensor shape: {pointnet_sample['left_tensor'].shape}")
        print(f"Right tensor shape: {pointnet_sample['right_tensor'].shape}")
        print(f"Pose tensor shape: {pointnet_sample['pose'].shape}")
        print(f"Actual left events: {pointnet_sample['metadata']['left_event_count']}")
        print(f"Actual right events: {pointnet_sample['metadata']['right_event_count']}")
        
        # Test batching
        print(f"\nTesting batch processing...")
        batch_samples = [dataset.get_pointnet_sample(i) for i in range(min(3, len(dataset)))]
        batched = collate_pointnet_batch(batch_samples)
        print(f"Batched left events shape: {batched['left_events'].shape}")
        print(f"Batched right events shape: {batched['right_events'].shape}")
        print(f"Batched poses shape: {batched['poses'].shape}")