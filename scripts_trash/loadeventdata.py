import numpy as np
import os
from bisect import bisect_left
import torch
from torch.utils.data import  Dataset

class EventExtractionDataset(Dataset): #Extracts events for N micro seconds
    def __init__(self, dataset_root, width=346, height=260, max_events=1000, min_duration_ms=5.0):
        self.width = width 
        self.height = height
        self.max_events = max_events
        self.min_duration_sec = min_duration_ms / 1000.0

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
    
    def _extract_initial_events(self, events):
        if len(events) == 0:
            return events, 0.0, 0.0
        
        first_n_events = events[:self.max_events] if len(events) >= self.max_events else events
        
        if len(first_n_events) > 0:
            t_start = events['t'][0]
            t_end_n_events = first_n_events['t'][-1]
            duration_n_events = t_end_n_events - t_start

            if duration_n_events < self.min_duration_sec:
                t_end_min_duration = t_start + self.min_duration_sec

                mask = events['t'] <= t_end_min_duration
                events_min_duration = events[mask]

                if len(events_min_duration) > len(first_n_events):
                    return events_min_duration, t_start, t_end_min_duration
                else:
                    return first_n_events, t_start, t_end_n_events
            else:
                return first_n_events, t_start, t_end_n_events
        else:
            return events, 0.0, 0.0
        
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
            
    def __getitem__(self, idx): #extract initial events and return a dictionary with keys - left, right, pose, t_start, t_end, sequence_info
        seq_info = self.sequences[idx]

        left_events, right_events, timestamps, poses = self._load_and_align_data(seq_info)
        left_extracted, left_t_start, left_t_end = self._extract_initial_events(left_events)
        right_extracted, right_t_start, right_t_end = self._extract_initial_events(right_events)

        t_start = min(left_t_start, right_t_start)
        t_end = max(left_t_end, right_t_end)
        t_mid = (t_start +t_end) / 2.0

        left_norm = self._normalize_events(left_extracted, t_start, t_end)
        right_norm = self. _normalize_events(right_extracted, t_start, t_end)

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

if __name__ == "__main__":
    dataset_root = r"/home/adarsh/Documents/SRM/dataset/train" #change with actual path if required
    dataset = extract_events(dataset_root)

    print(f"found {len(dataset)} sequences")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"sample from {sample['sequence_info']['scenario']}/{sample['sequence_info']['sequence']}:")
        print(f"left events: {len(sample['left'])}")
        print(f"right events: {len(sample['right'])}")
        print(f"Time Window: {sample['t_start']:.6f}s to {sample['t_end']:.6f}s ({(sample['t_end']-sample['t_start'])*1000:.2f}ms)")
        print(f"Pose Shape: {sample['pose'].shape}")

        if len(sample['left']) > 0:
            print(f"left events - X range: [{sample['left']['x'].min():.3f}, {sample['left']['x'].max():.3f}]")
            print(f"left events - T range: [{sample['left']['t'].min():.3f}, {sample['left']['t'].max():.3f}]")