import numpy as np
import os
from bisect import bisect_left
import torch
from torch.utils.data import Dataset

class EventExtractionDataset(Dataset):
    def __init__(self, dataset_root, width = 346, height = 260, N = 1024):
        self.width = width
        self.height = height
        self.N = N
        self.sequences = []
        self.discover_sequences(dataset_root)
    
    def discover_sequences(self, dataset_root):
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
    
    def load_and_normalize(self, seq_info):
        left_events = np.load(seq_info['left_events_path'], allow_pickle=True)['events']
        right_events = np.load(seq_info['right_events_path'], allow_pickle=True)['events']
        poses = np.loadtxt(seq_info['pose_path'])
        t0 = min(left_events['t'][0], right_events['t'][0], poses[0, 0])
        t1 = max(left_events['t'][-1], right_events['t'][-1], poses[-1, 0])
        poses[:, 0] = (poses[:, 0] - t0) / (t1 - t0)
        left_norm_events = np.zeros(left_events.shape, dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
        right_norm_events = np.zeros(right_events.shape, dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
        left_norm_events['x'] = left_events['x'] /  self.width
        right_norm_events['x'] = right_events['x'] / self.width
        left_norm_events['y'] = left_events['y'] / self.height
        right_norm_events['y'] = right_events['y'] / self.height
        left_norm_events['t'] = (left_events['t'] - t0) / (t1 - t0)
        right_norm_events['t'] = (right_events['t'] - t0) / (t1 - t0)
        left_norm_events['p'] = left_events['p'].astype(np.float32)
        right_norm_events['p'] = right_events['p'].astype(np.float32)
        return left_norm_events, right_norm_events, poses
    
    def extract_events(self, events, idx_start, N):
        if len(events) == 0:
            return events, 0.0, 0.0
        
        events_strip = events[idx_start: idx_start+N] if len(events) >= N else events[idx_start:]
        t_start = events_strip['t'][0]
        t_end = events_strip['t'][-1]
        events_strip_duration = t_end - t_start
        if len(events_strip) < N:
            pad_count = N - len(events_strip)
            pad_event = np.zeros(1, dtype=events.dtype)
            pad_event['x'] = -1
            pad_event['y'] = -1
            pad_event['t'] = -1
            pad_event['p'] = 0
            pad_array = np.tile(pad_event, pad_count)
            events_strip = np.concatenate((events_strip, pad_array))
        return events_strip, events_strip_duration
    
    def find_pose(self, poses, target_time):
        if len(poses) == 0:
            return np.zeros(7, dtype=np.float32)
        
        idx = bisect_left(poses[:, 0], target_time)
        if idx == 0:
            return poses[0, 1:].astype(np.float32)
        elif idx >= len(poses):
            return poses[-1, 1:].astype(np.float32)
        else:
            pose_before = poses[idx - 1]
            pose_after = poses[idx]
            if abs(pose_before[0] - target_time) < abs(pose_after[0] - target_time):
                return pose_before[1:].astype(np.float32)
            else:
                return pose_after[1:].astype(np.float32)
            
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        N = self.N
        idx_start = idx * N
        left_events, right_events, poses = self.load_and_normalize(seq_info)
        left_events_strip, left_events_strip_duration = self.extract_events(left_events, idx_start, N)
        right_events_strip, right_events_strip_duration = self.extract_events(right_events, idx_start, N)
        left_target_time = left_events_strip_duration / 2.0
        right_target_time = right_events_strip_duration / 2.0
        left_pose = self.find_pose(poses, left_target_time)
        right_pose = self.find_pose(poses, right_target_time)

        return{
            'left_events_strip': left_events_strip,
            'right_events_strip': right_events_strip,
            'left_pose': left_pose,
            'right_pose': right_pose,
            'left_events_strip_duration': left_events_strip_duration,
            'right_events_strip_duration': right_events_strip_duration,
            'sequence_info': {
                'scenario': seq_info['scenario'],
                'sequence': seq_info['sequence']
            }
        }
    
def event_extractor(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)

if __name__ == "__main__":
    dataset_root = r"/home/adarsh/Documents/SRM/dataset/train"
    dataset = event_extractor(dataset_root)
    print(f"found {len(dataset)} sequences")
    sample = dataset[0]
    print(len(sample['left_events_strip']))
    for i in range(len(sample['left_events_strip'])):
        print(sample['left_events_strip'][i])