import numpy as np
import os
from bisect import bisect_left
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset

class EventExtractionDataset(Dataset):
    def __init__(self, dataset_root, width = None, height = None, t = 1000, N = 1024):
        self.width = width
        self.height = height
        self.t = t
        self.N = N
        self.sequences = []
        self.time_samples = []
        self.discover_sequences(dataset_root)
        self.cache_metadata()
        self.generate_time_samples()
    
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
            if "indoor" in seq_info['sequence'].lower() or "outdoor" in seq_info['sequence'].lower():
                seq_info['height'] = 260
                seq_info['width'] = 346
                seq_info['use_ms_idx'] = False
            else:
                seq_info['height'] = 720
                seq_info['width'] = 1280
                seq_info['use_ms_idx'] = True
            
            with h5py.File(seq_info['left_events_path'], "r") as e_left, \
             h5py.File(seq_info['right_events_path'], "r") as e_right:
                seq_info['time_start'] = min(float(e_left["events/t"][0]), float(e_right["events/t"][0]))
                seq_info['time_end'] = max(float(e_left["events/t"][-1]), float(e_right["events/t"][-1]))

            poses = np.loadtxt(seq_info['pose_path'])
            seq_info['poses'] = poses

    def generate_time_samples(self):
        for seq_idx, seq_info in enumerate(self.sequences):
            t_start = seq_info['time_start']
            t_end = seq_info['time_end']
            current_time = t_start
            while current_time < t_end:
                window_end = min(current_time + self.t, t_end)
                self.time_samples.append((seq_idx, current_time, window_end))
                current_time += self.t

    def __len__(self):
        return len(self.time_samples)
    
    def get_events(self, file_path, t_start, t_end, use_ms_idx=True):
        with h5py.File(file_path, "r") as f:
            if use_ms_idx and "ms_to_idx" in f:
                ms_to_idx = f["ms_to_idx"]
                ms_start =  int(t_start / 1000)
                ms_end = int(t_end / 1000)
                start_idx = ms_to_idx[ms_start] if ms_start < len(ms_to_idx) else len(f["events/t"]) -1
                end_idx = ms_to_idx[ms_end] if ms_end < len(ms_to_idx) else len(f["events/t"])
            
            else:
                # use binary search, mainly for MVSEC while training
                start_idx = np.searchsorted(f["events/t"], t_start, side = "left")
                end_idx = np.searchsorted(f["events/t"], t_end, side='right')

            if start_idx > end_idx:
                return np.array([], dtype = [('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
            
            x = f["events/x"][start_idx:end_idx]
            y = f["events/y"][start_idx:end_idx]
            t = f["events/t"][start_idx:end_idx]
            p = f["events/p"][start_idx:end_idx]

            events = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
            
            events['x'] = x
            events['y'] = y
            events['t'] = t
            events['p'] = p.astype(np.uint8)

            return events

    def load_and_normalize(self, seq_info, t_start, t_end):
        with h5py.File(seq_info['left_events_path'], "r") as e_left, \
             h5py.File(seq_info['right_events_path'], "r") as e_right:
            
            t_start_norm = (t_start - seq_info['t0']) / seq_info['time_range']
            t_end_norm = (t_end - seq_info['t0']) / seq_info['time_range']
            
            if seq_info['use_ms_idx']:
                ms_to_idx_left = e_left["ms_to_idx"][:]
                ms_to_idx_right = e_right["ms_to_idx"][:]
                
                ms_start = int(t_start / 1000)
                ms_end = int(t_end / 1000)
                
                left_start_idx = ms_to_idx_left[ms_start] if ms_start < len(ms_to_idx_left) else len(e_left["events/t"])-1
                left_end_idx = ms_to_idx_left[ms_end] if ms_end < len(ms_to_idx_left) else len(e_left["events/t"])
                right_start_idx = ms_to_idx_right[ms_start] if ms_start < len(ms_to_idx_right) else len(e_right["events/t"])-1
                right_end_idx = ms_to_idx_right[ms_end] if ms_end < len(ms_to_idx_right) else len(e_right["events/t"])
            else:
                left_start_idx = np.searchsorted(e_left["events/t"], t_start_norm, side='left')
                left_end_idx = np.searchsorted(e_left["events/t"], t_end_norm, side='right')
                right_start_idx = np.searchsorted(e_right["events/t"], t_start_norm, side='left')
                right_end_idx = np.searchsorted(e_right["events/t"], t_end_norm, side='right')
            
            if left_start_idx >= left_end_idx:
                left_norm_events = np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
            else:
                left_x = e_left["events/x"][left_start_idx:left_end_idx]
                left_y = e_left["events/y"][left_start_idx:left_end_idx]
                left_t = e_left["events/t"][left_start_idx:left_end_idx]
                left_p = e_left["events/p"][left_start_idx:left_end_idx]
                
                left_norm_events = np.zeros(len(left_x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
                left_norm_events['x'] = left_x / seq_info['width']
                left_norm_events['y'] = left_y / seq_info['height']
                left_norm_events['t'] = (left_t - seq_info['t0']) / seq_info['time_range']
                left_norm_events['p'] = left_p.astype(np.uint8)
            
            if right_start_idx >= right_end_idx:
                right_norm_events = np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
            else:
                right_x = e_right["events/x"][right_start_idx:right_end_idx]
                right_y = e_right["events/y"][right_start_idx:right_end_idx]
                right_t = e_right["events/t"][right_start_idx:right_end_idx]
                right_p = e_right["events/p"][right_start_idx:right_end_idx]
                
                right_norm_events = np.zeros(len(right_x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
                right_norm_events['x'] = right_x / seq_info['width']
                right_norm_events['y'] = right_y / seq_info['height']
                right_norm_events['t'] = (right_t - seq_info['t0']) / seq_info['time_range']
                right_norm_events['p'] = right_p.astype(np.uint8)
        
        return left_norm_events, right_norm_events, seq_info['poses']
    
    def extract_events(self, seq_info, events, t_start, ms_to_idx, t=5000):
        if len(events) == 0:
            return events, 0.0
        
        if "indoor" in seq_info['sequence'].lower() or "outdoor" in seq_info['sequence'].lower():
            t1 = t_start + t
            start_idx = np.searchsorted(events['t'], t_start, side='left')
            end_idx   = np.searchsorted(events['t'], t1, side='right')
            events_slice = events[start_idx:end_idx]
            events_duration = t1 - t_start
        else:
            ms_start = int(t_start / 1000)
            ms_end   = int((t_start + t) / 1000)
            start_idx = ms_to_idx[ms_start] if ms_start < len(ms_to_idx) else len(events)-1
            end_idx   = ms_to_idx[ms_end]   if ms_end < len(ms_to_idx) else len(events)
            events_slice = events[start_idx:end_idx]
            events_duration = (events['t'][end_idx-1] - events['t'][start_idx]) if len(events_slice) > 0 else 0.0
            
        return events_slice, events_duration
    
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
        t = self.t
        t_start = idx * t
        t_end = t_start + t
        
        left_events_strip, right_events_strip, poses = self.load_and_normalize(seq_info, t_start, t_end)
        
        left_events_strip_duration = t if len(left_events_strip) > 0 else 0.0
        right_events_strip_duration = t if len(right_events_strip) > 0 else 0.0
        
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
                'sequence': seq_info['sequence']
            }
        }
    
def event_extractor(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)

if __name__ == "__main__":
    dataset_root = r"//media/adarsh/One Touch/EventSLAM/dataset/train"
    dataset = event_extractor(dataset_root)
    print(f"found {len(dataset)} sequences")
    sample = dataset[0]
    print(len(sample['left_events_strip']))
    for i in range(len(sample['left_events_strip'])):
        print(sample['left_events_strip'][i])