import numpy as np
import os
from bisect import bisect_left
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset

class EventExtractionDataset(Dataset):
    def __init__(self, dataset_root, width = None, height = None, t = 5000, N = 1024):
        self.width = width
        self.height = height
        self.t = t
        self.N = N
        self.sequences = []
        self.flat_samples = []
        self.discover_sequences(dataset_root)
        self.cache_metadata()
        self.generate_flat_samples()
    
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

    def generate_flat_samples(self):
        print("Generating flat samples...")
        for seq_idx, seq_info in enumerate(self.sequences):
            t_start = seq_info['time_start']
            t_end = seq_info['time_end']
            current_time = t_start
        
            while current_time < t_end:
                window_end = min(current_time + self.t, t_end)
                left_events = self.get_events(seq_info['left_events_path'], current_time, window_end, seq_info['use_ms_idx'])
                left_packets = self.chunk_events(left_events, self.t, self.N)
            
                for packet in left_packets:
                    if len(packet) == 0:  # Skip empty
                        continue
                
                    packet = self.normalize_events(packet, seq_info['width'], seq_info['height'])
                    mid_time = (current_time + window_end) / 2.0
                    pose = self.find_pose(seq_info['poses'], mid_time)
                
                    self.flat_samples.append({
                        'left_events_strip': packet,
                        'left_pose': pose,
                        'seq_idx': seq_idx,
                        'sequence': seq_info['sequence']
                    })
            
                current_time += self.t
    
        print(f"Generated {len(self.flat_samples)} valid packets")

    def __len__(self):
        return len(self.flat_samples)
    
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

    def normalize_events(self, events, width, height):
        # Normalize events in the range [0, 1] locally for the batch
        if  len(events) == 0:
            return events
        t_min = events['t'].min()
        t_max = events['t'].max()

        if t_min == t_max:
            events['t'] = np.zeros_like(events['t'])
        else:
            events['t'] = (events['t'] - t_min) / (t_max - t_min)
        
        events['x'] = events['x'] / width
        events['y'] = events['y'] / height

        return events
    
    def find_pose(self, poses, target_time):
        # Find pose using binary search

        if len(poses) == 0:
            return np.zeros(7, dtype = np.float32)
        
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
            
    def chunk_events(self, events, R = 5000, Np = 1024):
        """
        chunk events as per Ren et al. CVPR-2024
        for j in len(E) do
            Pi.append(ej→l); j = l; where tl - tj = R
            if (len(Pi) > Np): i = i + 1;
        """

        if len(events) == 0:
            return []
        
        packets = []
        j = 0

        while j < len(events):
            t_start = events['t'][j]
            l = j
            while l < len(events) and (events['t'][l] - t_start) <= R:
                l += 1

            packet = events[j:l].copy()

            if len(packet) > Np:
                truncated_packet = packet[:Np].copy()
                packets.append(truncated_packet)
                j = j + Np # Move to the next unprocessed event without skipping any events

            else:
                if len(packet) > 0:
                    packets.append(packet)
                    j = l # move to the next time window in the same batch of events

        return packets

    def __getitem__(self, idx):
        # Loads only one batch for the specified time

        seq_idx, t_start, t_end = self.flat_samples[idx]
        seq_info = self.sequences[seq_idx]
        left_events = self.get_events(seq_info['left_events_path'], t_start, t_end, seq_info['use_ms_idx'])
        right_events = self.get_events(seq_info['right_events_path'], t_start, t_end, seq_info['use_ms_idx'])

        R = self.t # 1 ms time interval
        Np = self.N

        left_packets = self.chunk_events(left_events, R, Np)
        right_packets = self.chunk_events(right_events, R, Np)

        all_samples = []

        # num_packets = len(left_packets)
        num_packets = min(len(left_packets), len(right_packets))

        for i in range(num_packets):
            left_chunk = left_packets[i]
            right_chunk = right_packets[i]

            left_chunk = self.normalize_events(left_chunk, seq_info['width'], seq_info['height'])
            right_chunk = self.normalize_events(right_chunk, seq_info['width'], seq_info['height'])

            # Mid-time of this packet for pose lookup
            mid_time = (t_start + t_end) / 2.0

            left_pose = self.find_pose(seq_info['poses'], mid_time)
            right_pose = self.find_pose(seq_info['poses'], mid_time)

            all_samples.append({
                'left_events_strip': left_chunk,
                'right_events_strip': right_chunk,
                'left_pose': left_pose,
                'right_pose': right_pose,
                'left_events_strip_duration': 1.0 if len(left_chunk) > 0 else 0.0,
                'right_events_strip_duration': 1.0 if len(right_chunk) > 0 else 0.0,
                'left_packets_count': len(left_packets),
                'right_packets_count': len(right_packets),
                'sequence_info': {
                    'sequence': seq_info['sequence'],
                    'time_window': (t_start, t_end),
                    'packet_index': i,
                    'total_packets': num_packets
                }
            })

        return all_samples
def event_extractor(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)

if __name__ == "__main__":
    dataset_root = r"//media/adarsh/One Touch/EventSLAM/dataset/train"
    dataset = event_extractor(dataset_root)
    print(f"found {len(dataset)} time samples")
    
    # Get the first time sample (which returns a list of packets)
    packets = dataset[0]
    print(f"Number of packets in first time sample: {len(packets)}")
    
    # Print details about each packet
    for i, packet in enumerate(packets):
        print(f"Packet {i}: {len(packet['left_events_strip'])} events")

    if len(packets) > 0:
        first_packet = packets[0]
        print(f"\nFirst packet events shape: {first_packet['left_events_strip'].shape}")
        print(f"First packet pose: {first_packet['left_pose']}")
        print(f"Sequence info: {first_packet['sequence_info']}")