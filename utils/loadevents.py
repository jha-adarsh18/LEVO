import numpy as np
import os
from bisect import bisect_left
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class EventExtractionDataset(Dataset):
    def __init__(self, dataset_root, width = None, height = None, t = 5000, N = 1024):
        self.width = width
        self.height = height
        self.t = t
        self.N = N
        self.flat_samples = []
        self.dataset_root = dataset_root
        self.sequence_map = {}
        self.discover_sequences(dataset_root)
        self.generate_flat_samples()
    
    def discover_sequences(self, dataset_root):
        # Check for NPZ files
        npz_files = [f for f in os.listdir(dataset_root) if f.endswith('.npz') and f.startswith('sample_')]
        
        if npz_files:
            self.is_npz = True
            self.total_samples = len(npz_files)
            print(f"Detected NPZ format with {self.total_samples} samples")
        else:
            raise ValueError("Only NPZ format is supported. Please convert H5 files first.")

    def generate_flat_samples(self):
        """Generate samples WITH sequence grouping for proper triplet sampling"""
        print("Generating flat samples with sequence metadata...")
        
        # Group samples by sequence
        sequence_map = {}  # {sequence_name: [list of sample info dicts]}
        
        for idx in tqdm(range(self.total_samples), desc="Loading metadata"):
            npz_path = os.path.join(self.dataset_root, f'sample_{idx:08d}.npz')
            data = np.load(npz_path)
            
            sequence_name = str(data['sequence'])
            time_window = tuple(data['time_window'])
            
            if sequence_name not in sequence_map:
                sequence_map[sequence_name] = []
            
            sequence_map[sequence_name].append({
                'sample_idx': idx,
                'sequence': sequence_name,
                'time_start': time_window[0],
                'time_end': time_window[1],
                'time_mid': (time_window[0] + time_window[1]) / 2.0
            })
        
        # Sort by time within each sequence
        for seq_name in sequence_map:
            sequence_map[seq_name].sort(key=lambda x: x['time_start'])
        
        # Flatten but keep sequence info
        for seq_name, samples in sequence_map.items():
            for pos, sample_info in enumerate(samples):
                sample_info['position_in_sequence'] = pos
                sample_info['sequence_length'] = len(samples)
                self.flat_samples.append(sample_info)
        
        self.sequence_map = sequence_map
        print(f"Generated {len(self.flat_samples)} samples from {len(sequence_map)} sequences")
        
        # Print sequence statistics
        for seq_name, samples in self.sequence_map.items():
            duration = (samples[-1]['time_end'] - samples[0]['time_start']) / 1e6  # Convert to seconds
            print(f"  {seq_name}: {len(samples)} packets, {duration:.1f}s duration")

    def __len__(self):
        return len(self.flat_samples)
    
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

    def __getitem__(self, idx):
        sample_info = self.flat_samples[idx]
        sample_idx = sample_info['sample_idx']
        
        # Load from NPZ file
        npz_path = os.path.join(self.dataset_root, f'sample_{sample_idx:08d}.npz')
        data = np.load(npz_path)
        
        events = data['events']  # Structured array with dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')]
        
        # Convert to [N, 4] numpy array
        N = len(events)
        events_array = np.zeros((N, 4), dtype=np.float32)
        events_array[:, 0] = events['x']
        events_array[:, 1] = events['y']
        events_array[:, 2] = events['t']
        events_array[:, 3] = events['p'].astype(np.float32)
        
        # Pad or truncate to self.N
        if N < self.N:
            # Pad with zeros (will be masked)
            padded = np.zeros((self.N, 4), dtype=np.float32)
            padded[:N] = events_array
            mask = np.zeros(self.N, dtype=bool)
            mask[:N] = True
            events_array = padded
        else:
            # Truncate
            events_array = events_array[:self.N]
            mask = np.ones(self.N, dtype=bool)
        
        # Convert to tensors
        events_tensor = torch.from_numpy(events_array)
        mask_tensor = torch.from_numpy(mask)
        
        # Return single sample with metadata for triplet sampling
        return {
            'events': events_tensor,  # [N, 4]
            'mask': mask_tensor,  # [N]
            'sequence': sample_info['sequence'],
            'position_in_sequence': sample_info['position_in_sequence'],
            'sequence_length': sample_info['sequence_length'],
            'time_start': sample_info['time_start'],
            'time_end': sample_info['time_end'],
            'time_mid': sample_info['time_mid'],
            'sample_idx': sample_idx
        }

def event_extractor(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)

if __name__ == "__main__":
    dataset_root = r"/workspace/PEVSLAM/npy_cache"
    dataset = event_extractor(dataset_root)
    print(f"found {len(dataset)} samples")
    
    # Get the first sample
    sample = dataset[0]
    print(f"\nSample events shape: {sample['events'].shape}")
    print(f"Sample mask shape: {sample['mask'].shape}")
    print(f"Valid events: {sample['mask'].sum().item()}")
    print(f"Sequence: {sample['sequence']}")
    print(f"Position in sequence: {sample['position_in_sequence']}/{sample['sequence_length']}")
    print(f"Time window: {sample['time_start']:.0f} - {sample['time_end']:.0f}")