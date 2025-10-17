import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool

class EventExtractionDataset(Dataset):
    def __init__(self, dataset_root, N=1024, num_workers=16):
        self.dataset_root = dataset_root
        self.N = N
        self.num_workers = num_workers
        self.flat_samples = []
        self.sequence_map = {}
        
        self.discover_and_load_metadata()
    
    def discover_and_load_metadata(self):
        """Load all metadata in parallel using all available cores"""
        print(f"Discovering NPZ files in {self.dataset_root}...")
        npz_files = sorted(glob(os.path.join(self.dataset_root, '*.npz')))
        
        if not npz_files:
            raise ValueError(f"No NPZ files found in {self.dataset_root}")
        
        print(f"Found {len(npz_files)} NPZ files")
        print(f"Loading metadata with {self.num_workers} workers...")
        
        # Parallel metadata loading
        with Pool(self.num_workers) as pool:
            all_metadata = list(tqdm(
                pool.imap(self._load_single_metadata, npz_files, chunksize=1000),
                total=len(npz_files),
                desc="Loading metadata"
            ))
        
        # Group by sequence
        sequence_map = {}
        for meta in all_metadata:
            seq = meta['sequence']
            if seq not in sequence_map:
                sequence_map[seq] = []
            sequence_map[seq].append(meta)
        
        # Sort by time within each sequence
        for seq_name in sequence_map:
            sequence_map[seq_name].sort(key=lambda x: x['time_start'])
        
        # Flatten with sequence position info
        for seq_name, samples in sequence_map.items():
            for pos, sample in enumerate(samples):
                sample['position_in_sequence'] = pos
                sample['sequence_length'] = len(samples)
                self.flat_samples.append(sample)
        
        self.sequence_map = sequence_map
        
        print(f"\nLoaded {len(self.flat_samples)} samples from {len(sequence_map)} sequences")
        print("\nSequence statistics:")
        for seq_name, samples in self.sequence_map.items():
            duration = (samples[-1]['time_end'] - samples[0]['time_start']) / 1e6  # seconds
            print(f"  {seq_name}: {len(samples)} packets, {duration:.1f}s duration")
    
    @staticmethod
    def _load_single_metadata(npz_path):
        """Static method for parallel loading"""
        data = np.load(npz_path)
        return {
            'filepath': npz_path,
            'filename': os.path.basename(npz_path),
            'sequence': str(data['sequence']),
            'time_start': float(data['time_window'][0]),
            'time_end': float(data['time_window'][1]),
            'time_mid': float(data['time_mid'])
        }
    
    def __len__(self):
        return len(self.flat_samples)
    
    def __getitem__(self, idx):
        sample_info = self.flat_samples[idx]
        
        # Load NPZ file
        npz_path = sample_info['filepath']
        data = np.load(npz_path)
        
        events = data['events']  # Structured array: dtype=[('x','f4'), ('y','f4'), ('t','f4'), ('p','u1')]
        
        # Convert structured array to [N, 4] array
        N = len(events)
        events_array = np.zeros((N, 4), dtype=np.float32)
        events_array[:, 0] = events['x']
        events_array[:, 1] = events['y']
        events_array[:, 2] = events['t']
        events_array[:, 3] = events['p'].astype(np.float32)
        
        # Pad or truncate to self.N (1024)
        if N < self.N:
            # Pad with zeros
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
        
        return {
            'events': events_tensor,              # [1024, 4]
            'mask': mask_tensor,                  # [1024]
            'sequence': sample_info['sequence'],
            'position_in_sequence': sample_info['position_in_sequence'],
            'sequence_length': sample_info['sequence_length'],
            'time_start': sample_info['time_start'],
            'time_end': sample_info['time_end'],
            'time_mid': sample_info['time_mid']
        }


def event_extractor(dataset_root, **kwargs):
    return EventExtractionDataset(dataset_root, **kwargs)


if __name__ == "__main__":
    dataset_root = r"/workspace/npy_cache"
    
    print("="*60)
    print("Testing EventExtractionDataset")
    print("="*60)
    
    dataset = event_extractor(dataset_root, N=1024, num_workers=16)
    
    print(f"\n{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"{'='*60}\n")
    
    # Test first sample
    print("Loading first sample...")
    sample = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Events shape: {sample['events'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Valid events: {sample['mask'].sum().item()}/{len(sample['mask'])}")
    print(f"  Sequence: {sample['sequence']}")
    print(f"  Position: {sample['position_in_sequence']}/{sample['sequence_length']}")
    print(f"  Time: {sample['time_start']:.0f} -> {sample['time_end']:.0f} (mid: {sample['time_mid']:.0f})")
    
    # Test a few more samples
    print("\nTesting random samples...")
    import random
    for _ in range(3):
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        valid_events = sample['mask'].sum().item()
        print(f"  Sample {idx}: {valid_events}/1024 valid events, seq={sample['sequence']}")
    
    print("\nDataset ready for training!")