import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle

def worker_init_fn(worker_id):
    pass

class EventVODataset(Dataset):
    def __init__(self, data_root, camera='left', dt_range=(50, 200), 
             event_window_ms=50, n_events=2048, augment=False, intrinsics_config='config.yaml'):
        self.data_root = Path(data_root)
        self.camera = camera
        self.dt_range = dt_range
        self.event_window_ms = event_window_ms
        self.n_events = n_events
        self.augment = augment
        
        with open(intrinsics_config, 'r') as f:
            self.intrinsics_config = yaml.safe_load(f)['intrinsics']
        
        self.h5_files = {}
        self.pairs = self._build_pairs()
        print(f"Loaded {len(self.pairs)} frame pairs from {len(self.sequences)} sequences")
        
        print("Preloading all event data to RAM...")
        self._open_h5_files()
        
        cache_file = self.data_root / f'event_indices_cache_{camera}_{event_window_ms}ms_dt{dt_range[0]}-{dt_range[1]}.pkl'
        
        if cache_file.exists():
            print(f"Loading pre-computed indices from cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                cached_indices = pickle.load(f)
            
            for seq_name, indices in cached_indices.items():
                if seq_name in self.sequences:
                    self.sequences[seq_name]['event_indices'] = indices
            print(f"✓ Loaded cached indices for {len(cached_indices)} sequences")
        else:
            print("Pre-computing event indices for fast loading (this will take ~1 hour but only happens once)...")
            self._precompute_event_indices()
            
            print(f"Saving pre-computed indices to cache: {cache_file.name}")
            cached_indices = {seq_name: seq['event_indices'] 
                             for seq_name, seq in self.sequences.items()}
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_indices, f)
            print("✓ Cached indices saved for future runs")
    
    def _get_intrinsics(self, seq_name):
        seq_lower = seq_name.lower()
        
        if 'indoor_flying' in seq_lower:
            K_params = self.intrinsics_config['indoor_flying']
        elif 'outdoor_night' in seq_lower:
            K_params = self.intrinsics_config['outdoor_night']
        elif 'outdoor_day' in seq_lower:
            K_params = self.intrinsics_config['outdoor_day']
        elif any(s in seq_lower for s in self.intrinsics_config['group_D']['sequences']):
            K_params = self.intrinsics_config['group_D']
        else:
            K_params = self.intrinsics_config['default']
        
        K = np.array([
            [K_params['fx'], 0, K_params['cx']],
            [0, K_params['fy'], K_params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _build_pairs(self):
        self.sequences = {}
        pairs = []
        
        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            seq_name = seq_dir.name
            is_mvsec = 'indoor' in seq_name.lower() or 'outdoor' in seq_name.lower() or 'ourdoor' in seq_name.lower()
            
            event_file = list(seq_dir.glob(f"*{self.camera}*.h5"))[0]
            pose_file = list(seq_dir.glob("*.txt"))[0]
            
            poses = np.loadtxt(pose_file)
            timestamps = poses[:, 0]
            
            if is_mvsec:
                timestamps = timestamps / 1e6
            
            K = self._get_intrinsics(seq_name)
            
            self.sequences[seq_name] = {
                'event_file': str(event_file),
                'timestamps': timestamps,
                'poses': poses[:, 1:],
                'is_mvsec': is_mvsec,
                'resolution': (346, 260) if is_mvsec else (1280, 720),
                'K': K,
                'has_ms_to_idx': False,
                'event_indices': {}
            }
            
            dt_min_sec = self.dt_range[0] / 1000.0
            dt_max_sec = self.dt_range[1] / 1000.0
            
            for i in range(len(timestamps) - 1):
                for j in range(i + 1, len(timestamps)):
                    dt = timestamps[j] - timestamps[i]
                    if dt > dt_max_sec:
                        break
                    if dt >= dt_min_sec:
                        pairs.append((seq_name, i, j))
        
        return pairs
    
    def _load_sequence_worker(self, args):
        seq_name, seq_info = args
        f = h5py.File(seq_info['event_file'], 'r')
        
        ms_to_idx = None
        if seq_info['has_ms_to_idx'] and 'ms_to_idx' in f:
            ms_to_idx = np.array(f['ms_to_idx'][:])
        
        events_x = np.array(f['events']['x'][:])
        events_y = np.array(f['events']['y'][:])
        events_t = np.array(f['events']['t'][:])
        events_p = np.array(f['events']['p'][:])

        if 't_offset' in f:
            t_offset = f['t_offset'][()][0] / 1e6
            events_t = events_t / 1e6 + t_offset
        else:
            events_t = events_t / 1e6

        events_x = events_x / (1280.0 if not seq_info['is_mvsec'] else 346.0)
        events_y = events_y / (720.0 if not seq_info['is_mvsec'] else 260.0)

        f.close()
        
        n_events = len(events_x)
        return seq_name, {
            'events_x': events_x,
            'events_y': events_y,
            'events_t': events_t,
            'events_p': events_p,
            'ms_to_idx': ms_to_idx,
            'n_events': n_events
        }
    
    def _open_h5_files(self):
        n_workers = min(32, cpu_count())
        print(f"Using {n_workers} workers for parallel loading...")
        
        with Pool(processes=n_workers) as pool:
            results = []
            for seq_name in self.sequences.keys():
                results.append((seq_name, self.sequences[seq_name]))
            
            pbar = tqdm(total=len(results), desc="Loading sequences to RAM", ncols=120)
            for seq_name, loaded_data in pool.imap_unordered(self._load_sequence_worker, results):
                self.sequences[seq_name]['events_x'] = loaded_data['events_x']
                self.sequences[seq_name]['events_y'] = loaded_data['events_y']
                self.sequences[seq_name]['events_t'] = loaded_data['events_t']
                self.sequences[seq_name]['events_p'] = loaded_data['events_p']
                if loaded_data['ms_to_idx'] is not None:
                    self.sequences[seq_name]['ms_to_idx'] = loaded_data['ms_to_idx']
                
                n_events = loaded_data['n_events']
                size_mb = (n_events * 4 * 4) / (1024 ** 2)
                
                pbar.set_postfix_str(f"{seq_name[:20]}: {n_events/1e6:.1f}M events, {size_mb:.1f}MB")
                pbar.update(1)
            
            pbar.close()
        
        total_ram = sum(len(seq.get('events_x', [])) * 4 * 4 for seq in self.sequences.values()) / (1024 ** 3)
        print(f"✓ Loaded {total_ram:.2f} GB to RAM")

    def _precompute_single_sequence(self, args):
        seq_name, timestamps_to_compute, seq_info, window_us = args
        
        event_indices = {}
        
        f = h5py.File(seq_info['event_file'], 'r')
        events_t = np.array(f['events']['t'][:])
        
        if 't_offset' in f:
            t_offset = f['t_offset'][()][0] / 1e6
            events_t = events_t / 1e6 + t_offset
        else:
            events_t = events_t / 1e6
        
        ms_to_idx = None
        if seq_info['has_ms_to_idx'] and 'ms_to_idx' in f:
            ms_to_idx = np.array(f['ms_to_idx'][:])
        
        f.close()
        
        for t in timestamps_to_compute:
            t_start = t - window_us / 2e6
            t_end = t + window_us / 2e6
            
            if seq_info['has_ms_to_idx'] and ms_to_idx is not None:
                t_start_ms = int(t_start * 1000)
                t_end_ms = int(t_end * 1000)
                start_idx = ms_to_idx[t_start_ms] if t_start_ms < len(ms_to_idx) else len(events_t)
                end_idx = ms_to_idx[t_end_ms] if t_end_ms < len(ms_to_idx) else len(events_t)
            else:
                start_idx = np.searchsorted(events_t, t_start)
                end_idx = np.searchsorted(events_t, t_end)
            
            event_indices[t] = (start_idx, end_idx)
        
        return seq_name, event_indices

    def _precompute_event_indices(self):
        window_us = self.event_window_ms * 1000
        
        seq_timestamps = {}
        for seq_name in self.sequences.keys():
            seq = self.sequences[seq_name]
            timestamps = seq['timestamps']
            
            unique_timestamps = set()
            for seq_n, i, j in self.pairs:
                if seq_n == seq_name:
                    unique_timestamps.add(timestamps[i])
                    unique_timestamps.add(timestamps[j])
            
            seq_timestamps[seq_name] = list(unique_timestamps)
        
        total_timestamps = sum(len(ts) for ts in seq_timestamps.values())
        print(f"Computing indices for {total_timestamps} unique timestamps across {len(self.sequences)} sequences...")
        
        tasks = []
        for seq_name, timestamps in seq_timestamps.items():
            tasks.append((
                seq_name,
                timestamps,
                self.sequences[seq_name],
                window_us
            ))
        
        n_workers = min(len(tasks), cpu_count())
        print(f"Using {n_workers} workers for parallel index computation...")
        
        with Pool(processes=n_workers) as pool:
            pbar = tqdm(total=len(tasks), desc="Pre-computing event indices", ncols=120)
            
            for seq_name, event_indices in pool.imap_unordered(self._precompute_single_sequence, tasks):
                self.sequences[seq_name]['event_indices'] = event_indices
                n_computed = len(event_indices)
                pbar.set_postfix_str(f"{seq_name[:25]}: {n_computed} timestamps")
                pbar.update(1)
            
            pbar.close()
        
        print(f"✓ Pre-computed {total_timestamps} event index ranges")

    def _load_events(self, seq_name, timestamp):
        seq = self.sequences[seq_name]
        
        if timestamp in seq['event_indices']:
            start_idx, end_idx = seq['event_indices'][timestamp]
        else:
            print(f"Warning: timestamp {timestamp} not pre-computed for {seq_name}")
            window_us = self.event_window_ms * 1000
            t_start = timestamp - window_us / 2e6
            t_end = timestamp + window_us / 2e6
            
            if seq['has_ms_to_idx']:
                t_start_ms = int(t_start * 1000)
                t_end_ms = int(t_end * 1000)
                ms_to_idx = seq['ms_to_idx']
                start_idx = ms_to_idx[t_start_ms] if t_start_ms < len(ms_to_idx) else len(seq['events_t'])
                end_idx = ms_to_idx[t_end_ms] if t_end_ms < len(ms_to_idx) else len(seq['events_t'])
            else:
                start_idx = np.searchsorted(seq['events_t'], t_start)
                end_idx = np.searchsorted(seq['events_t'], t_end)
        
        x = seq['events_x'][start_idx:end_idx]
        y = seq['events_y'][start_idx:end_idx]
        t = seq['events_t'][start_idx:end_idx]
        p = seq['events_p'][start_idx:end_idx]
        
        events = np.stack([x, y, t, p], axis=1).astype(np.float32)
        return events
    
    def _process_events(self, events, resolution):
        n = len(events)
        
        if n == 0:
            return np.zeros((self.n_events, 4), dtype=np.float32), np.zeros(self.n_events, dtype=np.float32)
        
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        if t_max > t_min:
            events[:, 2] = (events[:, 2] - t_min) / (t_max - t_min)
        else:
            events[:, 2] = 0.5
        
        if self.augment:
            events[:, :2] += np.random.uniform(-0.03, 0.03, (n, 2))
            events[:, :2] = np.clip(events[:, :2], 0, 1)
            events[:, 2] += np.random.uniform(-0.1, 0.1, n)
            events[:, 2] = np.clip(events[:, 2], 0, 1)
            if np.random.rand() < 0.1:
                events[:, 3] = 1 - events[:, 3]
        
        if n > self.n_events:
            indices = np.random.choice(n, self.n_events, replace=False)
            indices = np.sort(indices)
            sampled = events[indices]
            mask = np.ones(self.n_events, dtype=np.float32)
        else:
            sampled = np.zeros((self.n_events, 4), dtype=np.float32)
            sampled[:n] = events
            mask = np.zeros(self.n_events, dtype=np.float32)
            mask[:n] = 1.0
        
        return sampled, mask
    
    def _compute_relative_pose(self, pose1, pose2):
        p1, q1 = pose1[:3], pose1[3:7]
        p2, q2 = pose2[:3], pose2[3:7]
        
        R1 = self._quat_to_rot(q1)
        R2 = self._quat_to_rot(q2)
        
        R_rel = R2 @ R1.T
        t_rel = R1.T @ (p2 - p1)
        
        return R_rel, t_rel
    
    def _quat_to_rot(self, q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        return R
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        seq_name, i, j = self.pairs[idx]
        seq = self.sequences[seq_name]
        
        t1 = seq['timestamps'][i]
        t2 = seq['timestamps'][j]
        
        events1 = self._load_events(seq_name, t1)
        events2 = self._load_events(seq_name, t2)
        
        events1, mask1 = self._process_events(events1, seq['resolution'])
        events2, mask2 = self._process_events(events2, seq['resolution'])
        
        R_rel, t_rel = self._compute_relative_pose(seq['poses'][i], seq['poses'][j])
        
        return {
            'events1': torch.from_numpy(events1),
            'mask1': torch.from_numpy(mask1),
            'events2': torch.from_numpy(events2),
            'mask2': torch.from_numpy(mask2),
            'R_gt': torch.from_numpy(R_rel).float(),
            't_gt': torch.from_numpy(t_rel).float(),
            'K': torch.from_numpy(seq['K']).float(),
            'resolution': torch.tensor(seq['resolution'], dtype=torch.float32),
        }

def collate_fn(batch):
    return {
        'events1': torch.stack([item['events1'] for item in batch]),
        'mask1': torch.stack([item['mask1'] for item in batch]),
        'events2': torch.stack([item['events2'] for item in batch]),
        'mask2': torch.stack([item['mask2'] for item in batch]),
        'R_gt': torch.stack([item['R_gt'] for item in batch]),
        't_gt': torch.stack([item['t_gt'] for item in batch]),
        'K': torch.stack([item['K'] for item in batch]),
        'resolution': torch.stack([item['resolution'] for item in batch]),
    }

def create_dataloader(data_root, batch_size=8, num_workers=4, **kwargs):
    dataset = EventVODataset(data_root, **kwargs)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )