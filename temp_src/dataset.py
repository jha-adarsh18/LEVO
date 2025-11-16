import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
from tqdm import tqdm
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
        
        self.pairs = self._build_pairs()
        print(f"Loaded {len(self.pairs)} frame pairs from {len(self.sequences)} sequences")
        
        cache_file = self.data_root / f'event_indices_cache_{camera}_{event_window_ms}ms_dt{dt_range[0]}-{dt_range[1]}.pkl'
        
        if cache_file.exists():
            print(f"Loading pre-computed indices from cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                self.event_indices_cache = pickle.load(f)
            print(f"✓ Loaded cached indices")
        else:
            print("Pre-computing event indices (this happens once)...")
            self.event_indices_cache = self._precompute_event_indices()
            
            print(f"Saving pre-computed indices to cache: {cache_file.name}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.event_indices_cache, f)
            print("✓ Cached indices saved")
    
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
        elif any(s in seq_lower for s in self.intrinsics_config['group_E']['sequences']):
            K_params = self.intrinsics_config['group_E']
        else:
            print(f"Warning: Using default intrinsics for {seq_name}")
            K_params = self.intrinsics_config.get('default', self.intrinsics_config['indoor_flying'])
        
        K = np.array([
            [K_params['fx'], 0, K_params['cx']],
            [0, K_params['fy'], K_params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _is_group_d_sequence(self, seq_name):
        seq_lower = seq_name.lower()
        return any(s in seq_lower for s in self.intrinsics_config['group_D']['sequences'])
    
    def _build_pairs(self):
        self.sequences = {}
        all_pairs = []
        
        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            seq_name = seq_dir.name
            
            if self._is_group_d_sequence(seq_name):
                print(f"Skipping Group D sequence: {seq_name}")
                continue
            
            is_mvsec = 'indoor' in seq_name.lower() or 'outdoor' in seq_name.lower()
            
            event_files = list(seq_dir.glob(f"*{self.camera}*.h5"))
            if not event_files:
                print(f"No event file found for {seq_name}, skipping")
                continue
            event_file = event_files[0]
            
            pose_files = list(seq_dir.glob("*.txt"))
            if not pose_files:
                print(f"No pose file found for {seq_name}, skipping")
                continue
            pose_file = pose_files[0]
            
            poses = np.loadtxt(pose_file)
            timestamps = poses[:, 0]
            
            if is_mvsec:
                timestamps = timestamps / 1e6
            
            K = self._get_intrinsics(seq_name)
            
            if is_mvsec:
                resolution = (346, 260)
            else:
                K_params = None
                seq_lower = seq_name.lower()
                if any(s in seq_lower for s in self.intrinsics_config['group_E']['sequences']):
                    K_params = self.intrinsics_config['group_E']
                
                resolution = tuple(K_params['resolution']) if K_params else (1280, 720)
            
            self.sequences[seq_name] = {
                'event_file': str(event_file),
                'timestamps': timestamps,
                'poses': poses[:, 1:],
                'is_mvsec': is_mvsec,
                'resolution': resolution,
                'K': K,
            }
            
            dt_min_sec = self.dt_range[0] / 1000.0
            dt_max_sec = self.dt_range[1] / 1000.0
            
            seq_pairs = []
            for i in range(len(timestamps) - 1):
                for j in range(i + 1, len(timestamps)):
                    dt = timestamps[j] - timestamps[i]
                    if dt > dt_max_sec:
                        break
                    if dt >= dt_min_sec:
                        seq_pairs.append((seq_name, i, j))
            
            if len(seq_pairs) > 1000:
                seq_seed = 42 + hash(seq_name) % 1000
                rng = np.random.RandomState(seq_seed)
                sampled_indices = rng.choice(len(seq_pairs), 1000, replace=False)
                seq_pairs = [seq_pairs[idx] for idx in sampled_indices]
            
            all_pairs.extend(seq_pairs)
            print(f"{seq_name}: {len(seq_pairs)} pairs")
        
        return all_pairs
    
    def _precompute_event_indices(self):
        window_us = self.event_window_ms * 1000
        event_indices_cache = {}
        
        for seq_name in tqdm(self.sequences.keys(), desc="Pre-computing indices"):
            seq = self.sequences[seq_name]
            event_indices_cache[seq_name] = {}
            
            with h5py.File(seq['event_file'], 'r') as f:
                events_t = np.array(f['events']['t'][:])
                
                if 't_offset' in f:
                    t_offset = f['t_offset'][()][0] / 1e6
                    events_t = events_t / 1e6 + t_offset
                else:
                    events_t = events_t / 1e6
                
                has_ms_to_idx = 'ms_to_idx' in f
                ms_to_idx = np.array(f['ms_to_idx'][:]) if has_ms_to_idx else None
            
            unique_timestamps = set()
            for pair_seq_name, i, j in self.pairs:
                if pair_seq_name == seq_name:
                    unique_timestamps.add(seq['timestamps'][i])
                    unique_timestamps.add(seq['timestamps'][j])
            
            for t in unique_timestamps:
                t_start = t - window_us / 2e6
                t_end = t + window_us / 2e6
                
                if has_ms_to_idx and ms_to_idx is not None:
                    t_start_ms = int(t_start * 1000)
                    t_end_ms = int(t_end * 1000)
                    start_idx = ms_to_idx[t_start_ms] if t_start_ms < len(ms_to_idx) else len(events_t)
                    end_idx = ms_to_idx[t_end_ms] if t_end_ms < len(ms_to_idx) else len(events_t)
                else:
                    start_idx = np.searchsorted(events_t, t_start)
                    end_idx = np.searchsorted(events_t, t_end)
                
                event_indices_cache[seq_name][t] = (start_idx, end_idx)
        
        return event_indices_cache
    
    def _load_events(self, seq_name, timestamp):
        seq = self.sequences[seq_name]
        
        if seq_name in self.event_indices_cache and timestamp in self.event_indices_cache[seq_name]:
            start_idx, end_idx = self.event_indices_cache[seq_name][timestamp]
        else:
            window_us = self.event_window_ms * 1000
            t_start = timestamp - window_us / 2e6
            t_end = timestamp + window_us / 2e6
            
            with h5py.File(seq['event_file'], 'r') as f:
                events_t = np.array(f['events']['t'][:])
                if 't_offset' in f:
                    t_offset = f['t_offset'][()][0] / 1e6
                    events_t = events_t / 1e6 + t_offset
                else:
                    events_t = events_t / 1e6
                
                start_idx = np.searchsorted(events_t, t_start)
                end_idx = np.searchsorted(events_t, t_end)
        
        with h5py.File(seq['event_file'], 'r') as f:
            events_x = np.array(f['events']['x'][start_idx:end_idx])
            events_y = np.array(f['events']['y'][start_idx:end_idx])
            events_t = np.array(f['events']['t'][start_idx:end_idx])
            events_p = np.array(f['events']['p'][start_idx:end_idx])
            
            if 't_offset' in f:
                t_offset = f['t_offset'][()][0] / 1e6
                events_t = events_t / 1e6 + t_offset
            else:
                events_t = events_t / 1e6
        
        width, height = seq['resolution']
        events_x = events_x / float(width)
        events_y = events_y / float(height)
        
        events = np.stack([events_x, events_y, events_t, events_p], axis=1).astype(np.float32)
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
        x, y, z, w = q
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