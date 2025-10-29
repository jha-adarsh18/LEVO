import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if hasattr(dataset, '_open_h5_files'):
            dataset._open_h5_files()

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
            is_mvsec = 'indoor' in seq_name.lower() or 'outdoor' in seq_name.lower()
            
            event_file = list(seq_dir.glob(f"*{self.camera}*.h5"))[0]
            pose_file = list(seq_dir.glob("*.txt"))[0]
            
            poses = np.loadtxt(pose_file)
            timestamps = poses[:, 0]
            
            if not is_mvsec:
                timestamps = timestamps / 1e6
            
            K = self._get_intrinsics(seq_name)
            
            self.sequences[seq_name] = {
                'event_file': str(event_file),
                'timestamps': timestamps,
                'poses': poses[:, 1:],
                'is_mvsec': is_mvsec,
                'resolution': (346, 260) if is_mvsec else (1280, 720),
                'K': K,
                'has_ms_to_idx': not is_mvsec
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
    
    def _open_h5_files(self):
        for seq_name, seq in self.sequences.items():
            f = h5py.File(seq['event_file'], 'r')
            self.h5_files[seq_name] = f
            
            if seq['has_ms_to_idx'] and 'ms_to_idx' in f:
                self.sequences[seq_name]['ms_to_idx'] = f['ms_to_idx'][:]

    def _load_events(self, seq_name, timestamp):
        seq = self.sequences[seq_name]
        f = self.h5_files[seq_name]
    
        window_us = self.event_window_ms * 1000
        t_start = timestamp - window_us / 2e6
        t_end = timestamp + window_us / 2e6
    
        if seq['has_ms_to_idx']:
            t_start_ms = int(t_start * 1000)
            t_end_ms = int(t_end * 1000)
            
            ms_to_idx = seq['ms_to_idx']
            start_idx = ms_to_idx[t_start_ms] if t_start_ms < len(ms_to_idx) else len(f['t'])
            end_idx = ms_to_idx[t_end_ms] if t_end_ms < len(ms_to_idx) else len(f['t'])
            
            x = f['x'][start_idx:end_idx]
            y = f['y'][start_idx:end_idx]
            t = f['t'][start_idx:end_idx]
            p = f['p'][start_idx:end_idx]
        else:
            if not seq['is_mvsec']:
                t_start *= 1e6
                t_end *= 1e6
            
            t_data = f['events/t'] if seq['is_mvsec'] else f['t']
            start_idx = np.searchsorted(t_data, t_start)
            end_idx = np.searchsorted(t_data, t_end)
            
            if seq['is_mvsec']:
                x = f['events/x'][start_idx:end_idx]
                y = f['events/y'][start_idx:end_idx]
                t = f['events/t'][start_idx:end_idx]
                p = f['events/p'][start_idx:end_idx]
            else:
                x = f['x'][start_idx:end_idx]
                y = f['y'][start_idx:end_idx]
                t = f['t'][start_idx:end_idx]
                p = f['p'][start_idx:end_idx]
    
        events = np.stack([x, y, t, p], axis=1).astype(np.float32)
        return events
    
    def _process_events(self, events, resolution):
        n = len(events)
        
        if n == 0:
            return np.zeros((self.n_events, 4), dtype=np.float32), np.zeros(self.n_events, dtype=np.float32)
        
        events[:, 0] /= resolution[0]
        events[:, 1] /= resolution[1]
        
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