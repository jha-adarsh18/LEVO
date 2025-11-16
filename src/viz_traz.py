import torch
import torch.nn.functional as F
import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml

from model import EventVO

def quaternion_from_matrix(R):
    R = R.cpu().numpy()
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])

def load_events_window(event_file, timestamp, window_ms=50, n_events=2048, resolution=(1280, 720)):
    window_us = window_ms * 1000
    t_start = timestamp - window_us / 2e6
    t_end = timestamp + window_us / 2e6
    
    with h5py.File(event_file, 'r') as f:
        has_ms_to_idx = 'ms_to_idx' in f
        
        if has_ms_to_idx:
            ms_to_idx = f['ms_to_idx']
            t_start_ms = max(0, int(t_start * 1000))
            t_end_ms = int(t_end * 1000)
            
            start_idx = ms_to_idx[t_start_ms] if t_start_ms < len(ms_to_idx) else 0
            end_idx = ms_to_idx[t_end_ms] if t_end_ms < len(ms_to_idx) else len(f['events']['t'])
        else:
            events_t = np.array(f['events']['t'][:])
            if 't_offset' in f:
                t_offset = f['t_offset'][()][0] / 1e6
                events_t = events_t / 1e6 + t_offset
            else:
                events_t = events_t / 1e6
            
            start_idx = np.searchsorted(events_t, t_start)
            end_idx = np.searchsorted(events_t, t_end)
        
        events_x = np.array(f['events']['x'][start_idx:end_idx])
        events_y = np.array(f['events']['y'][start_idx:end_idx])
        events_t = np.array(f['events']['t'][start_idx:end_idx])
        events_p = np.array(f['events']['p'][start_idx:end_idx])
        
        if 't_offset' in f:
            t_offset = f['t_offset'][()][0] / 1e6
            events_t = events_t / 1e6 + t_offset
        else:
            events_t = events_t / 1e6
    
    width, height = resolution
    events_x = events_x / float(width)
    events_y = events_y / float(height)
    
    events = np.stack([events_x, events_y, events_t, events_p], axis=1).astype(np.float32)
    
    n = len(events)
    
    if n == 0:
        return np.zeros((n_events, 4), dtype=np.float32), np.zeros(n_events, dtype=np.float32)
    
    t_min, t_max = events[:, 2].min(), events[:, 2].max()
    if t_max > t_min:
        events[:, 2] = (events[:, 2] - t_min) / (t_max - t_min)
    else:
        events[:, 2] = 0.5
    
    if n > n_events:
        indices = np.random.choice(n, n_events, replace=False)
        indices = np.sort(indices)
        sampled = events[indices]
        mask = np.ones(n_events, dtype=np.float32)
    else:
        sampled = np.zeros((n_events, 4), dtype=np.float32)
        sampled[:n] = events
        mask = np.zeros(n_events, dtype=np.float32)
        mask[:n] = 1.0
    
    return sampled, mask

def get_timestamps_from_hdf5(event_file, dt_ms=100, max_frames=None):
    with h5py.File(event_file, 'r') as f:
        events_t = np.array(f['events']['t'][:])
        if 't_offset' in f:
            t_offset = f['t_offset'][()][0] / 1e6
            events_t = events_t / 1e6 + t_offset
        else:
            events_t = events_t / 1e6
    
    t_start = events_t[0]
    t_end = events_t[-1]
    
    dt_sec = dt_ms / 1000.0
    timestamps = np.arange(t_start, t_end, dt_sec)
    
    if max_frames is not None:
        timestamps = timestamps[:max_frames]
    
    return timestamps

def load_intrinsics_from_config(config_path, sequence_name):
    """Load camera intrinsics from config file for a specific sequence."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    intrinsics_data = config.get('intrinsics', {})
    
    # Check if sequence name directly exists in config
    if sequence_name in intrinsics_data:
        intr = intrinsics_data[sequence_name]
        K = torch.eye(3)
        K[0, 0] = intr['fx']
        K[1, 1] = intr['fy']
        K[0, 2] = intr['cx']
        K[1, 2] = intr['cy']
        resolution = intr['resolution']
        return K, resolution
    
    # Check if sequence is in a group
    for group_name, group_data in intrinsics_data.items():
        if 'sequences' in group_data:
            # Normalize sequence names for comparison
            normalized_sequences = [s.lower().replace('_', '-') for s in group_data['sequences']]
            normalized_query = sequence_name.lower().replace('_', '-')
            
            if normalized_query in normalized_sequences:
                K = torch.eye(3)
                K[0, 0] = group_data['fx']
                K[1, 1] = group_data['fy']
                K[0, 2] = group_data['cx']
                K[1, 2] = group_data['cy']
                resolution = group_data['resolution']
                return K, resolution
    
    raise ValueError(f"Intrinsics for sequence '{sequence_name}' not found in config")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-file', type=str, default='/home/adarsh/Downloads/corner_slow1.synced.left_event.hdf5')
    parser.add_argument('--checkpoint', type=str, default='/home/adarsh/PEVSLAM/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='./levo_results')
    parser.add_argument('--dt-ms', type=int, default=100)
    parser.add_argument('--event-window-ms', type=int, default=50)
    parser.add_argument('--n-events', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--seq-name', type=str, default='corner_slow1', help='Sequence name to load intrinsics from config')
    parser.add_argument('--config', type=str, default='/workspace/PEVSLAM/configs/config.yaml', help='Path to intrinsics config file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading intrinsics from config...")
    K_cpu, resolution = load_intrinsics_from_config(args.config, args.seq_name)
    K = K_cpu.to(device).unsqueeze(0)
    print(f"✓ Loaded intrinsics for '{args.seq_name}':")
    print(f"  fx={K_cpu[0,0]:.2f}, fy={K_cpu[1,1]:.2f}, cx={K_cpu[0,2]:.2f}, cy={K_cpu[1,2]:.2f}")
    print(f"  Resolution: {resolution}")
    
    print("Loading model...")
    model = EventVO(d_model=args.d_model, num_samples=args.num_samples).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    print("Getting timestamps...")
    timestamps = get_timestamps_from_hdf5(args.event_file, args.dt_ms, args.max_frames)
    print(f"✓ Found {len(timestamps)} frames")
    
    R_abs = torch.eye(3, device=device)
    t_abs = torch.zeros(3, device=device)
    
    trajectory = []
    trajectory.append([timestamps[0], t_abs[0].item(), t_abs[1].item(), t_abs[2].item(), 0, 0, 0, 1])
    
    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(len(timestamps) - 1)):
            t1 = timestamps[i]
            t2 = timestamps[i + 1]
            
            events1, mask1 = load_events_window(args.event_file, t1, args.event_window_ms, args.n_events, tuple(resolution))
            events2, mask2 = load_events_window(args.event_file, t2, args.event_window_ms, args.n_events, tuple(resolution))
            
            events1 = torch.from_numpy(events1).unsqueeze(0).to(device)
            mask1 = torch.from_numpy(mask1).unsqueeze(0).to(device)
            events2 = torch.from_numpy(events2).unsqueeze(0).to(device)
            mask2 = torch.from_numpy(mask2).unsqueeze(0).to(device)
            
            predictions = model(events1, mask1, events2, mask2, K)
            
            R_rel = predictions['R_pred'][0]
            t_rel = predictions['t_pred'][0]
            
            # Bug fix 1: Update translation FIRST using old rotation, THEN update rotation
            t_abs = t_abs + R_abs @ t_rel
            R_abs = R_abs @ R_rel
            
            quat = quaternion_from_matrix(R_abs)
            
            trajectory.append([t2, t_abs[0].item(), t_abs[1].item(), t_abs[2].item(), 
                             quat[0], quat[1], quat[2], quat[3]])
    
    trajectory = np.array(trajectory)
    
    traj_file = output_dir / 'trajectory.txt'
    np.savetxt(traj_file, trajectory, fmt='%.6f')
    print(f"✓ Trajectory saved to {traj_file}")
    print(f"Total frames: {len(trajectory)}")

if __name__ == '__main__':
    main()