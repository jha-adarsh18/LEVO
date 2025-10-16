import numpy as np
import os
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import time
from glob import glob
import h5py
import hdf5plugin

_worker_datasets = None
_h5_files = None

def init_worker(dataset_root):
    global _worker_datasets, _h5_files
    h5_files = []
    for pattern in ["*/*events_left.h5", "*/*-events_left.h5", "*/left_events.h5"]:
        h5_files.extend(glob(os.path.join(dataset_root, pattern)))
    _h5_files = sorted(list(set(h5_files)))
    
    # Keep H5 files open for entire worker lifetime
    _worker_datasets = {h5_path: h5py.File(h5_path, 'r') for h5_path in _h5_files}
    print(f"Worker initialized with {len(_h5_files)} H5 files")

def get_dataset_type(seq_name):
    seq_lower = seq_name.lower()
    # MVSEC has indoor_* and outdoor_* (including typos like "ourdoor")
    if 'indoor' in seq_lower or 'outdoor' in seq_lower or 'ourdoor' in seq_lower:
        return 'mvsec'
    return 'tum'

def get_events_from_h5(h5file, t_start, t_end, use_ms_idx=True):
    if use_ms_idx and "ms_to_idx" in h5file:
        ms_to_idx = h5file["ms_to_idx"][:]
        ms_start = int(t_start / 1000)
        ms_end = int(t_end / 1000)
        start_idx = int(ms_to_idx[ms_start]) if ms_start < len(ms_to_idx) else len(h5file["events/t"]) - 1
        end_idx = int(ms_to_idx[ms_end]) if ms_end < len(ms_to_idx) else len(h5file["events/t"])
    else:
        t_data = h5file["events/t"]
        start_idx = np.searchsorted(t_data, t_start, side="left")
        end_idx = np.searchsorted(t_data, t_end, side='right')

    if start_idx >= end_idx:
        return np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
    
    x = np.array(h5file["events/x"][start_idx:end_idx])
    y = np.array(h5file["events/y"][start_idx:end_idx])
    t = np.array(h5file["events/t"][start_idx:end_idx])
    p = np.array(h5file["events/p"][start_idx:end_idx])

    events = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
    events['x'] = x
    events['y'] = y
    events['t'] = t
    events['p'] = p.astype(np.uint8)
    return events

def chunk_events(events, Np=1024):
    if len(events) == 0:
        return []
    
    # Simple chunking: split into Np-sized packets
    packets = []
    for i in range(0, len(events), Np):
        packet = events[i:i+Np]
        if len(packet) > 0:
            packets.append(packet)
    return packets

def save_packets_worker(args):
    file_idx, window_batch, output_dir, seq_prefix, use_ms_idx = args
    
    try:
        h5_path = _h5_files[file_idx]
        h5file = _worker_datasets[h5_path]
        
        packets_saved = 0
        total_events = 0
        
        for window_idx, (t_start, t_end) in window_batch:
            events = get_events_from_h5(h5file, t_start, t_end, use_ms_idx)
            if len(events) == 0:
                continue
            
            # Chunk into 1024-event packets
            packets = chunk_events(events, Np=1024)
            
            for packet_idx, packet in enumerate(packets):
                output_path = os.path.join(
                    output_dir, 
                    f'{seq_prefix}_w{window_idx:06d}_p{packet_idx:03d}.npz'
                )
                np.savez_compressed(
                    output_path,
                    events=packet,
                    time_window=np.array([t_start, t_end]),
                    time_mid=(t_start + t_end) / 2.0,
                    sequence=seq_prefix
                )
                packets_saved += 1
                total_events += len(packet)
        
        return file_idx, True, None, packets_saved, total_events
    except Exception as e:
        return file_idx, False, str(e), 0, 0

def get_dataset_info(dataset_root, time_window_us=5000, stride_us=None, sample_rate=1):
    if stride_us is None:
        stride_us = time_window_us
    
    h5_files = []
    for pattern in ["*/*events_left.h5", "*/*-events_left.h5", "*/left_events.h5"]:
        h5_files.extend(glob(os.path.join(dataset_root, pattern)))
    h5_files = sorted(list(set(h5_files)))
    
    if not h5_files:
        raise FileNotFoundError(f"No left events HDF5 files found in {dataset_root}/")
    
    print(f"Found {len(h5_files)} H5 files")
    
    file_info = []
    mvsec_counter = 0
    tum_counter = 0
    
    for h5_path in h5_files:
        parent_dir = os.path.basename(os.path.dirname(h5_path))
        seq_name = parent_dir
        dataset_type = get_dataset_type(seq_name)
        
        if dataset_type == 'mvsec':
            seq_prefix = f'mvsec_seq{mvsec_counter:02d}'
            use_ms_idx = False
            mvsec_counter += 1
        else:
            seq_prefix = f'tum_seq{tum_counter:02d}'
            use_ms_idx = True
            tum_counter += 1
        
        with h5py.File(h5_path, 'r') as h5file:
            time_start = float(h5file["events/t"][0])
            time_end = float(h5file["events/t"][-1])
        
        # Sample windows if sample_rate < 1
        time_windows = []
        current_time = time_start
        window_counter = 0
        while current_time < time_end:
            window_end = min(current_time + time_window_us, time_end)
            if sample_rate >= 1 or (window_counter % int(1/sample_rate)) == 0:
                time_windows.append((current_time, window_end))
            current_time += stride_us
            window_counter += 1
        
        file_info.append({
            'path': h5_path,
            'sequence': seq_name,
            'dataset_type': dataset_type,
            'seq_prefix': seq_prefix,
            'use_ms_idx': use_ms_idx,
            'time_windows': time_windows,
            'time_start': time_start,
            'time_end': time_end
        })
        
        print(f"  {seq_prefix}: {seq_name} - {len(time_windows)} windows")
    
    return file_info

def convert_parallel(dataset_root, output_dir, num_workers=None, 
                     time_window_us=5000, stride_us=None, max_sequences=None, 
                     batch_size=100, sample_rate=1.0):
    
    os.makedirs(output_dir, exist_ok=True)
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    print(f"\nDataset: {dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}, Batch: {batch_size} windows/task")
    if sample_rate < 1.0:
        print(f"Sampling: {sample_rate*100:.1f}% of windows\n")
    else:
        print()
    
    file_info = get_dataset_info(dataset_root, time_window_us, stride_us, sample_rate)
    if max_sequences:
        file_info = file_info[:max_sequences]
    
    mvsec_seqs = [f for f in file_info if f['dataset_type'] == 'mvsec']
    tum_seqs = [f for f in file_info if f['dataset_type'] == 'tum']
    
    print(f"\nMVSEC: {len(mvsec_seqs)}, TUM-VIE: {len(tum_seqs)}")
    total_windows = sum(len(info['time_windows']) for info in file_info)
    print(f"Total windows: {total_windows:,}\n")
    
    # Split windows into batches
    worker_args = []
    for file_idx, info in enumerate(file_info):
        windows = [(i, w) for i, w in enumerate(info['time_windows'])]
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            worker_args.append((file_idx, batch, output_dir, info['seq_prefix'], info['use_ms_idx']))
    
    print(f"Created {len(worker_args)} tasks\n")
    
    start_time = time.time()
    total_packets = 0
    total_events = 0
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(dataset_root,)) as pool:
        for file_idx, success, error, packets_saved, events_count in tqdm(
            pool.imap_unordered(save_packets_worker, worker_args),
            total=len(worker_args),
            desc="Converting",
            unit="batch"
        ):
            if success:
                total_packets += packets_saved
                total_events += events_count
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Packets: {total_packets:,}")
    print(f"Events: {total_events:,}")
    print(f"Time: {elapsed_time/60:.1f} min ({total_packets/elapsed_time:.0f} packets/sec)")
    
    npz_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in npz_files)
    print(f"Disk: {total_size / (1024**3):.2f} GB ({total_size/len(npz_files)/1024:.1f} KB/packet)")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--workers", "-w", type=int, default=None)
    parser.add_argument("--time-window", "-t", type=int, default=5000)
    parser.add_argument("--stride", "-s", type=int, default=None)
    parser.add_argument("--max-sequences", "-m", type=int, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=100)
    parser.add_argument("--sample-rate", "-r", type=float, default=1.0,
                       help="Sample rate (0.1 = 10%% of windows, 1.0 = all)")
    args = parser.parse_args()
    
    convert_parallel(args.dataset, args.output, args.workers, 
                    args.time_window, args.stride, args.max_sequences, 
                    args.batch_size, args.sample_rate)