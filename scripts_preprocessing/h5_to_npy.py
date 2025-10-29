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
    
    _worker_datasets = {h5_path: h5py.File(h5_path, 'r') for h5_path in _h5_files}
    print(f"Worker initialized with {len(_h5_files)} H5 files")

def get_dataset_type(seq_name):
    seq_lower = seq_name.lower()
    if 'indoor' in seq_lower or 'outdoor' in seq_lower or 'ourdoor' in seq_lower:
        return 'mvsec'
    return 'tum'

def get_all_events(h5file, dataset_type='tum', tum_subsample=0.2):
    """Load all events from H5 file efficiently"""
    total_events = len(h5file["events/x"])
    
    # Subsample TUM-VIE to first 20%
    if dataset_type == 'tum':
        end_idx = int(total_events * tum_subsample)
    else:
        end_idx = total_events
    
    x = np.array(h5file["events/x"][:end_idx])
    y = np.array(h5file["events/y"][:end_idx])
    t = np.array(h5file["events/t"][:end_idx])
    p = np.array(h5file["events/p"][:end_idx])

    events = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'u1')])
    events['x'] = x
    events['y'] = y
    events['t'] = t
    events['p'] = p.astype(np.uint8)
    return events

def save_windows_worker(args):
    file_idx, window_batch, output_dir, seq_prefix, dataset_type = args
    
    try:
        h5_path = _h5_files[file_idx]
        h5file = _worker_datasets[h5_path]
        
        # Load all events once (with TUM subsampling)
        all_events = get_all_events(h5file, dataset_type, tum_subsample=0.2)
        total_events = len(all_events)
        
        windows_saved = 0
        
        for window_idx, (start_idx, end_idx) in window_batch:
            if start_idx >= total_events or end_idx > total_events:
                continue
            
            window_events = all_events[start_idx:end_idx]
            
            if len(window_events) == 0:
                continue
            
            # Get time range for this window
            t_start = float(window_events['t'][0])
            t_end = float(window_events['t'][-1])
            t_mid = (t_start + t_end) / 2.0
            
            output_path = os.path.join(
                output_dir, 
                f'{seq_prefix}_w{window_idx:06d}.npz'
            )
            
            np.savez_compressed(
                output_path,
                events=window_events,
                time_start=t_start,
                time_end=t_end,
                time_mid=t_mid,
                sequence=seq_prefix
            )
            windows_saved += 1
        
        return file_idx, True, None, windows_saved
    except Exception as e:
        return file_idx, False, str(e), 0

def get_dataset_info(dataset_root, events_per_window=1024, stride_events=512, sample_rate=1.0, tum_subsample=0.2):
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
            mvsec_counter += 1
        else:
            seq_prefix = f'tum_seq{tum_counter:02d}'
            tum_counter += 1
        
        with h5py.File(h5_path, 'r') as h5file:
            total_events_raw = len(h5file["events/t"])
            
            # Apply TUM subsampling at counting stage
            if dataset_type == 'tum':
                total_events = int(total_events_raw * tum_subsample)
            else:
                total_events = total_events_raw
            
            time_start = float(h5file["events/t"][0])
            # Get end time from subsampled range
            time_end = float(h5file["events/t"][total_events - 1])
        
        # Create event-count based windows using SUBSAMPLED count
        event_windows = []
        current_idx = 0
        window_counter = 0
        
        while current_idx + events_per_window <= total_events:
            end_idx = current_idx + events_per_window
            
            # Sample windows based on sample_rate
            if sample_rate >= 1.0 or (window_counter % int(1/sample_rate)) == 0:
                event_windows.append((current_idx, end_idx))
            
            current_idx += stride_events
            window_counter += 1
        
        file_info.append({
            'path': h5_path,
            'sequence': seq_name,
            'dataset_type': dataset_type,
            'seq_prefix': seq_prefix,
            'event_windows': event_windows,
            'total_events': total_events,
            'total_events_raw': total_events_raw,
            'time_start': time_start,
            'time_end': time_end
        })
        
        subsample_note = f" (first {tum_subsample*100:.0f}%)" if dataset_type == 'tum' else ""
        print(f"  {seq_prefix}: {seq_name} - {len(event_windows)} windows ({total_events:,} events{subsample_note})")
    
    return file_info

def convert_parallel(dataset_root, output_dir, num_workers=None, 
                     events_per_window=1024, stride_events=512, 
                     max_sequences=None, batch_size=100, sample_rate=1.0, tum_subsample=0.2):
    
    os.makedirs(output_dir, exist_ok=True)
    if num_workers is None:
        num_workers = cpu_count() - 1  # Use all but 1 core
    
    print(f"\nDataset: {dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Events per window: {events_per_window}, Stride: {stride_events}")
    print(f"TUM-VIE subsample: {tum_subsample*100:.0f}%")
    print(f"Workers: {num_workers}, Batch: {batch_size} windows/task")
    if sample_rate < 1.0:
        print(f"Sampling: {sample_rate*100:.1f}% of windows\n")
    else:
        print()
    
    file_info = get_dataset_info(dataset_root, events_per_window, stride_events, sample_rate, tum_subsample)
    if max_sequences:
        file_info = file_info[:max_sequences]
    
    mvsec_seqs = [f for f in file_info if f['dataset_type'] == 'mvsec']
    tum_seqs = [f for f in file_info if f['dataset_type'] == 'tum']
    
    print(f"\nMVSEC: {len(mvsec_seqs)}, TUM-VIE: {len(tum_seqs)}")
    total_windows = sum(len(info['event_windows']) for info in file_info)
    print(f"Total windows: {total_windows:,}\n")
    
    # Split windows into batches
    worker_args = []
    for file_idx, info in enumerate(file_info):
        windows = [(i, w) for i, w in enumerate(info['event_windows'])]
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            worker_args.append((file_idx, batch, output_dir, info['seq_prefix'], info['dataset_type']))
    
    print(f"Created {len(worker_args)} tasks\n")
    
    start_time = time.time()
    total_windows_saved = 0
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(dataset_root,)) as pool:
        for file_idx, success, error, windows_saved in tqdm(
            pool.imap_unordered(save_windows_worker, worker_args),
            total=len(worker_args),
            desc="Converting",
            unit="batch"
        ):
            if success:
                total_windows_saved += windows_saved
            else:
                print(f"\nError in file {file_idx}: {error}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Windows saved: {total_windows_saved:,}")
    print(f"Time: {elapsed_time/60:.1f} min ({total_windows_saved/elapsed_time:.0f} windows/sec)")
    
    npz_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in npz_files)
    print(f"Disk: {total_size / (1024**3):.2f} GB ({total_size/len(npz_files)/1024:.1f} KB/window)")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--workers", "-w", type=int, default=None)
    parser.add_argument("--events-per-window", "-e", type=int, default=1024,
                       help="Number of events per window")
    parser.add_argument("--stride", "-s", type=int, default=512,
                       help="Stride in events (overlap = events_per_window - stride)")
    parser.add_argument("--max-sequences", "-m", type=int, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=100)
    parser.add_argument("--sample-rate", "-r", type=float, default=1.0,
                       help="Sample rate (0.1 = 10%% of windows, 1.0 = all)")
    parser.add_argument("--tum-subsample", "-t", type=float, default=0.2,
                       help="TUM-VIE subsample rate (0.2 = first 20%%, 1.0 = all)")
    args = parser.parse_args()
    
    convert_parallel(args.dataset, args.output, args.workers, 
                    args.events_per_window, args.stride, 
                    args.max_sequences, args.batch_size, args.sample_rate, args.tum_subsample)