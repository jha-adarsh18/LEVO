import numpy as np
import os
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import time
from glob import glob

from utils.loadevents import event_extractor

# Global variable for worker processes
_worker_datasets = None
_h5_files = None

def init_worker(dataset_root):
    """Initialize each worker with connections to all left_events.h5 files"""
    global _worker_datasets, _h5_files
    
    # Find all left_events.h5 files across sequences
    _h5_files = sorted(glob(os.path.join(dataset_root, "*/left_events.h5")))
    if not _h5_files:
        raise FileNotFoundError(f"No left_events.h5 files found in {dataset_root}/*/")
    
    # Open all datasets once per worker
    _worker_datasets = {
        h5_path: event_extractor(h5_path) 
        for h5_path in _h5_files
    }

def get_dataset_type(seq_name):
    """Determine if sequence is MVSEC or TUM-VIE based on name"""
    seq_lower = seq_name.lower()
    if 'indoor' in seq_lower or 'outdoor' in seq_lower:
        return 'mvsec'
    else:
        return 'tum'

def save_sample_worker(args):
    """Worker function - load from appropriate HDF5 file based on index"""
    file_idx, sample_idx, output_dir, seq_prefix = args
    
    try:
        h5_path = _h5_files[file_idx]
        dataset = _worker_datasets[h5_path]
        
        sample = dataset[sample_idx]
        events = sample['left_events_strip']
        seq_info = sample['sequence_info']
        
        # Use the pre-computed prefix for naming
        output_path = os.path.join(output_dir, f'{seq_prefix}_s{sample_idx:08d}.npz')
        np.savez_compressed(
            output_path,
            events=events,
            sequence=seq_info['sequence'],
            time_window=seq_info['time_window']
        )
        
        return file_idx, sample_idx, True, None
    except Exception as e:
        return file_idx, sample_idx, False, str(e)

def get_dataset_info(dataset_root):
    """Get info about all left_events.h5 files and their sample counts"""
    h5_files = sorted(glob(os.path.join(dataset_root, "*/left_events.h5")))
    
    if not h5_files:
        raise FileNotFoundError(f"No left_events.h5 files found in {dataset_root}/*/")
    
    print(f"\n📂 Found {len(h5_files)} sequences with left_events.h5")
    
    file_info = []
    total_samples = 0
    
    # Track sequence counters for each dataset type
    mvsec_counter = 0
    tum_counter = 0
    
    for h5_path in h5_files:
        seq_name = os.path.basename(os.path.dirname(h5_path))
        dataset_type = get_dataset_type(seq_name)
        
        # Assign sequential numbering based on type
        if dataset_type == 'mvsec':
            seq_prefix = f'mvsec_seq{mvsec_counter:02d}'
            mvsec_counter += 1
        else:
            seq_prefix = f'tum_seq{tum_counter:02d}'
            tum_counter += 1
        
        dataset = event_extractor(h5_path)
        num_samples = len(dataset)
        
        file_info.append({
            'path': h5_path,
            'sequence': seq_name,
            'dataset_type': dataset_type,
            'seq_prefix': seq_prefix,
            'num_samples': num_samples
        })
        total_samples += num_samples
        del dataset
    
    return file_info, total_samples

def convert_parallel(dataset_root, output_dir, num_workers=None, max_samples=None, chunksize=10):
    """
    Convert left_events.h5 from all sequences to NPZ format using parallel processing
    
    Args:
        dataset_root: Path containing sequence folders (e.g., /workspace/train-split)
        output_dir: Directory to save NPZ files
        num_workers: Number of parallel workers (default: CPU count - 2)
        max_samples: Limit conversion to first N samples across all sequences
        chunksize: Number of samples per task (default: 10)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers is None:
        # For 8 available CPUs, use 6 workers
        num_workers = max(1, cpu_count() - 2)
    
    print("\n" + "="*70)
    print("🚀 Left Events HDF5 → NPZ Converter")
    print("="*70)
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Chunksize: {chunksize}")
    
    # Get info about all left_events.h5 files
    print("\n📂 Scanning sequences...")
    file_info, total_samples = get_dataset_info(dataset_root)
    
    # Display sequence information grouped by type
    print("\n" + "="*70)
    print("📊 Sequence Information")
    print("="*70)
    
    mvsec_seqs = [f for f in file_info if f['dataset_type'] == 'mvsec']
    tum_seqs = [f for f in file_info if f['dataset_type'] == 'tum']
    
    if mvsec_seqs:
        print("\n🎥 MVSEC Sequences:")
        for info in mvsec_seqs:
            print(f"  {info['seq_prefix']:12s} <- {info['sequence']:30s} ({info['num_samples']:,} samples)")
    
    if tum_seqs:
        print("\n🎥 TUM-VIE Sequences:")
        for info in tum_seqs:
            print(f"  {info['seq_prefix']:12s} <- {info['sequence']:30s} ({info['num_samples']:,} samples)")
    
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    print("\n" + "="*70)
    print(f"📊 Total samples to convert: {total_samples:,}")
    print("="*70)
    
    # Build worker arguments: (file_idx, sample_idx, output_dir, seq_prefix)
    print("\n⚙️  Preparing worker arguments...")
    worker_args = []
    samples_added = 0
    
    for file_idx, info in enumerate(file_info):
        for sample_idx in range(info['num_samples']):
            if max_samples and samples_added >= max_samples:
                break
            worker_args.append((file_idx, sample_idx, output_dir, info['seq_prefix']))
            samples_added += 1
        
        if max_samples and samples_added >= max_samples:
            break
    
    # Process in parallel
    print(f"\n🔄 Converting left_events.h5 samples...")
    print(f"⚡ Each worker opens {len(file_info)} files ONCE (very efficient!)")
    print(f"⚡ Using chunksize={chunksize} for optimal throughput")
    start_time = time.time()
    
    failed_samples = []
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(dataset_root,)) as pool:
        results = list(tqdm(
            pool.imap_unordered(save_sample_worker, worker_args, chunksize=chunksize),
            total=len(worker_args),
            desc="Converting",
            unit="samples",
            smoothing=0.1
        ))
    
    # Check for failures
    for file_idx, sample_idx, success, error in results:
        if not success:
            failed_samples.append((file_idx, sample_idx, error))
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("✅ Conversion Complete!")
    print("="*70)
    print(f"Sequences processed: {len(file_info)}")
    print(f"  - MVSEC: {len(mvsec_seqs)}")
    print(f"  - TUM-VIE: {len(tum_seqs)}")
    print(f"Total samples: {len(worker_args):,}")
    print(f"Successful: {len(worker_args) - len(failed_samples):,}")
    print(f"Failed: {len(failed_samples)}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
    print(f"Average speed: {len(worker_args)/elapsed_time:.1f} samples/sec")
    
    # Calculate disk usage
    try:
        npz_files = [f for f in os.listdir(output_dir) if f.endswith('.npz') and not f.startswith('conversion_')]
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in npz_files
        )
        print(f"Total disk usage: {total_size / (1024**3):.2f} GB")
        print(f"Average file size: {total_size / len(npz_files) / 1024:.2f} KB")
    except Exception as e:
        print(f"Could not calculate disk usage: {e}")
    
    if failed_samples:
        print(f"\n⚠️  Failed samples: {len(failed_samples)}")
        print("First 5 failures:")
        for file_idx, sample_idx, error in failed_samples[:5]:
            seq_info = file_info[file_idx]
            print(f"  {seq_info['seq_prefix']} (sample {sample_idx}): {error}")
    
    print("="*70)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'conversion_metadata.npz')
    np.savez(
        metadata_path,
        total_samples=len(worker_args),
        successful_samples=len(worker_args) - len(failed_samples),
        failed_indices=[(f, s) for f, s, _ in failed_samples],
        conversion_time=elapsed_time,
        num_sequences=len(file_info),
        sequences=[info['seq_prefix'] for info in file_info],
        original_names=[info['sequence'] for info in file_info],
        dataset_types=[info['dataset_type'] for info in file_info]
    )
    print(f"\n💾 Metadata saved to: {metadata_path}")
    
    # Save detailed sequence mapping
    mapping_path = os.path.join(output_dir, 'sequence_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write("Sequence Name Mapping\n")
        f.write("="*70 + "\n\n")
        f.write("MVSEC Sequences:\n")
        for info in mvsec_seqs:
            f.write(f"  {info['seq_prefix']:12s} <- {info['sequence']}\n")
        f.write("\nTUM-VIE Sequences:\n")
        for info in tum_seqs:
            f.write(f"  {info['seq_prefix']:12s} <- {info['sequence']}\n")
    print(f"💾 Sequence mapping saved to: {mapping_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert left_events.h5 from all sequences to NPZ format")
    
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        type=str,
        help="Path to root directory containing sequence folders (e.g., /workspace/train-split)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Output directory for NPZ files"
    )
    
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 2, suggested: 6 for your 8-core setup)"
    )
    
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Limit conversion to first N samples (default: all)"
    )
    
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=10,
        help="Number of samples per task (default: 10, try 20-50 for faster processing)"
    )
    
    args = parser.parse_args()
    
    convert_parallel(args.dataset, args.output, args.workers, args.max_samples, args.chunksize)