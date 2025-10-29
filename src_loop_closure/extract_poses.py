import argparse
import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation


def parse_pose_file(pose_path):
    """Parse pose.txt file and return timestamps, positions, and quaternions."""
    data = np.loadtxt(pose_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # tx, ty, tz
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    return timestamps, positions, quaternions


def get_event_timestamps(h5_path):
    """Extract timestamps from event h5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Try common timestamp keys
        for key in ['events/ts', 'events/t', 'events/timestamp', 'events/timestamps', 'ts', 't', 'timestamp', 'timestamps']:
            if key in f:
                return f[key][:]
        
        # If not found, search recursively
        def find_timestamps(group):
            for key in group.keys():
                if 'time' in key.lower() or key in ['ts', 't']:
                    return group[key][:]
                if isinstance(group[key], h5py.Group):
                    result = find_timestamps(group[key])
                    if result is not None:
                        return result
            return None
        
        result = find_timestamps(f)
        if result is not None:
            return result
    
    raise KeyError(f"Could not find timestamp data in {h5_path}")


def interpolate_poses(pose_times, positions, quaternions, target_times):
    """Interpolate poses to target timestamps using linear interpolation for positions and slerp for quaternions."""
    # Ensure timestamps are sorted
    sort_idx = np.argsort(pose_times)
    pose_times = pose_times[sort_idx]
    positions = positions[sort_idx]
    quaternions = quaternions[sort_idx]
    
    # Filter target times to be within pose time range
    valid_mask = (target_times >= pose_times[0]) & (target_times <= pose_times[-1])
    valid_target_times = target_times[valid_mask]
    
    if len(valid_target_times) == 0:
        print(f"Warning: No overlapping timestamps between pose and events")
        return target_times, np.zeros((len(target_times), 3)), np.zeros((len(target_times), 4))
    
    # Interpolate positions (linear)
    pos_interp = interp1d(pose_times, positions, axis=0, kind='linear', fill_value='extrapolate')
    interp_positions = pos_interp(target_times)
    
    # Interpolate quaternions (slerp)
    # Create Rotation object and use Slerp
    rotations = Rotation.from_quat(quaternions)  # expects (x, y, z, w) format
    slerp = Slerp(pose_times, rotations)
    interp_rotations = slerp(target_times)
    interp_quaternions = interp_rotations.as_quat()  # returns (x, y, z, w) format
    
    return target_times, interp_positions, interp_quaternions


def is_mvsec_sequence(seq_name):
    """Check if sequence is from MVSEC dataset."""
    seq_lower = seq_name.lower()
    return 'indoor' in seq_lower or 'outdoor' in seq_lower


def process_sequence(seq_path, output_h5, sample_rate_hz=100):
    """Process a single sequence and add to output HDF5 file."""
    seq_name = seq_path.name
    print(f"Processing sequence: {seq_name}")
    
    # Determine dataset type
    is_mvsec = is_mvsec_sequence(seq_name)
    dataset_type = "MVSEC" if is_mvsec else "TUM-VIE"
    print(f"  Dataset type: {dataset_type}")
    
    # Find pose file
    pose_files = list(seq_path.glob("pose.txt")) + list(seq_path.glob("*.txt"))
    if not pose_files:
        print(f"  Warning: No pose file found in {seq_path}, skipping")
        return
    pose_file = pose_files[0]
    
    # Parse pose data
    pose_times, positions, quaternions = parse_pose_file(pose_file)
    print(f"  Loaded {len(pose_times)} poses")
    
    # Find event files to get time range
    event_files = list(seq_path.glob("*events*.h5")) + list(seq_path.glob("*.h5"))
    if not event_files:
        print(f"  Warning: No event h5 files found in {seq_path}, skipping")
        return
    
    # Get time range from event files (just min/max, not all timestamps)
    min_time = float('inf')
    max_time = float('-inf')
    
    for event_file in event_files:
        try:
            with h5py.File(event_file, 'r') as f:
                # Try common timestamp keys
                ts_data = None
                for key in ['events/ts', 'events/t', 'events/timestamp', 'ts', 't']:
                    if key in f:
                        ts_data = f[key]
                        break
                
                if ts_data is not None:
                    # Read only first and last timestamp (efficient!)
                    min_time = min(min_time, float(ts_data[0]))
                    max_time = max(max_time, float(ts_data[-1]))
        except Exception as e:
            print(f"  Warning: Could not get time range from {event_file.name}: {e}")
    
    if min_time == float('inf') or max_time == float('-inf'):
        print(f"  Warning: Could not determine time range, skipping")
        return
    
    print(f"  Raw time range: {min_time} to {max_time}")
    
    # Calculate dt based on dataset type
    if is_mvsec:
        # MVSEC: Unix time in microseconds
        dt = 1e6 / sample_rate_hz  # microseconds per sample
        print(f"  Using dt = {dt} microseconds (MVSEC Unix time)")
    else:
        # TUM-VIE: Offset microseconds from start
        dt = 1e6 / sample_rate_hz  # microseconds per sample
        print(f"  Using dt = {dt} microseconds (TUM-VIE offset time)")
    
    # Calculate duration
    duration_sec = (max_time - min_time) / 1e6
    print(f"  Sequence duration: {duration_sec:.2f} seconds")
    
    # Sanity check
    if duration_sec > 7200:  # More than 2 hours
        print(f"  Warning: Duration {duration_sec:.2f}s seems too long. Skipping.")
        return
    if duration_sec < 1:  # Less than 1 second
        print(f"  Warning: Duration {duration_sec:.2f}s seems too short. Skipping.")
        return
    
    # Ensure we're within the pose time range
    min_time = max(min_time, pose_times[0])
    max_time = min(max_time, pose_times[-1])
    
    # Generate regular timestamps at desired sample rate
    sampled_times = np.arange(min_time, max_time, dt)
    print(f"  Generated {len(sampled_times)} timestamps at {sample_rate_hz}Hz")
    
    # Interpolate poses to sampled timestamps
    interp_times, interp_pos, interp_quat = interpolate_poses(
        pose_times, positions, quaternions, sampled_times
    )
    
    # Save to output HDF5
    grp = output_h5.create_group(seq_name)
    grp.create_dataset('timestamps', data=interp_times, compression='gzip')
    grp.create_dataset('positions', data=interp_pos, compression='gzip')
    grp.create_dataset('quaternions', data=interp_quat, compression='gzip')
    
    print(f"  Saved {len(interp_times)} interpolated poses to /{seq_name}/")


def main():
    parser = argparse.ArgumentParser(
        description='Convert pose.txt files to poses.h5 with interpolation to sampled timestamps'
    )
    parser.add_argument('input_folder', type=str, help='Path to folder containing sequence subdirectories')
    parser.add_argument('--output', type=str, default='poses.h5', help='Output HDF5 file path (default: poses.h5)')
    parser.add_argument('--sample-rate', type=int, default=100, help='Sample rate in Hz (default: 100)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder {input_path} does not exist")
        return
    
    # Find all sequence directories
    seq_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not seq_dirs:
        print(f"Error: No subdirectories found in {input_path}")
        return
    
    print(f"Found {len(seq_dirs)} sequences")
    print(f"Sample rate: {args.sample_rate}Hz")
    
    # Create output HDF5 file
    with h5py.File(args.output, 'w') as out_h5:
        for seq_dir in sorted(seq_dirs):
            try:
                process_sequence(seq_dir, out_h5, sample_rate_hz=args.sample_rate)
            except Exception as e:
                print(f"Error processing {seq_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nDone! Output saved to {args.output}")


if __name__ == '__main__':
    main()