#!/usr/bin/env python3
"""
Convert pose.txt files to poses.h5 with interpolation to event timestamps.

Input structure:
  - folder/
    - seq_name_1/
      - pose.txt (timestamp, tx, ty, tz, qx, qy, qz, qw)
      - left_events.h5 (contains timestamps)
      - right_events.h5 (contains timestamps)
    - seq_name_2/
      ...

Output: poses.h5 with structure:
  /seq_name_1/timestamps [N]
  /seq_name_1/positions [N, 3]
  /seq_name_1/quaternions [N, 4]
  /seq_name_2/...
"""

import argparse
import h5py
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


def process_sequence(seq_path, output_h5):
    """Process a single sequence and add to output HDF5 file."""
    seq_name = seq_path.name
    print(f"Processing sequence: {seq_name}")
    
    # Find pose file
    pose_files = list(seq_path.glob("pose.txt")) + list(seq_path.glob("*.txt"))
    if not pose_files:
        print(f"  Warning: No pose file found in {seq_path}, skipping")
        return
    pose_file = pose_files[0]
    
    # Parse pose data
    pose_times, positions, quaternions = parse_pose_file(pose_file)
    print(f"  Loaded {len(pose_times)} poses")
    
    # Find event files
    event_files = list(seq_path.glob("*events*.h5")) + list(seq_path.glob("*.h5"))
    if not event_files:
        print(f"  Warning: No event h5 files found in {seq_path}, skipping")
        return
    
    # Collect all unique event timestamps from all h5 files
    all_event_times = []
    for event_file in event_files:
        try:
            event_times = get_event_timestamps(event_file)
            all_event_times.append(event_times)
            print(f"  Loaded {len(event_times)} timestamps from {event_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load timestamps from {event_file.name}: {e}")
    
    if not all_event_times:
        print(f"  Warning: No valid event timestamps found, skipping")
        return
    
    # Merge and sort all event timestamps
    merged_times = np.concatenate(all_event_times)
    unique_times = np.unique(merged_times)
    print(f"  Total unique event timestamps: {len(unique_times)}")
    
    # Interpolate poses to event timestamps
    interp_times, interp_pos, interp_quat = interpolate_poses(
        pose_times, positions, quaternions, unique_times
    )
    
    # Save to output HDF5
    grp = output_h5.create_group(seq_name)
    grp.create_dataset('timestamps', data=interp_times, compression='gzip')
    grp.create_dataset('positions', data=interp_pos, compression='gzip')
    grp.create_dataset('quaternions', data=interp_quat, compression='gzip')
    
    print(f"  Saved {len(interp_times)} interpolated poses to /{seq_name}/")


def main():
    parser = argparse.ArgumentParser(
        description='Convert pose.txt files to poses.h5 with interpolation to event timestamps'
    )
    parser.add_argument('input_folder', type=str, help='Path to folder containing sequence subdirectories')
    parser.add_argument('--output', type=str, default='poses.h5', help='Output HDF5 file path (default: poses.h5)')
    
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
    
    # Create output HDF5 file
    with h5py.File(args.output, 'w') as out_h5:
        for seq_dir in sorted(seq_dirs):
            try:
                process_sequence(seq_dir, out_h5)
            except Exception as e:
                print(f"Error processing {seq_dir.name}: {e}")
                continue
    
    print(f"\nDone! Output saved to {args.output}")


if __name__ == '__main__':
    main()