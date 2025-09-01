import numpy as np
import argparse
import os

def convert_unix_to_us(input_file, output_file=None):
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_us{ext}"

    # Load the pose file
    data = np.loadtxt(input_file)
    if data.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in pose file, got {data.shape[1]}")

    # Convert first column from seconds → microseconds
    data[:, 0] = (data[:, 0] * 1e6).astype(np.int64)

    # Save to new file
    np.savetxt(output_file, data, fmt=['%d'] + ['%.6f']*7, delimiter=' ')
    print(f"✅ Converted timestamps to microseconds and saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UNIX timestamps in pose.txt to microseconds")
    parser.add_argument("--input_file", required=True, help="Path to input pose.txt")
    parser.add_argument("--output_file", default=None, help="Optional output file path")
    args = parser.parse_args()

    convert_unix_to_us(args.input_file, args.output_file)
