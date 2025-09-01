import numpy as np
import h5py
import argparse

def npz_to_h5(npz_file, h5_file):
    data = np.load(npz_file, allow_pickle=True)
    events = data["events"]

    # Convert fields to desired types
    x = events["x"].astype(np.uint16)
    y = events["y"].astype(np.uint16)
    t = (events["t"] * 1e6).astype(np.int64)  # seconds → microseconds
    p = ((events["p"] + 1) // 2).astype(np.uint8)


    # Build ms_to_idx mapping (per millisecond)
    t_ms = t // 1000
    t_min = t_ms.min()
    t_max = t_ms.max()
    num_ms = int(t_max - t_min + 1)
    ms_to_idx = np.zeros(num_ms, dtype=np.int64)

    curr_idx = 0
    for i, ms in enumerate(range(t_min, t_max + 1)):
        while curr_idx < len(t_ms) and t_ms[curr_idx] < ms:
            curr_idx += 1
        ms_to_idx[i] = curr_idx

    # Write HDF5
    with h5py.File(h5_file, "w") as f:
        events_group = f.create_group("events")
        events_group.create_dataset("x", data=x, compression="gzip")
        events_group.create_dataset("y", data=y, compression="gzip")
        events_group.create_dataset("t", data=t, compression="gzip")
        events_group.create_dataset("p", data=p, compression="gzip")
        f.create_dataset("ms_to_idx", data=ms_to_idx, compression="gzip")

    print(f"✅ Converted {npz_file} → {h5_file} with correct datatypes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_h5", required=True)
    args = parser.parse_args()

    npz_to_h5(args.input_npz, args.output_h5)
