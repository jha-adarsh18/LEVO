import h5py
import numpy as np
import hdf5plugin   # enables BLOSC/LZ4/SZ filters

h5_file = "/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor-flying1-events_right.h5"

with h5py.File(h5_file, "r") as f:
    # Print high-level structure
    print("Top-level keys:", list(f.keys()))
    if "events" in f:
        print("Events keys:", list(f["events"].keys()))
    if "ms_to_idx" in f:
        print("ms_to_idx shape:", f["ms_to_idx"].shape)

    # Access datasets
    x = f["events/x"][:5]
    y = f["events/y"][:5]
    t = f["events/t"][:5]
    p = f["events/p"][:5]

    print("\nSample 5 events with full timestamps:")
    for i in range(len(x)):
        print(f"Event {i}: x={x[i]}, y={y[i]}, t={int(t[i])}, p={p[i]}")
