import h5py
import hdf5plugin

file_path = "/media/adarsh/One Touch/EventSLAM/dataset/train/bike_easy/bike-easy-events_left.h5"

with h5py.File(file_path, "r") as f:
    x = f["events/x"][:5]
    y = f["events/y"][:5]
    t = f["events/t"][:5]   # timestamps, usually in microseconds
    p = f["events/p"][:5]

    for i in range(5):
        print(f"Event {i}: x={x[i]}, y={y[i]}, t={t[i]}, p={p[i]}")
