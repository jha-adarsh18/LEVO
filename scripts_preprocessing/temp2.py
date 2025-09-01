import hdf5plugin   # must be imported FIRST
import h5py

file_path = "/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor-flying1-events_left.h5"

with h5py.File(file_path, "r") as f:
    events = f["events"]

    # First timestamp (full precision, no truncation)
    first_t = events["t"][4932837]
    print("First timestamp:", repr(first_t))

    # Dtypes of fields
    for key in ["x", "y", "t", "p"]:
        print(f"{key}: dtype={events[key].dtype}, shape={events[key].shape}")
