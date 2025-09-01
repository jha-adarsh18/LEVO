import numpy as np
import argparse

def print_first_ts(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    events = data["events"]   # structured array with fields (x, y, t, p)

    print("Fields:", events.dtype.names)

    t = events["t"]           # timestamps in seconds already
    print("Number of events:", len(t))
    print("First timestamp:", t[0])
    print("Last timestamp:", t[-1])
    print("Duration (s):", t[-1] - t[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    args = parser.parse_args()

    print_first_ts(args.input_npz)
