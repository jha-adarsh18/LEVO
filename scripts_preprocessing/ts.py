import numpy as np

# Load your events from .npz or .npy
events = np.load('/home/adarsh/Documents/SRM/left_Events.npz')['events']  # or use np.load('left_events.npy', allow_pickle=True).item()['events']

# Extract timestamps
timestamps = events['t']  # dtype is typically float64, in seconds

# Optional: sort and remove duplicates
timestamps = np.sort(np.unique(timestamps))

# Save to file
with open('timestamps.txt', 'w') as f:
    for t in timestamps:
        f.write(f"{t:.9f}\n")  # full nanosecond precision
