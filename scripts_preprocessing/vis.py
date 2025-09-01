import numpy as np
import matplotlib.pyplot as plt

def load_events(npz_path):
    """Load event data from an MVSEC .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return data['events']

def get_sensor_size(events):
    """Automatically infer sensor resolution from event data."""
    max_x = int(events['x'].max()) + 1
    max_y = int(events['y'].max()) + 1
    return (max_y, max_x)  # (height, width)

def create_colored_polarity_map(events, sensor_size):
    """
    Create an RGB image where:
    - Red channel is set if polarity == +1
    - Blue channel is set if polarity == -1
    """
    img = np.zeros((sensor_size[0], sensor_size[1], 3), dtype=np.uint8)

    for e in events:
        x, y, p = int(e['x']), int(e['y']), int(e['p'])
        if 0 <= x < sensor_size[1] and 0 <= y < sensor_size[0]:
            if p == 1:
                img[y, x, 0] = 255  # Red
            elif p == -1:
                img[y, x, 2] = 255  # Blue
    return img

def visualize_polarity_colored(npz_path, title='Polarity Map'):
    events = load_events(npz_path)
    sensor_size = get_sensor_size(events)
    color_img = create_colored_polarity_map(events, sensor_size)

    plt.figure(figsize=(7, 6))
    plt.imshow(color_img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set your MVSEC .npz file paths
    left_path = "/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor_flying1/left_Events.npz"
    right_path = "/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor_flying1/right_Events.npz"

    visualize_polarity_colored(left_path, title='Left Camera Polarity Map')
    visualize_polarity_colored(right_path, title='Right Camera Polarity Map')
