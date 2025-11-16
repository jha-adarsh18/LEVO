import hdf5plugin
import numpy as np
import h5py
import cv2
import yaml
import argparse

def load_intrinsics(config_path, seq_name='corner_slow1'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    K_params = config['intrinsics'][seq_name]
    resolution = tuple(K_params['resolution'])
    return resolution

def load_events_at_timestamp(h5_path, timestamp, window_ms, resolution):
    width, height = resolution
    with h5py.File(h5_path, 'r') as f:
        t_offset = f['t_offset'][()][0] / 1e6
        window_us = window_ms * 1000
        t_start = timestamp - window_us / 2e6
        t_end = timestamp + window_us / 2e6
        
        ms_to_idx = f['ms_to_idx']
        t_start_relative = t_start - t_offset
        t_end_relative = t_end - t_offset
        t_start_ms = max(0, int(t_start_relative * 1000))
        t_end_ms = int(t_end_relative * 1000)
        
        start_idx = ms_to_idx[t_start_ms] if t_start_ms < len(ms_to_idx) else 0
        end_idx = ms_to_idx[t_end_ms] if t_end_ms < len(ms_to_idx) else len(f['events']['t'])
        
        events_x = np.array(f['events']['x'][start_idx:end_idx])
        events_y = np.array(f['events']['y'][start_idx:end_idx])
        events_p = np.array(f['events']['p'][start_idx:end_idx])
    
    raw_events = np.stack([events_x, events_y, events_p], axis=1).astype(np.float32)
    return raw_events

def create_camera_like_image(raw_events, resolution):
    """
    Create a grayscale image that looks like a normal camera.
    Accumulates positive and negative events to reconstruct intensity.
    """
    width, height = resolution
    
    # Accumulate events - positive events add, negative events subtract
    img = np.zeros((height, width), dtype=np.float32)
    
    for event in raw_events:
        x, y, p = event
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            if p > 0:
                img[y, x] += 1
            else:
                img[y, x] -= 1
    
    # The accumulated difference represents edges/gradients
    # Take absolute value to get edge magnitude
    img_abs = np.abs(img)
    
    # Apply slight smoothing to make it more camera-like
    img_smooth = cv2.GaussianBlur(img_abs, (3, 3), 0.5)
    
    # Normalize to 0-255
    if img_smooth.max() > 0:
        # Use percentile for better contrast
        vmax = np.percentile(img_smooth, 99.5)
        img_norm = np.clip(img_smooth / vmax * 255, 0, 255).astype(np.uint8)
    else:
        img_norm = np.zeros((height, width), dtype=np.uint8)
    
    # Invert so edges are bright on dark background (like a normal image)
    img_norm = 255 - img_norm
    
    # Convert to color for display
    img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    
    return img_color

def create_edge_map(raw_events, resolution):
    """
    Alternative: Show events as edge activations (bright edges on dark background)
    """
    width, height = resolution
    
    # Just accumulate absolute event counts
    img = np.zeros((height, width), dtype=np.float32)
    
    for event in raw_events:
        x, y, p = event
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            img[y, x] += 1
    
    # Apply slight blur for smoother appearance
    img_smooth = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Normalize
    if img_smooth.max() > 0:
        vmax = np.percentile(img_smooth[img_smooth > 0], 99)
        img_norm = np.clip(img_smooth / vmax * 255, 0, 255).astype(np.uint8)
    else:
        img_norm = np.zeros((height, width), dtype=np.uint8)
    
    # Convert to color
    img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    
    return img_color

def create_reconstructed_image(raw_events, resolution):
    """
    Reconstruct a clean, sharp intensity image by integrating events.
    This simulates what a camera would see.
    """
    width, height = resolution
    
    # Start with a higher resolution accumulation for smoother result
    scale = 2
    img_hr = np.ones((height * scale, width * scale), dtype=np.float32) * 127
    
    # Accumulate events at higher resolution with spatial spread
    for event in raw_events:
        x, y, p = event
        x_hr, y_hr = int(x * scale), int(y * scale)
        
        # Add events with small spatial kernel for smoothness
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                xx, yy = x_hr + dx, y_hr + dy
                if 0 <= xx < width * scale and 0 <= yy < height * scale:
                    weight = 3.0 if (dx == 0 and dy == 0) else 1.0
                    if p > 0:
                        img_hr[yy, xx] += weight
                    else:
                        img_hr[yy, xx] -= weight
    
    # Clip and downsample
    img_hr = np.clip(img_hr, 0, 255).astype(np.uint8)
    img = cv2.resize(img_hr, (width, height), interpolation=cv2.INTER_AREA)
    
    # Apply bilateral filter for edge-preserving smoothing
    img_smooth = cv2.bilateralFilter(img, 7, 75, 75)
    
    # Gentle sharpening
    img_blur = cv2.GaussianBlur(img_smooth, (0, 0), 2.0)
    img_sharp = cv2.addWeighted(img_smooth, 1.5, img_blur, -0.5, 0)
    img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_sharp)
    
    # Convert to color
    img_color = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)
    
    return img_color

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        param['x'] = x
        param['y'] = y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--timestamps', type=float, nargs='+', required=True)
    parser.add_argument('--window-ms', type=int, default=50, 
                        help='Time window in milliseconds (default: 50ms)')
    parser.add_argument('--mode', type=str, default='camera',
                        choices=['camera', 'edges', 'reconstruct'],
                        help='Visualization mode: camera (inverted edges), edges (bright edges), reconstruct (intensity)')
    
    args = parser.parse_args()
    
    event_resolution = load_intrinsics(args.config)
    
    print(f"\nVisualization mode: {args.mode}")
    print(f"Time window: {args.window_ms}ms")
    print(f"Loading {len(args.timestamps)} images...")
    
    # Load all images first
    images = []
    titles = []
    for i, t in enumerate(args.timestamps):
        print(f"Loading image {i+1}/{len(args.timestamps)} at t={t:.6f}")
        raw_events = load_events_at_timestamp(args.h5_path, t, args.window_ms, event_resolution)
        print(f"  Found {len(raw_events)} events")
        
        if args.mode == 'camera':
            img = create_camera_like_image(raw_events, event_resolution)
        elif args.mode == 'edges':
            img = create_edge_map(raw_events, event_resolution)
        else:  # reconstruct
            img = create_reconstructed_image(raw_events, event_resolution)
        
        images.append(img)
        titles.append(f"Image {i+1} - t={t:.6f}")
    
    # Create all windows
    mouse_data_list = []
    for i, (img, title) in enumerate(zip(images, titles)):
        mouse_data = {'x': 0, 'y': 0}
        mouse_data_list.append(mouse_data)
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse_callback, mouse_data)
        # Position windows in a grid
        col = i % 2
        row = i // 2
        cv2.moveWindow(title, col * 650, row * 550)
    
    print("\nAll windows open. Hover mouse to see coordinates. Press any key to close all.")
    
    # Display loop for all windows
    while True:
        for i, (img, title, mouse_data) in enumerate(zip(images, titles, mouse_data_list)):
            display = img.copy()
            x, y = mouse_data['x'], mouse_data['y']
            
            # Draw coordinate text at top
            text = f"X: {x}  Y: {y}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(title, display)
        
        if cv2.waitKey(1) != -1:
            break
    
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == '__main__':
    main()