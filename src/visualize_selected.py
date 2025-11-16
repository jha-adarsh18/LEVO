import hdf5plugin
import torch
import numpy as np
import h5py
import cv2
from pathlib import Path
import yaml
import argparse

def load_intrinsics(config_path, seq_name='corner_slow1'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    K_params = config['intrinsics'][seq_name]
    K_event = np.array([
        [K_params['fx'], 0, K_params['cx']],
        [0, K_params['fy'], K_params['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    resolution = tuple(K_params['resolution'])
    return K_event, resolution

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
        events_t = np.array(f['events']['t'][start_idx:end_idx])
        events_p = np.array(f['events']['p'][start_idx:end_idx])
        
        events_t = events_t / 1e6 + t_offset
    
    raw_events = np.stack([events_x, events_y, events_t, events_p], axis=1).astype(np.float32)
    return raw_events

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
        x, y, t, p = event  # Note: includes t even though we don't use it
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

def visualize_all_matches(images, matches_12, matches_13, matches_14):
    """
    Visualize matches across 4 images stacked horizontally with white separators
    matches_12: green (0, 255, 0) - image 1 to image 2
    matches_13: red (0, 0, 255) - image 1 to image 3
    matches_14: blue (255, 0, 0) - image 1 to image 4
    """
    h, w, _ = images[0].shape
    gap = 10  # 10 pixel gap between images
    
    # Create horizontal canvas with gaps
    canvas = np.zeros((h, w * 4 + gap * 3, 3), dtype=np.uint8)
    
    # Place images horizontally with gaps
    offsets = [0, w + gap, 2*w + 2*gap, 3*w + 3*gap]
    canvas[:, offsets[0]:offsets[0]+w] = images[0]
    canvas[:, offsets[1]:offsets[1]+w] = images[1]
    canvas[:, offsets[2]:offsets[2]+w] = images[2]
    canvas[:, offsets[3]:offsets[3]+w] = images[3]
    
    # Draw white vertical lines as separators
    separator_color = (255, 255, 255)
    for i in range(3):
        sep_x = offsets[i] + w + gap // 2
        cv2.line(canvas, (sep_x, 0), (sep_x, h-1), separator_color, 2)
    
    # Draw matches between image 1 and 2 (green)
    color_12 = (0, 255, 0)
    for kp1, kp2 in matches_12:
        pt1 = (int(kp1[0]) + offsets[0], int(kp1[1]))
        pt2 = (int(kp2[0]) + offsets[1], int(kp2[1]))
        cv2.line(canvas, pt1, pt2, color_12, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color_12, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color_12, -1, cv2.LINE_AA)
    
    # Draw matches between image 1 and 3 (red)
    color_13 = (0, 0, 255)
    for kp1, kp3 in matches_13:
        pt1 = (int(kp1[0]) + offsets[0], int(kp1[1]))
        pt3 = (int(kp3[0]) + offsets[2], int(kp3[1]))
        cv2.line(canvas, pt1, pt3, color_13, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color_13, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt3, 3, color_13, -1, cv2.LINE_AA)
    
    # Draw matches between image 1 and 4 (blue)
    color_14 = (255, 0, 0)
    for kp1, kp4 in matches_14:
        pt1 = (int(kp1[0]) + offsets[0], int(kp1[1]))
        pt4 = (int(kp4[0]) + offsets[3], int(kp4[1]))
        cv2.line(canvas, pt1, pt4, color_14, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color_14, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt4, 3, color_14, -1, cv2.LINE_AA)
    
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--timestamps', type=float, nargs=4, required=True, 
                        help='Four timestamps')
    parser.add_argument('--output', type=str, default='all_matches.png')
    parser.add_argument('--window-ms', type=int, default=200)
    
    args = parser.parse_args()
    
    K_event, event_resolution = load_intrinsics(args.config)
    
    t1, t2, t3, t4 = args.timestamps
    
    print(f"Loading events for timestamps: {t1:.6f}, {t2:.6f}, {t3:.6f}, {t4:.6f}")
    
    # Load all events
    raw_events = []
    for t in [t1, t2, t3, t4]:
        events = load_events_at_timestamp(args.h5_path, t, args.window_ms, event_resolution)
        raw_events.append(events)
    
    print("Creating reconstructed images...")
    images = [create_reconstructed_image(events, event_resolution) for events in raw_events]
    
    # Matches between image 1 and 2 (green)
    matches_12 = [
        ((169, 169), (173, 170)),
        ((463, 183), (466, 181)),
        ((49, 297), (54, 295)),
        ((333, 256), (334, 255)),
        ((300, 122), (302, 121)),
        ((296, 239), (302, 236)),
        ((323, 167), (327, 165)),
        ((410, 171), (415, 169)),
        ((200, 230), (201, 228)),
        ((194, 129), (196, 130))
    ]
    
    # Matches between image 1 and 3 (red) - formerly 2_3, now using image 1 points
    matches_13 = [
        ((174, 168), (178, 170)),
        ((155, 194), (162, 193)),
        ((56, 297), (61, 297)),
        ((299, 111), (300, 111)),
        ((324, 201), (328, 200)),
        ((301, 241), (304, 239)),
        ((10, 237), (21, 240)),
        ((466, 246), (475, 239)),
        ((344, 240), (347, 235)),
        ((196, 128), (203, 133))
    ]
    
    # Matches between image 1 and 4 (blue) - formerly 3_4, now using image 1 points
    matches_14 = [
        ((203, 133), (209, 135)),
        ((131, 141), (132, 141)),
        ((158, 194), (166, 196)),
        ((179, 167), (185, 168)),
        ((527, 229), (525, 226)),
        ((184, 233), (186, 238)),
        ((202, 273), (216, 274)),
        ((301, 113), (304, 115)),
        ((58, 299), (70, 299)),
        ((429, 94), (427, 89))
    ]
    
    print("Creating visualization...")
    viz = visualize_all_matches(images, matches_12, matches_13, matches_14)
    
    cv2.imwrite(args.output, viz)
    print(f"Saved visualization to {args.output}")

if __name__ == '__main__':
    main()