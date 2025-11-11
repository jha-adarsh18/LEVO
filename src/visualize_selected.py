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

def create_polarity_image(raw_events, resolution):
    width, height = resolution
    pos_accum = np.zeros((height, width), dtype=np.float32)
    neg_accum = np.zeros((height, width), dtype=np.float32)
    
    for event in raw_events:
        x, y, t, p = event
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            if p > 0:
                pos_accum[y, x] += 1
            else:
                neg_accum[y, x] += 1
    
    pos_accum = np.log1p(pos_accum)
    neg_accum = np.log1p(neg_accum)
    
    pos_max = np.percentile(pos_accum[pos_accum > 0], 95) if (pos_accum > 0).any() else 1
    neg_max = np.percentile(neg_accum[neg_accum > 0], 95) if (neg_accum > 0).any() else 1
    
    pos_norm = np.clip(pos_accum / pos_max * 255, 0, 255).astype(np.uint8)
    neg_norm = np.clip(neg_accum / neg_max * 255, 0, 255).astype(np.uint8)
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 2] = pos_norm
    img[:, :, 0] = neg_norm
    
    return img

def visualize_selected_matches(img1, img2, matches):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    
    color = (0, 255, 0)
    
    for kp1, kp2 in matches:
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]) + w1, int(kp2[1]))
        
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
    
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--timestamp1', type=float, required=True)
    parser.add_argument('--timestamp2', type=float, required=True)
    parser.add_argument('--output', type=str, default='selected_matches.png')
    parser.add_argument('--window-ms', type=int, default=200)
    
    args = parser.parse_args()
    
    K_event, event_resolution = load_intrinsics(args.config)
    
    t1 = args.timestamp1
    t2 = args.timestamp2
    
    print(f"Loading events for timestamps: {t1:.6f}, {t2:.6f}")
    
    raw_events1 = load_events_at_timestamp(args.h5_path, t1, args.window_ms, event_resolution)
    raw_events2 = load_events_at_timestamp(args.h5_path, t2, args.window_ms, event_resolution)
    
    print("Creating polarity images...")
    img1 = create_polarity_image(raw_events1, event_resolution)
    img2 = create_polarity_image(raw_events2, event_resolution)
    
    selected_matches = [
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
    
    viz = visualize_selected_matches(img1, img2, selected_matches)
    
    cv2.imwrite(args.output, viz)
    print(f"Saved visualization to {args.output}")

if __name__ == '__main__':
    main()