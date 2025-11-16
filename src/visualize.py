import hdf5plugin
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import yaml
import argparse
from model import EventVO

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

def load_events_at_timestamp(h5_path, timestamp, window_ms, resolution, n_events=2048):
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
    
    events_x_norm = events_x / float(width)
    events_y_norm = events_y / float(height)
    events = np.stack([events_x_norm, events_y_norm, events_t, events_p], axis=1).astype(np.float32)
    
    n = len(events)
    if n == 0:
        return np.zeros((n_events, 4), dtype=np.float32), np.zeros(n_events, dtype=np.float32), raw_events
    
    t_min, t_max = events[:, 2].min(), events[:, 2].max()
    if t_max > t_min:
        events[:, 2] = (events[:, 2] - t_min) / (t_max - t_min)
    else:
        events[:, 2] = 0.5
    
    if n > n_events:
        indices = np.random.choice(n, n_events, replace=False)
        indices = np.sort(indices)
        sampled = events[indices]
        mask = np.ones(n_events, dtype=np.float32)
    else:
        sampled = np.zeros((n_events, 4), dtype=np.float32)
        sampled[:n] = events
        mask = np.zeros(n_events, dtype=np.float32)
        mask[:n] = 1.0
    
    return sampled, mask, raw_events

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

def visualize_matches(img1, img2, kp1, kp2, matches, resolution, top_n=100):
    width, height = resolution
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    
    match_scores = matches.max(dim=1)[0]
    match_indices = matches.argmax(dim=1)
    
    top_k = min(top_n, len(match_scores))
    top_scores, top_idx = torch.topk(match_scores, top_k)
    
    kp1_pixels = kp1.copy()
    kp1_pixels[:, 0] *= width
    kp1_pixels[:, 1] *= height
    
    kp2_pixels = kp2.copy()
    kp2_pixels[:, 0] *= width
    kp2_pixels[:, 1] *= height
    
    match_data = []
    
    for idx in top_idx:
        i = idx.item()
        j = match_indices[i].item()
        
        pt1 = (int(kp1_pixels[i, 0]), int(kp1_pixels[i, 1]))
        pt2 = (int(kp2_pixels[j, 0]) + w1, int(kp2_pixels[j, 1]))
        pt2_actual = (int(kp2_pixels[j, 0]), int(kp2_pixels[j, 1]))
        
        match_data.append({
            'kp1': pt1,
            'kp2': pt2_actual,
            'score': match_scores[i].item()
        })
        
        if 0 <= pt1[0] < w1 and 0 <= pt1[1] < h1 and 0 <= pt2[0] - w1 < w2 and 0 <= pt2[1] < h2:
            color = (0, 255, 0)
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
    
    return canvas, match_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--h5-path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--timestamp1', type=float, required=True)
    parser.add_argument('--timestamp2', type=float, required=True)
    parser.add_argument('--output', type=str, default='event_matches_viz.png')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--window-ms', type=int, default=200)
    parser.add_argument('--n-events', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=500)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    K_event, event_resolution = load_intrinsics(args.config)
    
    t1 = args.timestamp1
    t2 = args.timestamp2
    
    print(f"Loading events for timestamps: {t1:.6f}, {t2:.6f}")
    
    events1, mask1, raw_events1 = load_events_at_timestamp(
        args.h5_path, t1, args.window_ms, event_resolution, args.n_events
    )
    events2, mask2, raw_events2 = load_events_at_timestamp(
        args.h5_path, t2, args.window_ms, event_resolution, args.n_events
    )
    
    print(f"Loaded {int(mask1.sum())} events for frame 1")
    print(f"Loaded {int(mask2.sum())} events for frame 2")
    
    print("Creating polarity images...")
    img1 = create_polarity_image(raw_events1, event_resolution)
    img2 = create_polarity_image(raw_events2, event_resolution)
    
    model = EventVO(d_model=args.d_model, num_samples=args.num_samples).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Running inference...")
    with torch.no_grad():
        events1_t = torch.from_numpy(events1).unsqueeze(0).to(device)
        mask1_t = torch.from_numpy(mask1).unsqueeze(0).to(device)
        events2_t = torch.from_numpy(events2).unsqueeze(0).to(device)
        mask2_t = torch.from_numpy(mask2).unsqueeze(0).to(device)
        K_t = torch.from_numpy(K_event).unsqueeze(0).to(device)
        
        predictions = model(events1_t, mask1_t, events2_t, mask2_t, K_t)
    
    kp1_event = predictions['keypoints1'][0].cpu().numpy()
    kp2_event = predictions['keypoints2'][0].cpu().numpy()
    matches = predictions['matches'][0].cpu()
    
    print(f"Detected {len(kp1_event)} keypoints in frame 1")
    print(f"Detected {len(kp2_event)} keypoints in frame 2")
    
    viz, match_data = visualize_matches(img1, img2, kp1_event, kp2_event, matches, event_resolution, args.top_n)
    
    cv2.imwrite(args.output, viz)
    
    match_file = args.output.replace('.png', '_matches.txt')
    with open(match_file, 'w') as f:
        f.write("# kp1_x kp1_y kp2_x kp2_y score\n")
        for m in match_data:
            f.write(f"{m['kp1'][0]} {m['kp1'][1]} {m['kp2'][0]} {m['kp2'][1]} {m['score']:.6f}\n")
    
    print(f"Saved visualization to {args.output}")
    print(f"Saved match data to {match_file}")

if __name__ == '__main__':
    main()