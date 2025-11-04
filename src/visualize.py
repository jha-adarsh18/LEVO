import numpy as np
import argparse
from pathlib import Path
import pickle
import viser
import time


def load_trajectory(path):
    if path.suffix == '.txt':
        traj = np.loadtxt(path)
        if traj.shape[1] == 16:
            traj = traj.reshape(-1, 4, 4)
        elif traj.shape[1] == 12:
            traj_4x4 = np.zeros((len(traj), 4, 4))
            traj_4x4[:, :3, :] = traj.reshape(-1, 3, 4)
            traj_4x4[:, 3, 3] = 1
            traj = traj_4x4
    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
            traj = data['trajectory']
    return traj


def load_ground_truth(path):
    poses = np.loadtxt(path)
    timestamps = poses[:, 0]
    trajectory = []
    for i in range(len(poses)):
        p = poses[i, 1:4]
        q = poses[i, [7, 4, 5, 6]]
        R = quat_to_rot(q)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = p
        trajectory.append(pose)
    return np.array(trajectory), timestamps


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])


def visualize_trajectory(server, pred_traj, gt_traj=None, loop_edges=None):
    pred_pos = pred_traj[:, :3, 3].astype(np.float32)
    
    server.scene.add_spline_catmull_rom(
        "/pred_trajectory",
        pred_pos,
        tension=0.5,
        line_width=3.0,
        color=(0, 100, 255)
    )
    
    server.scene.add_icosphere(
        "/pred_start",
        radius=0.15,
        color=(0, 255, 0),
        position=pred_pos[0]
    )
    
    server.scene.add_icosphere(
        "/pred_end",
        radius=0.15,
        color=(255, 0, 0),
        position=pred_pos[-1]
    )
    
    if gt_traj is not None:
        gt_pos = gt_traj[:, :3, 3].astype(np.float32)
        server.scene.add_spline_catmull_rom(
            "/gt_trajectory",
            gt_pos,
            tension=0.5,
            line_width=2.0,
            color=(50, 50, 50)
        )
    
    if loop_edges is not None and len(loop_edges) > 0:
        for idx, (kf_i, kf_j, _, _) in enumerate(loop_edges):
            if kf_i < len(pred_traj) and kf_j < len(pred_traj):
                pos_i = pred_traj[kf_i, :3, 3].astype(np.float32)
                pos_j = pred_traj[kf_j, :3, 3].astype(np.float32)
                points = np.stack([pos_i, pos_j])
                server.scene.add_spline_catmull_rom(
                    f"/loop_{idx}",
                    points,
                    tension=0.0,
                    line_width=1.5,
                    color=(255, 50, 50)
                )


def visualize_point_cloud(server, points):
    points = points.astype(np.float32)
    colors = np.zeros_like(points, dtype=np.uint8)
    z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
    colors[:, 0] = (z_norm * 255).astype(np.uint8)
    colors[:, 1] = ((1 - z_norm) * 255).astype(np.uint8)
    colors[:, 2] = 100
    
    server.scene.add_point_cloud(
        "/point_cloud",
        points=points,
        colors=colors,
        point_size=0.02
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--results-root', type=str, default='results')
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    
    seq_dir = Path(args.results_root) / args.sequence
    if not seq_dir.exists():
        print(f"Error: {seq_dir} does not exist")
        return
    
    result_pkl = seq_dir / 'result.pkl'
    point_cloud_txt = seq_dir / 'point_cloud.txt'
    
    print(f"Loading results from {seq_dir}")
    with open(result_pkl, 'rb') as f:
        result = pickle.load(f)
        pred_traj = result['trajectory']
        loop_edges = result.get('loop_edges', None)
    
    gt_traj = None
    if args.gt:
        print(f"Loading ground truth from {args.gt}")
        gt_traj, _ = load_ground_truth(Path(args.gt))
    
    server = viser.ViserServer(port=args.port, verbose=False)
    print(f"Viser server running at http://localhost:{args.port}")
    if args.share:
        print("Share link enabled")
    
    visualize_trajectory(server, pred_traj, gt_traj, loop_edges)
    
    if point_cloud_txt.exists():
        print(f"Loading point cloud from {point_cloud_txt}")
        points = np.loadtxt(point_cloud_txt)
        print(f"Visualizing {len(points)} points")
        visualize_point_cloud(server, points)
    
    if loop_edges:
        print(f"Visualized {len(loop_edges)} loop closures")
    
    print("Visualization ready. Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()