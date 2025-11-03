import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import pickle


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
    else:
        raise ValueError(f"Unknown format: {path.suffix}")
    
    return traj


def load_ground_truth(path):
    poses = np.loadtxt(path)
    timestamps = poses[:, 0]
    
    trajectory = []
    for i in range(len(poses)):
        p = poses[i, 1:4]
        q = poses[i, 4:8]
        
        R = quat_to_rot(q)
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = p
        trajectory.append(pose)
    
    return np.array(trajectory), timestamps


def quat_to_rot(q):
    w, x, y, z = q
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])
    return R


def plot_trajectory_3d(pred_traj, gt_traj=None, loop_edges=None, save_path=None, title=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pred_pos = pred_traj[:, :3, 3]
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
            'b-', linewidth=2, label='Predicted')
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], 
               c='g', s=100, marker='o', label='Start')
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], 
               c='r', s=100, marker='x', label='End')
    
    if gt_traj is not None:
        gt_pos = gt_traj[:, :3, 3]
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                'k--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    
    if loop_edges is not None and len(loop_edges) > 0:
        for kf_i, kf_j, _, _ in loop_edges:
            if kf_i < len(pred_traj) and kf_j < len(pred_traj):
                pos_i = pred_traj[kf_i, :3, 3]
                pos_j = pred_traj[kf_j, :3, 3]
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]], 
                       'r--', linewidth=1, alpha=0.6)
        ax.plot([], [], 'r--', linewidth=1, alpha=0.6, label=f'Loop Closures ({len(loop_edges)})')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trajectory_2d(pred_traj, gt_traj=None, loop_edges=None, save_path=None, title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    pred_pos = pred_traj[:, :3, 3]
    
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], 'b-', linewidth=2, label='Predicted')
    ax1.scatter(pred_pos[0, 0], pred_pos[0, 1], c='g', s=100, marker='o', label='Start')
    ax1.scatter(pred_pos[-1, 0], pred_pos[-1, 1], c='r', s=100, marker='x', label='End')
    if gt_traj is not None:
        gt_pos = gt_traj[:, :3, 3]
        ax1.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    if loop_edges is not None and len(loop_edges) > 0:
        for kf_i, kf_j, _, _ in loop_edges:
            if kf_i < len(pred_traj) and kf_j < len(pred_traj):
                pos_i = pred_traj[kf_i, :3, 3]
                pos_j = pred_traj[kf_j, :3, 3]
                ax1.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                        'r--', linewidth=1, alpha=0.5)
        ax1.plot([], [], 'r--', linewidth=1, alpha=0.5, label=f'Loops ({len(loop_edges)})')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Top View (XY)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    ax2.plot(pred_pos[:, 0], pred_pos[:, 2], 'b-', linewidth=2, label='Predicted')
    ax2.scatter(pred_pos[0, 0], pred_pos[0, 2], c='g', s=100, marker='o', label='Start')
    ax2.scatter(pred_pos[-1, 0], pred_pos[-1, 2], c='r', s=100, marker='x', label='End')
    if gt_traj is not None:
        ax2.plot(gt_pos[:, 0], gt_pos[:, 2], 'k--', linewidth=1.5, alpha=0.7, label='Ground Truth')
    if loop_edges is not None and len(loop_edges) > 0:
        for kf_i, kf_j, _, _ in loop_edges:
            if kf_i < len(pred_traj) and kf_j < len(pred_traj):
                pos_i = pred_traj[kf_i, :3, 3]
                pos_j = pred_traj[kf_j, :3, 3]
                ax2.plot([pos_i[0], pos_j[0]], [pos_i[2], pos_j[2]], 
                        'r--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.set_title('Side View (XZ)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_errors(errors, error_type='ATE', save_path=None, title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    frames = np.arange(len(errors))
    
    ax1.plot(frames, errors, 'b-', linewidth=1.5)
    ax1.fill_between(frames, errors, alpha=0.3)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel(f'{error_type} (m)', fontsize=12)
    ax1.set_title(f'{error_type} over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}m')
    ax2.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}m')
    ax2.set_xlabel(f'{error_type} (m)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{error_type} Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rotation_translation_errors(rot_errors, trans_errors, save_path=None, title=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    frames = np.arange(len(rot_errors))
    
    ax1.plot(frames, rot_errors, 'b-', linewidth=1.5, label='Rotation Error')
    ax1.fill_between(frames, rot_errors, alpha=0.3)
    ax1.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax1.set_title(f'Rotation Error - Mean: {np.mean(rot_errors):.2f}°, Median: {np.median(rot_errors):.2f}°', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(frames, trans_errors, 'r-', linewidth=1.5, label='Translation Error')
    ax2.fill_between(frames, trans_errors, alpha=0.3, color='red')
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Translation Error (m)', fontsize=12)
    ax2.set_title(f'Translation Error - Mean: {np.mean(trans_errors):.3f}m, Median: {np.median(trans_errors):.3f}m', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_point_cloud(points, save_path=None, title=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', s=1, alpha=0.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Sparse Point Cloud ({len(points)} points)', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sequence', type=str, default='')
    parser.add_argument('--point-cloud', type=str, default=None)
    parser.add_argument('--errors', type=str, default=None)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predicted trajectory from {args.pred}")
    pred_path = Path(args.pred)
    
    if pred_path.suffix == '.pkl':
        with open(pred_path, 'rb') as f:
            result = pickle.load(f)
            pred_traj = result['trajectory']
            loop_edges = result.get('loop_edges', None)
    else:
        pred_traj = load_trajectory(pred_path)
        loop_edges = None
    
    gt_traj = None
    if args.gt:
        print(f"Loading ground truth from {args.gt}")
        gt_traj, _ = load_ground_truth(Path(args.gt))
    
    title = f"Trajectory - {args.sequence}" if args.sequence else "Trajectory"
    
    print("Generating 3D trajectory plot...")
    plot_trajectory_3d(pred_traj, gt_traj, loop_edges,
                      output_dir / 'trajectory_3d.png', title)
    
    print("Generating 2D trajectory plots...")
    plot_trajectory_2d(pred_traj, gt_traj, loop_edges,
                      output_dir / 'trajectory_2d.png', title)
    
    if args.point_cloud:
        print(f"Loading point cloud from {args.point_cloud}")
        points = np.loadtxt(args.point_cloud)
        print("Generating point cloud plot...")
        plot_point_cloud(points, output_dir / 'point_cloud.png', 
                        f"Point Cloud - {args.sequence}" if args.sequence else None)
    
    if args.errors:
        print(f"Loading errors from {args.errors}")
        with open(args.errors, 'rb') as f:
            error_data = pickle.load(f)
        
        if 'ate' in error_data:
            plot_errors(error_data['ate'], 'ATE', 
                       output_dir / 'ate_errors.png', f"ATE - {args.sequence}")
        
        if 'rot_errors' in error_data and 'trans_errors' in error_data:
            plot_rotation_translation_errors(
                error_data['rot_errors'], error_data['trans_errors'],
                output_dir / 'rot_trans_errors.png', f"Errors - {args.sequence}"
            )
    
    if loop_edges and len(loop_edges) > 0:
        print(f"Visualized {len(loop_edges)} loop closures")
    
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()