import torch
import numpy as np
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import pickle

from model import EventVO
from dataset import EventVODataset
from backend import VOBackend
from stereo import StereoProcessor
from geometry import ransac_pnp, ransac_essential, refine_pose_gauss_newton, MotionModel
from loop_closure import LoopClosureDetector


class VOInference:
    def __init__(self, model, stereo_processor, backend, loop_detector=None, use_stereo=True, use_backend=True, use_loop_closure=True):
        self.model = model
        self.stereo_processor = stereo_processor
        self.backend = backend
        self.use_stereo = use_stereo
        self.use_backend = use_backend
        self.loop_detector = loop_detector
        self.use_loop_closure = use_loop_closure
        self.frame_count = 0
        self.motion_model = MotionModel(alpha=0.7)

        self.current_pose = np.eye(4, dtype=np.float64)
        self.trajectory = [self.current_pose.copy()]
        self.timestamps = []

    @torch.no_grad()
    def process_frame_pair(self, events1_l, mask1_l, events2_l, mask2_l,
                           events1_r=None, mask1_r=None, events2_r=None, mask2_r=None,
                           K=None, timestamp=None):

        pred_left = self.model(events1_l, mask1_l, events2_l, mask2_l)

        R_pred = pred_left['R_pred'][0].cpu().numpy()
        t_pred = pred_left['t_pred'][0].cpu().numpy()

        if self.use_stereo and events1_r is not None:
            pred_right = self.model(events1_r, mask1_r, events2_r, mask2_r)

            scale = self.stereo_processor.recover_scale(pred_left, pred_left, pred_right, pred_right)
            t_pred = t_pred * scale[0].cpu().numpy()

            points_3d, valid_mask = self.stereo_processor.process_stereo_pair(pred_left, pred_right)
            points_3d = np.ascontiguousarray(points_3d[0].cpu().numpy())
            valid_mask = valid_mask[0].cpu().numpy()
        else:
            points_3d = None
            valid_mask = None

        if points_3d is not None and valid_mask.sum() > 10 and K is not None:
            kp_1 = np.ascontiguousarray(pred_left['keypoints1'][0].cpu().numpy())
            points_3d_valid = points_3d[valid_mask]
            kp_1_valid = kp_1[valid_mask]

            R_ransac, t_ransac, inliers = ransac_pnp(points_3d_valid, kp_1_valid, K)

            if R_ransac is not None and inliers.sum() > 10:
                R_refined, t_refined = refine_pose_gauss_newton(
                    R_ransac, t_ransac,
                    points_3d_valid[inliers],
                    kp_1_valid[inliers],
                    K, max_iter=5
                )
                R_pred = R_refined
                t_pred = t_refined

        rel_pose = np.eye(4, dtype=np.float64)
        rel_pose[:3, :3] = R_pred
        rel_pose[:3, 3] = t_pred

        self.current_pose = self.current_pose @ rel_pose

        if self.use_backend and points_3d is not None and valid_mask is not None and valid_mask.sum() > 0:
            if self.backend.should_insert_keyframe(self.current_pose):
                kp_1 = np.ascontiguousarray(pred_left['keypoints1'][0].cpu().numpy())
                descriptors = np.ascontiguousarray(pred_left['descriptors1'][0].cpu().numpy())
                
                n_valid = valid_mask.sum()
                
                self.backend.insert_keyframe(
                    len(self.trajectory),
                    self.current_pose.copy(),
                    points_3d[valid_mask],
                    descriptors[:n_valid],
                    kp_1[:n_valid]
                )

                self.backend.local_bundle_adjustment(K)

                if len(self.backend.keyframes) % self.backend.pgo_every_n_frames == 0:
                    self.backend.pose_graph_optimization()

                if len(self.backend.keyframes) > 0:
                    self.current_pose = self.backend.keyframes[-1].pose.copy()

                if self.use_loop_closure and self.loop_detector is not None:
                    if len(self.backend.keyframes) > 1:
                        current_kf = self.backend.keyframes[-1]
                        self.loop_detector.add_keyframe(current_kf)

                        loop_kf_id, loop_rel_pose = self.loop_detector.detect_loop(current_kf, K)

                        if loop_kf_id is not None:
                            print(f"\n🔄 Loop closure detected: {loop_kf_id} -> {current_kf.frame_id}")
                            self.backend.add_loop_closure(loop_kf_id, current_kf.frame_id, loop_rel_pose)
                            self.backend.pose_graph_optimization()
                            self.current_pose = self.backend.keyframes[-1].pose.copy()

                self.frame_count += 1

        self.trajectory.append(self.current_pose.copy())
        if timestamp is not None:
            self.timestamps.append(timestamp)

        return self.current_pose.copy()

    def get_trajectory(self):
        return np.array(self.trajectory)

    def get_timestamps(self):
        return np.array(self.timestamps)


def load_model(checkpoint_path, device, d_model=256, num_samples=500):
    model = EventVO(d_model=d_model, num_samples=num_samples).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_stereo_config(config_path, sequence_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    stereo_config = config.get('stereo', {})

    seq_lower = sequence_name.lower()
    if 'indoor_flying' in seq_lower:
        calib = stereo_config.get('indoor_flying', {})
    elif 'outdoor_night' in seq_lower:
        calib = stereo_config.get('outdoor_night', {})
    elif 'outdoor_day' in seq_lower:
        calib = stereo_config.get('outdoor_day', {})
    else:
        calib = stereo_config.get('default', {})

    K_left = np.ascontiguousarray(calib.get('K_left', np.eye(3)), dtype=np.float32)
    K_right = np.ascontiguousarray(calib.get('K_right', np.eye(3)), dtype=np.float32)
    baseline = calib.get('baseline', 0.1)

    return K_left, K_right, baseline


def run_sequence(model, dataset_left, dataset_right, sequence_indices, stereo_processor, backend, loop_detector,
                 device, use_stereo=True, use_backend=True, use_loop_closure=True):

    vo = VOInference(model, stereo_processor, backend, loop_detector, use_stereo, use_backend, use_loop_closure)

    K = np.ascontiguousarray(dataset_left.sequences[dataset_left.pairs[sequence_indices[0]][0]]['K'])

    for idx in tqdm(sequence_indices, desc="Processing frames"):
        seq_name, i, j = dataset_left.pairs[idx]
        seq = dataset_left.sequences[seq_name]

        sample_left = dataset_left[idx]

        events1_l = sample_left['events1'].unsqueeze(0).to(device)
        mask1_l = sample_left['mask1'].unsqueeze(0).to(device)
        events2_l = sample_left['events2'].unsqueeze(0).to(device)
        mask2_l = sample_left['mask2'].unsqueeze(0).to(device)

        events1_r = None
        mask1_r = None
        events2_r = None
        mask2_r = None

        if use_stereo and dataset_right is not None:
            sample_right = dataset_right[idx]
            events1_r = sample_right['events1'].unsqueeze(0).to(device)
            mask1_r = sample_right['mask1'].unsqueeze(0).to(device)
            events2_r = sample_right['events2'].unsqueeze(0).to(device)
            mask2_r = sample_right['mask2'].unsqueeze(0).to(device)

        timestamp = seq['timestamps'][j]

        pose = vo.process_frame_pair(
            events1_l, mask1_l, events2_l, mask2_l,
            events1_r, mask1_r, events2_r, mask2_r,
            K, timestamp
        )

    return vo.get_trajectory(), vo.get_timestamps()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--intrinsics-config', type=str, default='configs/config.yaml')
    parser.add_argument('--stereo-config', type=str, default='configs/stereo_config.yaml')
    parser.add_argument('--backend-config', type=str, default='configs/backend_config.yaml')
    parser.add_argument('--loop-config', type=str, default='configs/loop_config.yaml')
    parser.add_argument('--camera', type=str, default='left')
    parser.add_argument('--dt-range', type=int, nargs=2, default=[50, 200])
    parser.add_argument('--n-events', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--use-stereo', action='store_true')
    parser.add_argument('--use-backend', action='store_true')
    parser.add_argument('--use-loop-closure', action='store_true')
    parser.add_argument('--no-stereo', dest='use_stereo', action='store_false')
    parser.add_argument('--no-backend', dest='use_backend', action='store_false')
    parser.add_argument('--no-loop-closure', dest='use_loop_closure', action='store_false')
    parser.set_defaults(use_stereo=True, use_backend=True, use_loop_closure=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model}")
    model = load_model(args.model, device, args.d_model, args.num_samples)

    print("Loading left camera dataset...")
    dataset_left = EventVODataset(
        args.data_root,
        camera='left',
        dt_range=tuple(args.dt_range),
        n_events=args.n_events,
        augment=False,
        intrinsics_config=args.intrinsics_config
    )

    dataset_right = None
    if args.use_stereo:
        print("Loading right camera dataset...")
        dataset_right = EventVODataset(
            args.data_root,
            camera='right',
            dt_range=tuple(args.dt_range),
            n_events=args.n_events,
            augment=False,
            intrinsics_config=args.intrinsics_config
        )

    sequence_indices = [i for i, (seq, _, _) in enumerate(dataset_left.pairs) if seq == args.sequence]

    if len(sequence_indices) == 0:
        print(f"No frames found for sequence {args.sequence}")
        return

    print(f"Found {len(sequence_indices)} frame pairs for {args.sequence}")

    K_left, K_right, baseline = load_stereo_config(args.stereo_config, args.sequence)
    stereo_processor = StereoProcessor(K_left, K_right, baseline)

    with open(args.backend_config, 'r') as f:
        backend_config = yaml.safe_load(f)
    backend = VOBackend(backend_config)

    loop_detector = None
    if args.use_loop_closure:
        with open(args.loop_config, 'r') as f:
            loop_config = yaml.safe_load(f)
        loop_detector = LoopClosureDetector(loop_config)

    print("Running inference...")
    trajectory, timestamps = run_sequence(
        model, dataset_left, dataset_right, sequence_indices, stereo_processor, backend, loop_detector,
        device, args.use_stereo, args.use_backend, args.use_loop_closure
    )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_path / 'trajectory.txt', trajectory.reshape(-1, 16))
    if len(timestamps) > 0:
        np.savetxt(output_path / 'timestamps.txt', timestamps)

    result = {
        'trajectory': trajectory,
        'timestamps': timestamps,
        'sequence': args.sequence,
        'use_stereo': args.use_stereo,
        'use_backend': args.use_backend,
        'use_loop_closure': args.use_loop_closure,
        'loop_edges': backend.loop_edges if args.use_loop_closure else []
    }

    with open(output_path / 'result.pkl', 'wb') as f:
        pickle.dump(result, f)

    if args.use_backend:
        point_cloud = backend.get_point_cloud()
        np.savetxt(output_path / 'point_cloud.txt', point_cloud)
        print(f"Saved {len(point_cloud)} map points")

    if args.use_loop_closure and len(backend.loop_edges) > 0:
        print(f"Detected {len(backend.loop_edges)} loop closures")

    print(f"Results saved to {output_path}")
    print(f"Trajectory length: {len(trajectory)} poses")


if __name__ == '__main__':
    main()