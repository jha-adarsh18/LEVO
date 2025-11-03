import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import os

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Keyframe:
    def __init__(self, frame_id, pose, points_3d, descriptors, keypoints_2d):
        self.frame_id = frame_id
        self.pose = pose
        self.points_3d = points_3d
        self.descriptors = descriptors
        self.keypoints_2d = keypoints_2d
        self.point_ids = []

class MapPoint:
    def __init__(self, point_id, position, descriptor):
        self.point_id = point_id
        self.position = position
        self.descriptor = descriptor
        self.observations = []
        self.outlier = False

class VOBackend:
    def __init__(self, config=None):
        config = config or {}
        self.min_translation = config.get('min_translation', 0.15)
        self.min_rotation = config.get('min_rotation', 8.0)
        self.max_keyframes = config.get('max_keyframes', 40)
        self.ba_window_size = config.get('ba_window_size', 5)
        self.ba_iterations = config.get('ba_iterations', 5)
        self.min_observations = config.get('min_observations', 2)
        self.max_reprojection_error = config.get('max_reprojection_error', 4.0)
        self.pgo_every_n_frames = config.get('pgo_every_n_frames', 50)
        self.ba_every_n_frames = config.get('ba_every_n_frames', 5)
        self.keyframes = []
        self.map_points = {}
        self.next_point_id = 0
        self.pose_graph_edges = []
        self.loop_edges = []
        self.frame_count = 0
        
    def should_insert_keyframe(self, current_pose):
        if len(self.keyframes) == 0:
            return True
        last_pose = self.keyframes[-1].pose
        rel_pose = np.linalg.inv(last_pose) @ current_pose
        translation = np.linalg.norm(rel_pose[:3, 3])
        rotation_angle = np.linalg.norm(R.from_matrix(rel_pose[:3, :3]).as_rotvec()) * 180 / np.pi
        return translation > self.min_translation or rotation_angle > self.min_rotation
    
    def insert_keyframe(self, frame_id, pose, points_3d, descriptors, keypoints_2d):
        points_3d = np.ascontiguousarray(points_3d)
        descriptors = np.ascontiguousarray(descriptors)
        keypoints_2d = np.ascontiguousarray(keypoints_2d)
        kf = Keyframe(frame_id, pose.copy(), points_3d, descriptors, keypoints_2d)
        if len(self.keyframes) > 0:
            kf.point_ids = self._associate_points_vectorized(kf)
        else:
            kf.point_ids = self._create_new_points(kf)
        self.keyframes.append(kf)
        if len(self.keyframes) > 1:
            last_kf = self.keyframes[-2]
            rel_pose = np.linalg.inv(last_kf.pose) @ kf.pose
            self.pose_graph_edges.append((last_kf.frame_id, kf.frame_id, rel_pose, np.eye(6)))
        if len(self.keyframes) > self.max_keyframes:
            self._cull_keyframes()
        return kf
    
    def _associate_points_vectorized(self, kf):
        point_ids = []
        recent_kfs = self.keyframes[-3:]
        if len(recent_kfs) == 0:
            return self._create_new_points(kf)
        all_ref_descs = []
        all_ref_ids = []
        for ref_kf in recent_kfs:
            all_ref_descs.append(ref_kf.descriptors)
            all_ref_ids.extend(ref_kf.point_ids)
        ref_descs = np.vstack(all_ref_descs)
        ref_descs_gpu = torch.from_numpy(ref_descs).to(device)
        chunk_size = 128
        n_desc = len(kf.descriptors)
        valid_map_points = set(self.map_points.keys())
        for start_idx in range(0, n_desc, chunk_size):
            end_idx = min(start_idx + chunk_size, n_desc)
            desc_chunk = kf.descriptors[start_idx:end_idx]
            desc_chunk_gpu = torch.from_numpy(desc_chunk).to(device)
            scores = (desc_chunk_gpu @ ref_descs_gpu.T).cpu().numpy()
            best_indices = np.argmax(scores, axis=1)
            best_scores = np.max(scores, axis=1)
            for i, (best_idx, score) in enumerate(zip(best_indices, best_scores)):
                actual_i = start_idx + i
                if score > 0.85:
                    best_match_id = all_ref_ids[best_idx]
                    if best_match_id in valid_map_points:
                        point_ids.append(best_match_id)
                        self.map_points[best_match_id].observations.append((kf.frame_id, actual_i))
                        continue
                new_id = self.next_point_id
                self.next_point_id += 1
                self.map_points[new_id] = MapPoint(new_id, kf.points_3d[actual_i], kf.descriptors[actual_i])
                self.map_points[new_id].observations.append((kf.frame_id, actual_i))
                point_ids.append(new_id)
        return point_ids
    
    def _create_new_points(self, kf):
        point_ids = []
        for i, (pt, desc) in enumerate(zip(kf.points_3d, kf.descriptors)):
            point_id = self.next_point_id
            self.next_point_id += 1
            self.map_points[point_id] = MapPoint(point_id, pt.copy(), desc.copy())
            self.map_points[point_id].observations.append((kf.frame_id, i))
            point_ids.append(point_id)
        return point_ids
    
    def local_bundle_adjustment(self, K):
        self.frame_count += 1
        if self.frame_count % self.ba_every_n_frames != 0:
            return
        if len(self.keyframes) < 2:
            return
        window_kfs = self.keyframes[-self.ba_window_size:]
        active_points = set()
        for kf in window_kfs:
            active_points.update(kf.point_ids)
        active_points = [pid for pid in active_points 
                        if pid in self.map_points and len(self.map_points[pid].observations) >= self.min_observations]
        if len(active_points) < 10:
            return
        x0 = self._pack_parameters(window_kfs, active_points)
        result = least_squares(
            self._reprojection_residuals_vectorized,
            x0,
            args=(window_kfs, active_points, K),
            max_nfev=self.ba_iterations,
            verbose=0,
            ftol=1e-3,
            xtol=1e-3
        )
        self._unpack_parameters(result.x, window_kfs, active_points)
        self._cull_outlier_points_vectorized(K)
    
    def _pack_parameters(self, kfs, point_ids):
        n_kfs = len(kfs)
        n_points = len(point_ids)
        params = np.empty(n_kfs * 6 + n_points * 3, dtype=np.float64)
        idx = 0
        for kf in kfs:
            rvec = R.from_matrix(kf.pose[:3, :3]).as_rotvec()
            params[idx:idx+3] = rvec
            params[idx+3:idx+6] = kf.pose[:3, 3]
            idx += 6
        for pid in point_ids:
            params[idx:idx+3] = self.map_points[pid].position
            idx += 3
        return params
    
    def _unpack_parameters(self, params, kfs, point_ids):
        idx = 0
        for kf in kfs:
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            kf.pose[:3, :3] = R.from_rotvec(rvec).as_matrix()
            kf.pose[:3, 3] = tvec
            idx += 6
        for pid in point_ids:
            self.map_points[pid].position = params[idx:idx+3].copy()
            idx += 3
    
    def _reprojection_residuals_vectorized(self, params, kfs, point_ids, K):
        self._unpack_parameters(params, kfs, point_ids)
        all_pts_3d = []
        all_pts_2d = []
        all_poses = []
        point_ids_set = set(point_ids)
        for kf in kfs:
            kf_point_indices = [i for i, pid in enumerate(kf.point_ids) if pid in point_ids_set]
            if not kf_point_indices:
                continue
            n_pts = len(kf_point_indices)
            pts_3d = np.empty((n_pts, 3), dtype=np.float64)
            for idx, i in enumerate(kf_point_indices):
                pts_3d[idx] = self.map_points[kf.point_ids[i]].position
            pts_2d = kf.keypoints_2d[kf_point_indices]
            all_pts_3d.append(pts_3d)
            all_pts_2d.append(pts_2d)
            all_poses.extend([kf.pose] * len(pts_3d))
        if not all_pts_3d:
            return np.array([])
        all_pts_3d = np.vstack(all_pts_3d)
        all_pts_2d = np.vstack(all_pts_2d)
        all_poses = np.array(all_poses)
        all_pts_3d_gpu = torch.from_numpy(np.ascontiguousarray(all_pts_3d)).to(device)
        all_pts_2d_gpu = torch.from_numpy(np.ascontiguousarray(all_pts_2d)).to(device)
        R_mats_gpu = torch.from_numpy(all_poses[:, :3, :3]).to(device)
        t_vecs_gpu = torch.from_numpy(all_poses[:, :3, 3]).to(device)
        pts_cam_gpu = torch.einsum('nij,nj->ni', R_mats_gpu.transpose(1, 2), all_pts_3d_gpu - t_vecs_gpu)
        valid = (pts_cam_gpu[:, 2] > 0).cpu().numpy()
        n_total = len(all_pts_3d)
        residuals = np.zeros(n_total * 2, dtype=np.float64)
        if not np.any(valid):
            residuals.fill(10.0)
            return residuals
        K_gpu = torch.from_numpy(K).to(device)
        pts_proj_gpu = (K_gpu @ pts_cam_gpu[valid].T).T
        pts_proj_gpu = pts_proj_gpu[:, :2] / pts_proj_gpu[:, 2:3]
        diff = (pts_proj_gpu - all_pts_2d_gpu[valid]).cpu().numpy()
        valid_indices = np.where(valid)[0]
        n_valid = len(valid_indices)
        idx = np.empty(n_valid * 2, dtype=np.int32)
        idx[0::2] = valid_indices * 2
        idx[1::2] = valid_indices * 2 + 1
        residuals[idx] = diff.ravel()
        return residuals
    
    def _cull_outlier_points_vectorized(self, K):
        K_gpu = torch.from_numpy(K).to(device)
        for kf in self.keyframes:
            valid_indices = [i for i, pid in enumerate(kf.point_ids) 
                           if pid in self.map_points and not self.map_points[pid].outlier]
            if not valid_indices:
                continue
            n_valid = len(valid_indices)
            pts_3d = np.empty((n_valid, 3), dtype=np.float64)
            for idx, i in enumerate(valid_indices):
                pts_3d[idx] = self.map_points[kf.point_ids[i]].position
            R_mat = kf.pose[:3, :3].T
            t_vec = kf.pose[:3, 3]
            pts_3d_gpu = torch.from_numpy(pts_3d).to(device)
            R_mat_gpu = torch.from_numpy(R_mat).to(device)
            t_vec_gpu = torch.from_numpy(t_vec).to(device)
            pts_cam_gpu = pts_3d_gpu @ R_mat_gpu.T - t_vec_gpu @ R_mat_gpu.T
            valid = (pts_cam_gpu[:, 2] > 0).cpu().numpy()
            if not np.any(valid):
                continue
            pts_proj_gpu = (K_gpu @ pts_cam_gpu[valid].T).T
            pts_proj_gpu = pts_proj_gpu[:, :2] / pts_proj_gpu[:, 2:3]
            obs_2d = kf.keypoints_2d[np.array(valid_indices)[valid]]
            obs_2d_gpu = torch.from_numpy(obs_2d).to(device)
            errors = torch.norm(pts_proj_gpu - obs_2d_gpu, dim=1).cpu().numpy()
            outlier_mask = errors > self.max_reprojection_error
            outlier_indices = np.where(valid)[0][outlier_mask]
            for idx in outlier_indices:
                pid = kf.point_ids[valid_indices[idx]]
                self.map_points[pid].outlier = True
    
    def _cull_keyframes(self):
        if len(self.keyframes) <= self.max_keyframes:
            return
        n_to_remove = len(self.keyframes) - self.max_keyframes
        candidates = self.keyframes[:-5]
        if len(candidates) < n_to_remove:
            return
        scores = []
        for kf in candidates:
            n_covisible = sum(1 for pid in kf.point_ids 
                            if pid in self.map_points and len(self.map_points[pid].observations) > 1)
            scores.append((kf, n_covisible))
        scores.sort(key=lambda x: x[1])
        for kf, _ in scores[:n_to_remove]:
            self.keyframes.remove(kf)
    
    def pose_graph_optimization(self):
        if len(self.keyframes) < 3:
            return
        kf_map = {kf.frame_id: i for i, kf in enumerate(self.keyframes)}
        n_kfs = len(self.keyframes)
        x0 = np.empty(n_kfs * 6, dtype=np.float64)
        idx = 0
        for kf in self.keyframes:
            rvec = R.from_matrix(kf.pose[:3, :3]).as_rotvec()
            x0[idx:idx+3] = rvec
            x0[idx+3:idx+6] = kf.pose[:3, 3]
            idx += 6
        result = least_squares(
            self._pgo_residuals,
            x0,
            args=(kf_map,),
            max_nfev=30,
            verbose=0,
            ftol=1e-3,
            xtol=1e-3
        )
        idx = 0
        for kf in self.keyframes:
            rvec = result.x[idx:idx+3]
            tvec = result.x[idx+3:idx+6]
            kf.pose[:3, :3] = R.from_rotvec(rvec).as_matrix()
            kf.pose[:3, 3] = tvec
            idx += 6
    
    def _pgo_residuals(self, params, kf_map):
        residuals = []
        n_poses = len(kf_map)
        poses = {}
        params_reshaped = params.reshape(n_poses, 6)
        idx = 0
        for kf_id in kf_map.keys():
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = R.from_rotvec(params_reshaped[idx, :3]).as_matrix()
            pose[:3, 3] = params_reshaped[idx, 3:6]
            poses[kf_id] = pose
            idx += 1
        n_edges = len(self.pose_graph_edges) + len(self.loop_edges)
        residuals = np.empty(n_edges * 6, dtype=np.float64)
        res_idx = 0
        for kf_i, kf_j, rel_pose, info in self.pose_graph_edges:
            if kf_i not in poses or kf_j not in poses:
                continue
            predicted_rel = np.linalg.inv(poses[kf_i]) @ poses[kf_j]
            error_pose = np.linalg.inv(rel_pose) @ predicted_rel
            error_rvec = R.from_matrix(error_pose[:3, :3]).as_rotvec()
            error_tvec = error_pose[:3, 3]
            residuals[res_idx:res_idx+3] = error_rvec
            residuals[res_idx+3:res_idx+6] = error_tvec
            res_idx += 6
        for kf_i, kf_j, rel_pose, info in self.loop_edges:
            if kf_i not in poses or kf_j not in poses:
                continue
            predicted_rel = np.linalg.inv(poses[kf_i]) @ poses[kf_j]
            error_pose = np.linalg.inv(rel_pose) @ predicted_rel
            error_rvec = R.from_matrix(error_pose[:3, :3]).as_rotvec()
            error_tvec = error_pose[:3, 3]
            residuals[res_idx:res_idx+3] = error_rvec * 2.0
            residuals[res_idx+3:res_idx+6] = error_tvec * 2.0
            res_idx += 6
        return residuals[:res_idx]
    
    def add_loop_closure(self, kf_i_id, kf_j_id, rel_pose, information=None):
        if information is None:
            information = np.eye(6) * 10.0
        self.loop_edges.append((kf_i_id, kf_j_id, rel_pose, information))
    
    def get_trajectory(self):
        return np.array([kf.pose for kf in self.keyframes])
    
    def get_point_cloud(self):
        points = []
        for pid, mp in self.map_points.items():
            if not mp.outlier and len(mp.observations) >= self.min_observations:
                points.append(mp.position)
        return np.array(points) if points else np.empty((0, 3))