import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import os

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())


def ransac_pnp(points_3d, points_2d, K, max_iter=1000, reproj_thresh=3.0):
    points_3d = np.ascontiguousarray(points_3d.reshape(-1, 3), dtype=np.float64)
    points_2d = np.ascontiguousarray(points_2d.reshape(-1, 2), dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)
    
    if len(points_3d) < 4:
        return None, None, np.zeros(len(points_3d), dtype=bool)
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, K, None,
        iterationsCount=max_iter,
        reprojectionError=reproj_thresh,
        flags=cv2.SOLVEPNP_EPNP
    )
    
    if not success or inliers is None:
        return None, None, np.zeros(len(points_3d), dtype=bool)
    
    inlier_mask = np.zeros(len(points_3d), dtype=bool)
    inlier_mask[inliers.flatten()] = True
    
    R_mat = cv2.Rodrigues(rvec)[0]
    t_vec = tvec.flatten()
    
    return R_mat, t_vec, inlier_mask


def ransac_essential(points1, points2, K, max_iter=1000, thresh=1e-3):
    points1 = np.ascontiguousarray(points1.reshape(-1, 2), dtype=np.float64)
    points2 = np.ascontiguousarray(points2.reshape(-1, 2), dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)
    
    if len(points1) < 8:
        return None, None, np.zeros(len(points1), dtype=bool)
    
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=thresh)
    
    if E is None:
        return None, None, np.zeros(len(points1), dtype=bool)
    
    _, R_mat, t_vec, pose_mask = cv2.recoverPose(E, points1, points2, K, mask=mask)
    
    return R_mat, t_vec.flatten(), mask.flatten().astype(bool)


def refine_pose_gauss_newton(R_init, t_init, points_3d, points_2d, K, max_iter=10):
    rvec = R.from_matrix(R_init).as_rotvec()
    x0 = np.concatenate([rvec, t_init])
    
    result = least_squares(
        lambda x: reprojection_residuals(x, points_3d, points_2d, K),
        x0,
        method='lm',
        max_nfev=max_iter
    )
    
    R_opt = R.from_rotvec(result.x[:3]).as_matrix()
    return R_opt, result.x[3:6]


def reprojection_residuals(params, points_3d, points_2d, K):
    points_3d = np.ascontiguousarray(points_3d)
    points_2d = np.ascontiguousarray(points_2d)
    
    R_mat = R.from_rotvec(params[:3]).as_matrix()
    tvec = params[3:6]
    
    points_cam = np.dot(points_3d, R_mat.T) + tvec
    valid = points_cam[:, 2] > 0
    
    n_pts = len(points_3d)
    residuals = np.zeros(n_pts * 2, dtype=np.float64)
    
    if not np.any(valid):
        residuals.fill(10.0)
        return residuals
    
    points_proj = (K @ points_cam[valid].T).T
    points_proj = points_proj[:, :2] / points_proj[:, 2:3]
    diff = points_proj - points_2d[valid]
    
    valid_indices = np.where(valid)[0]
    idx = np.empty(len(valid_indices) * 2, dtype=np.int32)
    idx[0::2] = valid_indices * 2
    idx[1::2] = valid_indices * 2 + 1
    residuals[idx] = diff.ravel()
    
    return residuals


def compute_reprojection_error(R_mat, t_vec, points_3d, points_2d, K):
    points_3d = np.ascontiguousarray(points_3d)
    points_cam = np.dot(points_3d, R_mat.T) + t_vec
    valid = points_cam[:, 2] > 0
    
    errors = np.ones(len(points_3d), dtype=np.float64) * 1e6
    if not np.any(valid):
        return errors
    
    points_proj = (K @ points_cam[valid].T).T
    points_proj = points_proj[:, :2] / points_proj[:, 2:3]
    errors[valid] = np.linalg.norm(points_proj - points_2d[valid], axis=1)
    
    return errors


def filter_matches_geometric(points1, points2, matches, K, method='fundamental', thresh=3.0):
    match_threshold = 0.1
    valid_matches = matches > match_threshold
    
    if not np.any(valid_matches):
        return np.zeros_like(matches, dtype=bool)
    
    match_indices = np.argmax(matches, axis=1)
    match_scores = np.max(matches, axis=1)
    confident = match_scores > match_threshold
    
    if not np.any(confident):
        return np.zeros_like(matches, dtype=bool)
    
    pts1 = np.ascontiguousarray(points1[confident], dtype=np.float64)
    pts2_matched = np.ascontiguousarray(points2[match_indices[confident]], dtype=np.float64)
    
    if method == 'fundamental':
        F, mask = cv2.findFundamentalMat(pts1, pts2_matched, cv2.FM_RANSAC, thresh)
    elif method == 'essential':
        E, mask = cv2.findEssentialMat(pts1, pts2_matched, K.astype(np.float64), cv2.RANSAC, 0.999, thresh / K[0, 0])
    else:
        return confident
    
    if mask is None:
        return confident
    
    inlier_mask_full = np.zeros(len(points1), dtype=bool)
    inlier_mask_full[np.where(confident)[0][mask.flatten().astype(bool)]] = True
    
    return inlier_mask_full


def triangulate_points_cv(points1, points2, K, R, t):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    
    points1 = np.ascontiguousarray(points1.reshape(-1, 2), dtype=np.float64).T
    points2 = np.ascontiguousarray(points2.reshape(-1, 2), dtype=np.float64).T
    
    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d = points_4d[:3] / points_4d[3:]
    
    return points_3d.T


def check_cheirality(points_3d, R, t):
    valid_cam1 = points_3d[:, 2] > 0
    points_cam2 = np.dot(points_3d, R.T) + t
    valid_cam2 = points_cam2[:, 2] > 0
    return valid_cam1 & valid_cam2


def compute_fundamental_matrix(R, t, K):
    t_skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)
    E = t_skew @ R
    K_inv = np.linalg.inv(K)
    return K_inv.T @ E @ K_inv


def sampson_distance(points1, points2, F):
    points1 = np.ascontiguousarray(points1)
    points2 = np.ascontiguousarray(points2)
    
    points1_h = np.hstack([points1, np.ones((len(points1), 1))])
    points2_h = np.hstack([points2, np.ones((len(points2), 1))])
    
    Fx1 = (F @ points1_h.T).T
    FTx2 = (F.T @ points2_h.T).T
    
    numerator = np.sum(points2_h * Fx1, axis=1) ** 2
    denominator = Fx1[:, 0]**2 + Fx1[:, 1]**2 + FTx2[:, 0]**2 + FTx2[:, 1]**2
    
    return numerator / (denominator + 1e-8)


class MotionModel:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.last_pose = np.eye(4, dtype=np.float64)
        self.velocity = np.eye(4, dtype=np.float64)
    
    def predict(self):
        return self.last_pose @ self.velocity
    
    def update(self, new_pose):
        new_velocity = np.linalg.inv(self.last_pose) @ new_pose
        self.velocity = self._interpolate_se3(self.velocity, new_velocity, self.alpha)
        self.last_pose = new_pose.copy()
    
    def _interpolate_se3(self, pose1, pose2, alpha):
        r1 = R.from_matrix(pose1[:3, :3])
        r2 = R.from_matrix(pose2[:3, :3])
        r_interp = R.from_quat((1 - alpha) * r1.as_quat() + alpha * r2.as_quat())
        
        pose_interp = np.eye(4, dtype=np.float64)
        pose_interp[:3, :3] = r_interp.as_matrix()
        pose_interp[:3, 3] = (1 - alpha) * pose1[:3, 3] + alpha * pose2[:3, 3]
        
        return pose_interp
    
    def reset(self):
        self.last_pose = np.eye(4, dtype=np.float64)
        self.velocity = np.eye(4, dtype=np.float64)