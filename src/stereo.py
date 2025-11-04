import numpy as np
import torch
import torch.nn.functional as F


def triangulate_points(kp_left, kp_right, K_left, K_right, baseline, min_disparity=0.1):
    B = kp_left.shape[0]
    device = kp_left.device
    
    if isinstance(K_left, np.ndarray):
        K_left = torch.from_numpy(K_left).float().to(device)
    if isinstance(K_right, np.ndarray):
        K_right = torch.from_numpy(K_right).float().to(device)
    
    if K_left.dim() == 2:
        K_left = K_left.unsqueeze(0).expand(B, -1, -1)
    if K_right.dim() == 2:
        K_right = K_right.unsqueeze(0).expand(B, -1, -1)
    
    disparities = kp_left[..., 0] - kp_right[..., 0]
    valid_mask = disparities > min_disparity
    
    fx = K_left[:, 0, 0].unsqueeze(1)
    fy = K_left[:, 1, 1].unsqueeze(1)
    cx = K_left[:, 0, 2].unsqueeze(1)
    cy = K_left[:, 1, 2].unsqueeze(1)
    
    depths = (baseline * fx) / (disparities + 1e-8)
    depths = depths * valid_mask.float()
    
    x = (kp_left[..., 0] - cx) * depths / fx
    y = (kp_left[..., 1] - cy) * depths / fy
    
    points_3d = torch.stack([x, y, depths], dim=-1)
    
    return points_3d, valid_mask


def match_stereo_descriptors(desc_left, desc_right, threshold=0.85, max_disparity=100):
    B, N, D = desc_left.shape
    
    similarity = torch.bmm(desc_left, desc_right.transpose(1, 2))
    max_sim, matched_indices = similarity.max(dim=2)
    valid_matches = max_sim > threshold
    matched_indices = matched_indices * valid_matches.long()
    
    return matched_indices, valid_matches


def compute_stereo_scale(kp_left_1, kp_left_2, kp_right_1, kp_right_2, 
                         desc_left_1, desc_right_1, K, baseline, min_points=10):
    B = kp_left_1.shape[0]
    device = kp_left_1.device
    scales = torch.ones(B, device=device)
    return scales


def stereo_consistency_check(kp_left, kp_right, desc_left, desc_right, 
                             threshold=2.0, descriptor_threshold=0.8):
    matched_indices, valid_matches = match_stereo_descriptors(desc_left, desc_right, descriptor_threshold)
    
    B, N = kp_left.shape[:2]
    device = kp_left.device
    
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    kp_right_matched = kp_right[batch_indices, matched_indices]
    
    y_diff = torch.abs(kp_left[..., 1] - kp_right_matched[..., 1])
    epipolar_valid = y_diff < threshold
    
    return valid_matches & epipolar_valid


def filter_3d_points(points_3d, valid_mask, min_depth=0.1, max_depth=50.0):
    depths = points_3d[..., 2]
    depth_valid = (depths > min_depth) & (depths < max_depth)
    final_valid = valid_mask & depth_valid
    
    return points_3d * final_valid.unsqueeze(-1).float(), final_valid


def create_projection_matrix(K, R=None, t=None):
    if R is None:
        R = torch.eye(3, device=K.device, dtype=K.dtype)
    if t is None:
        t = torch.zeros(3, device=K.device, dtype=K.dtype)
    
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(-1)
    elif t.dim() == 2 and t.shape[1] == 3:
        t = t.unsqueeze(-1)
    
    Rt = torch.cat([R, t], dim=-1)
    
    if K.dim() == 2:
        K = K.unsqueeze(0)
    
    return torch.bmm(K, Rt)


class StereoProcessor:
    def __init__(self, K_left, K_right, baseline, config=None):
        config = config or {}
        self.K_left = K_left
        self.K_right = K_right
        self.baseline = baseline
        
        self.min_disparity = config.get('min_disparity', 0.1)
        self.max_disparity = config.get('max_disparity', 100.0)
        self.descriptor_threshold = config.get('descriptor_threshold', 0.85)
        self.epipolar_threshold = config.get('epipolar_threshold', 2.0)
        self.min_depth = config.get('min_depth', 0.1)
        self.max_depth = config.get('max_depth', 50.0)
        self.min_scale_points = config.get('min_scale_points', 10)
        
        self._K_left_tensor = None
        self._K_right_tensor = None
        self._device = None
    
    def _get_K_tensors(self, device):
        if self._device != device:
            self._K_left_tensor = torch.from_numpy(self.K_left).float().to(device)
            self._K_right_tensor = torch.from_numpy(self.K_right).float().to(device)
            self._device = device
        return self._K_left_tensor, self._K_right_tensor
    
    def process_stereo_pair(self, pred_left, pred_right):
        kp_left = pred_left['keypoints1']
        kp_right = pred_right['keypoints1']
        desc_left = pred_left['descriptors1']
        desc_right = pred_right['descriptors1']
        
        B = kp_left.shape[0]
        device = kp_left.device
        
        K_left_t, K_right_t = self._get_K_tensors(device)
        
        matched_indices, valid_matches = match_stereo_descriptors(
            desc_left, desc_right, self.descriptor_threshold
        )
        
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, kp_left.shape[1])
        kp_right_matched = kp_right[batch_indices, matched_indices]
        
        y_diff = torch.abs(kp_left[..., 1] - kp_right_matched[..., 1])
        epipolar_valid = y_diff < self.epipolar_threshold
        valid_stereo = valid_matches & epipolar_valid
        
        kp_left_valid = kp_left * valid_stereo.unsqueeze(-1).float()
        kp_right_valid = kp_right_matched * valid_stereo.unsqueeze(-1).float()
        
        points_3d, valid_tri = triangulate_points(
            kp_left_valid, kp_right_valid, K_left_t, K_right_t, 
            self.baseline, self.min_disparity
        )
        
        valid_mask = valid_stereo & valid_tri
        points_3d, valid_mask = filter_3d_points(
            points_3d, valid_mask, self.min_depth, self.max_depth
        )
        
        return points_3d, valid_mask
    
    def recover_scale(self, pred_left_1, pred_left_2, pred_right_1, pred_right_2):
        kp_left_1 = pred_left_1['keypoints1']
        device = kp_left_1.device
        K_left_t, _ = self._get_K_tensors(device)
        
        scales = compute_stereo_scale(
            kp_left_1, pred_left_2['keypoints1'], 
            pred_right_1['keypoints1'], pred_right_2['keypoints1'],
            pred_left_1['descriptors1'], pred_right_1['descriptors1'], 
            K_left_t, self.baseline, self.min_scale_points
        )
        
        return scales
    
    def get_metric_pose(self, t_pred, scale):
        if isinstance(scale, torch.Tensor):
            if scale.dim() == 0:
                scale = scale.item()
            elif scale.dim() == 1:
                scale = scale.unsqueeze(-1)
        
        return t_pred * scale