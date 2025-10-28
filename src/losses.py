import torch
import torch.nn as nn
import torch.nn.functional as F


class VOLoss(nn.Module):
    def __init__(self, w_pose=1.0, w_epipolar=0.1):
        super().__init__()
        self.w_pose = w_pose
        self.w_epipolar = w_epipolar
    
    def rotation_loss(self, R_pred, R_gt):
        trace = torch.einsum('bii->b', torch.bmm(R_pred, R_gt.transpose(1, 2)))
        geodesic = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
        loss = 1 - geodesic
        return loss.mean()
    
    def translation_loss(self, t_pred, t_gt):
        t_gt_norm = F.normalize(t_gt, p=2, dim=1)
        cos_sim = (t_pred * t_gt_norm).sum(dim=1)
        loss = 1 - cos_sim
        return loss.mean()
    
    def epipolar_loss(self, kp1, kp2, matches, R_gt, t_gt, K, resolution):
        B, K1, _ = kp1.shape
        K2 = kp2.shape[1]
        
        F_mat = self.fundamental_matrix(R_gt, t_gt, K)
        
        kp1_pixel = kp1 * resolution.unsqueeze(1)
        kp2_pixel = kp2 * resolution.unsqueeze(1)
        
        kp1_homo = torch.cat([kp1_pixel, torch.ones(B, K1, 1, device=kp1.device)], dim=2)
        kp2_homo = torch.cat([kp2_pixel, torch.ones(B, K2, 1, device=kp2.device)], dim=2)
        
        match_mask = matches > 0.1
        
        batch_idx = torch.arange(B, device=kp1.device).view(B, 1, 1).expand(B, K1, K2)
        kp1_idx = torch.arange(K1, device=kp1.device).view(1, K1, 1).expand(B, K1, K2)
        kp2_idx = torch.arange(K2, device=kp1.device).view(1, 1, K2).expand(B, K1, K2)
        
        valid_matches = match_mask.sum()
        if valid_matches < 5:
            return torch.tensor(0.0, device=kp1.device)
        
        b_indices = batch_idx[match_mask]
        i_indices = kp1_idx[match_mask]
        j_indices = kp2_idx[match_mask]
        
        p1_matched = kp1_homo[b_indices, i_indices]
        p2_matched = kp2_homo[b_indices, j_indices]
        F_matched = F_mat[b_indices]
        
        errors = torch.abs(torch.einsum('ni,nij,nj->n', p2_matched, F_matched, p1_matched))
        
        return errors.mean()
    
    def fundamental_matrix(self, R, t, K):
        B = R.shape[0]
        device = R.device
        
        t_norm = F.normalize(t, p=2, dim=1)
        
        t_skew = torch.zeros(B, 3, 3, device=device)
        t_skew[:, 0, 1] = -t_norm[:, 2]
        t_skew[:, 0, 2] = t_norm[:, 1]
        t_skew[:, 1, 0] = t_norm[:, 2]
        t_skew[:, 1, 2] = -t_norm[:, 0]
        t_skew[:, 2, 0] = -t_norm[:, 1]
        t_skew[:, 2, 1] = t_norm[:, 0]
        
        E = torch.bmm(t_skew, R)
        K_inv = torch.inverse(K)
        F = torch.bmm(torch.bmm(K_inv.transpose(1, 2), E), K_inv)
        
        return F
    
    def forward(self, predictions, targets):
        R_pred = predictions['R_pred']
        t_pred = predictions['t_pred']
        R_gt = targets['R_gt']
        t_gt = targets['t_gt']
        K = targets['K']
        resolution = targets['resolution']
        
        rot_loss = self.rotation_loss(R_pred, R_gt)
        trans_loss = self.translation_loss(t_pred, t_gt)
        pose_loss = rot_loss + trans_loss
        
        epi_loss = self.epipolar_loss(
            predictions['keypoints1'],
            predictions['keypoints2'],
            predictions['matches'],
            R_gt, t_gt, K, resolution
        )
        
        total_loss = self.w_pose * pose_loss + self.w_epipolar * epi_loss
        
        stats = {
            'loss': total_loss.item(),
            'pose_loss': pose_loss.item(),
            'rot_loss': rot_loss.item(),
            'trans_loss': trans_loss.item(),
            'epi_loss': epi_loss.item()
        }
        
        return total_loss, stats
    
    def update_weights(self, epoch, total_epochs):
        progress = epoch / total_epochs
        self.w_epipolar = 0.1 * (1 + progress * 0.5)