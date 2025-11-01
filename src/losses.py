import torch
import torch.nn as nn
import torch.nn.functional as F


class VOLoss(nn.Module):
    def __init__(self, w_pose=1.0, w_match=0.5, w_epipolar=0.1):
        super().__init__()
        self.w_pose = w_pose
        self.w_match = w_match
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
    
    def correspondence_loss(self, kp1, kp2, matches, R_gt, t_gt):
        B = kp1.shape[0]
        device = kp1.device
        
        total_loss = 0.0
        n_valid = 0
        
        for b in range(B):
            match_weights = matches[b]
            threshold = match_weights.max() * 0.1
            i_idx, j_idx = torch.where(match_weights > threshold)
            
            if len(i_idx) < 3:
                continue
            
            pts1 = kp1[b, i_idx]
            pts2 = kp2[b, j_idx]
            weights = match_weights[i_idx, j_idx]
            
            pts1_homo = torch.cat([pts1, torch.ones(len(pts1), 1, device=device)], dim=1)
            pts2_transformed = (R_gt[b] @ pts1_homo.T).T[:, :2]
            
            errors = torch.norm(pts2 - pts2_transformed, dim=1)
            weighted_error = (errors * weights).sum() / (weights.sum() + 1e-8)
            
            total_loss += weighted_error
            n_valid += 1
        
        if n_valid == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / n_valid
    
    def epipolar_loss(self, kp1, kp2, matches, R_gt, t_gt, K):
        B = kp1.shape[0]
        device = kp1.device
        
        F_mat = self.fundamental_matrix(R_gt, t_gt, K)
        
        total_loss = 0.0
        n_valid = 0
        
        for b in range(B):
            match_weights = matches[b]
            threshold = match_weights.max() * 0.1
            i_idx, j_idx = torch.where(match_weights > threshold)
            
            if len(i_idx) < 3:
                continue
            
            pts1 = kp1[b, i_idx]
            pts2 = kp2[b, j_idx]
            weights = match_weights[i_idx, j_idx]
            
            pts1_homo = torch.cat([pts1, torch.ones(len(pts1), 1, device=device)], dim=1)
            pts2_homo = torch.cat([pts2, torch.ones(len(pts2), 1, device=device)], dim=1)
            
            errors = torch.abs(torch.einsum('ni,ij,nj->n', pts2_homo, F_mat[b], pts1_homo))
            weighted_error = (errors * weights).sum() / (weights.sum() + 1e-8)
            
            total_loss += weighted_error
            n_valid += 1
        
        if n_valid == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / n_valid
    
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
        Fmat = torch.bmm(torch.bmm(K_inv.transpose(1, 2), E), K_inv)
        
        return Fmat
    
    def forward(self, predictions, targets):
        R_pred = predictions['R_pred']
        t_pred = predictions['t_pred']
        R_gt = targets['R_gt']
        t_gt = targets['t_gt']
        K = targets['K']
        
        rot_loss = self.rotation_loss(R_pred, R_gt)
        trans_loss = self.translation_loss(t_pred, t_gt)
        pose_loss = rot_loss + trans_loss
        
        match_loss = self.correspondence_loss(
            predictions['keypoints1'],
            predictions['keypoints2'],
            predictions['matches'],
            R_gt, t_gt
        )
        
        epi_loss = self.epipolar_loss(
            predictions['keypoints1'],
            predictions['keypoints2'],
            predictions['matches'],
            R_gt, t_gt, K
        )
        
        total_loss = self.w_pose * pose_loss + self.w_match * match_loss + self.w_epipolar * epi_loss
        
        stats = {
            'loss': total_loss.item(),
            'pose_loss': pose_loss.item(),
            'rot_loss': rot_loss.item(),
            'trans_loss': trans_loss.item(),
            'match_loss': match_loss.item(),
            'epi_loss': epi_loss.item()
        }
        
        return total_loss, stats
    
    def update_weights(self, epoch, total_epochs):
        progress = epoch / total_epochs
        self.w_match = 0.5 * (1 + progress)
        self.w_epipolar = 0.1 * (1 + progress * 0.5)