import torch
import torch.nn as nn
import torch.nn.functional as F


class VOLoss(nn.Module):
    def __init__(self, w_pose=1.0, w_match=0.0, w_epipolar=0.1):
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
        B, N1, _ = kp1.shape
        N2 = kp2.shape[1]
        device = kp1.device
        
        threshold = 0.01
        match_mask = matches > threshold
        
        has_matches = match_mask.sum(dim=(1, 2)) >= 3
        
        if not has_matches.any():
            return torch.tensor(0.0, device=device)
        
        kp1_expanded = kp1.unsqueeze(2).expand(B, N1, N2, 2)
        kp2_expanded = kp2.unsqueeze(1).expand(B, N1, N2, 2)
        
        pts1_homo = torch.cat([kp1_expanded, torch.ones(B, N1, N2, 1, device=device)], dim=-1)
        
        R_gt_expanded = R_gt.unsqueeze(1).unsqueeze(1).expand(B, N1, N2, 3, 3)
        pts2_transformed = torch.matmul(R_gt_expanded, pts1_homo.unsqueeze(-1)).squeeze(-1)[..., :2]
        
        errors = torch.norm(kp2_expanded - pts2_transformed, dim=-1)
        
        weighted_errors = errors * matches
        match_sums = matches.sum(dim=(1, 2))
        
        batch_losses = weighted_errors.sum(dim=(1, 2)) / (match_sums + 1e-8)
        batch_losses = batch_losses * has_matches.float()
        
        total_loss = batch_losses.sum() / (has_matches.sum() + 1e-8)
        
        return total_loss
    
    def epipolar_loss(self, kp1, kp2, matches, R_gt, t_gt, K):
        B, N1, _ = kp1.shape
        N2 = kp2.shape[1]
        device = kp1.device
        
        F_mat = self.fundamental_matrix(R_gt, t_gt, K)
        
        threshold = matches.amax(dim=(1, 2), keepdim=True) * 0.1
        match_mask = matches > threshold
        
        has_matches = match_mask.sum(dim=(1, 2)) >= 3
        
        if not has_matches.any():
            return torch.tensor(0.0, device=device)
        
        kp1_expanded = kp1.unsqueeze(2).expand(B, N1, N2, 2)
        kp2_expanded = kp2.unsqueeze(1).expand(B, N1, N2, 2)
        
        pts1_homo = torch.cat([kp1_expanded, torch.ones(B, N1, N2, 1, device=device)], dim=-1)
        pts2_homo = torch.cat([kp2_expanded, torch.ones(B, N1, N2, 1, device=device)], dim=-1)
        
        F_mat_expanded = F_mat.unsqueeze(1).unsqueeze(1).expand(B, N1, N2, 3, 3)
        
        errors = torch.abs(torch.einsum('bijk,bijk->bij', pts2_homo, torch.matmul(F_mat_expanded, pts1_homo.unsqueeze(-1)).squeeze(-1)))
        
        weighted_errors = errors * matches
        match_sums = matches.sum(dim=(1, 2))
        
        batch_losses = weighted_errors.sum(dim=(1, 2)) / (match_sums + 1e-8)
        batch_losses = batch_losses * has_matches.float()
        
        total_loss = batch_losses.sum() / (has_matches.sum() + 1e-8)
        
        return total_loss
    
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
        
        # Always compute pose loss
        rot_loss = self.rotation_loss(R_pred, R_gt)
        trans_loss = self.translation_loss(t_pred, t_gt)
        pose_loss = rot_loss + trans_loss
        
        # Only compute expensive geometric losses if their weight is non-zero
        match_loss = torch.tensor(0.0, device=R_pred.device)
        if self.w_match > 1e-6:
            match_loss = self.correspondence_loss(
                predictions['keypoints1'],
                predictions['keypoints2'],
                predictions['matches'],
                R_gt, t_gt
            )
        
        epi_loss = torch.tensor(0.0, device=R_pred.device)
        if self.w_epipolar > 1e-6:
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