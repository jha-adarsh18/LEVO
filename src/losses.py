import torch
import torch.nn as nn
import torch.nn.functional as F


class VOLoss(nn.Module):
    def __init__(self, w_pose=1.0, w_match=0.0, w_epipolar=0.0, w_contrastive=0.0):
        super().__init__()
        self.w_pose = w_pose
        self.w_match = w_match
        self.w_epipolar = w_epipolar
        self.w_contrastive = w_contrastive
    
    def rotation_loss(self, R_pred, R_gt):
        trace = torch.einsum('bii->b', torch.bmm(R_pred, R_gt.transpose(1, 2)))
        geodesic = ((trace - 1) / 2).clamp(-0.9999, 0.9999)  # aggressive clamp
        loss = 1 - geodesic
    
        ortho_loss = torch.norm(torch.bmm(R_pred, R_pred.transpose(1, 2)) - torch.eye(3, device=R_pred.device).unsqueeze(0), p='fro', dim=(1, 2))
    
        return loss.mean() + 0.5 * ortho_loss.mean()  # increased from 0.1

    def translation_loss(self, t_pred, t_gt):
        t_pred_norm = F.normalize(t_pred, p=2, dim=1)
        t_gt_norm = F.normalize(t_gt, p=2, dim=1)
        cos_sim = (t_pred_norm * t_gt_norm).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        loss = torch.acos(cos_sim)
        return loss.mean()
    
    def correspondence_loss(self, kp1, kp2, matches, R_gt, t_gt):
        B, N1, _ = kp1.shape
        N2 = kp2.shape[1]
        device = kp1.device
        
        threshold = 0.01
        match_mask = matches > threshold
        has_matches = match_mask.sum(dim=(1, 2)) >= 3
        
        if not has_matches.any():
            return (matches * 0.0).sum()
        
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
        total_loss = batch_losses.sum() / (has_matches.sum().float() + 1e-8)
        
        return total_loss
    
    def epipolar_loss(self, kp1, kp2, matches, R_gt, t_gt, K):
        B, N1, _ = kp1.shape
        N2 = kp2.shape[1]
        device = kp1.device
        
        F_mat = self.fundamental_matrix(R_gt, t_gt, K)
        
        matches_flat = matches.reshape(B, -1)
        threshold_vals = torch.quantile(matches_flat, 0.8, dim=1, keepdim=True)
        threshold = threshold_vals.view(B, 1, 1).expand(B, N1, N2)
        match_mask = matches > threshold
        has_matches = match_mask.sum(dim=(1, 2)) >= 3
        
        if not has_matches.any():
            return (matches * 0.0).sum()
        
        kp1_expanded = kp1.unsqueeze(2).expand(B, N1, N2, 2)
        kp2_expanded = kp2.unsqueeze(1).expand(B, N1, N2, 2)
        pts1_homo = torch.cat([kp1_expanded, torch.ones(B, N1, N2, 1, device=device)], dim=-1)
        pts2_homo = torch.cat([kp2_expanded, torch.ones(B, N1, N2, 1, device=device)], dim=-1)
        
        F_mat_expanded = F_mat.unsqueeze(1).unsqueeze(1).expand(B, N1, N2, 3, 3)
        Fx1 = torch.matmul(F_mat_expanded, pts1_homo.unsqueeze(-1)).squeeze(-1)
        FTx2 = torch.matmul(F_mat_expanded.transpose(-2, -1), pts2_homo.unsqueeze(-1)).squeeze(-1)
        
        numerator = torch.einsum('bijk,bijk->bij', pts2_homo, Fx1) ** 2
        denominator = Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + FTx2[..., 0] ** 2 + FTx2[..., 1] ** 2 + 1e-8
        sampson_errors = numerator / denominator
        
        confident_matches = matches * match_mask.float()
        weighted_errors = sampson_errors * confident_matches
        match_sums = confident_matches.sum(dim=(1, 2))
        
        batch_losses = weighted_errors.sum(dim=(1, 2)) / (match_sums + 1e-8)
        batch_losses = batch_losses * has_matches.float()
        total_loss = batch_losses.sum() / (has_matches.sum().float() + 1e-8)
        
        return total_loss
    
    def contrastive_loss(self, desc1, desc2, matches):
        B, N, D = desc1.shape
        device = desc1.device
        temperature = 0.07
        
        threshold = matches.max(dim=2, keepdim=True)[0] * 0.5
        positive_mask = matches > threshold
        has_positives = positive_mask.sum(dim=(1, 2)) > 0
        
        if not has_positives.any():
            return (desc1 * 0.0).sum() + (desc2 * 0.0).sum()
        
        similarity = torch.bmm(desc1, desc2.transpose(1, 2)) / temperature
        positive_sim = similarity * positive_mask.float()
        positive_exp = torch.exp(positive_sim) * positive_mask.float()
        all_exp = torch.exp(similarity)
        
        loss_per_point = -torch.log((positive_exp.sum(dim=2) + 1e-8) / (all_exp.sum(dim=2) + 1e-8))
        valid_points = positive_mask.sum(dim=2) > 0
        batch_losses = (loss_per_point * valid_points.float()).sum(dim=1) / (valid_points.sum(dim=1).float() + 1e-8)
        batch_losses = batch_losses * has_positives.float()
        total_loss = batch_losses.sum() / (has_positives.sum().float() + 1e-8)
        
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
        K_inv = torch.linalg.inv(K)
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
        
        match_loss = (R_pred * 0.0).sum()
        if self.w_match > 1e-6:
            match_loss = self.correspondence_loss(
                predictions['keypoints1'],
                predictions['keypoints2'],
                predictions['matches'],
                R_gt, t_gt
            )
        
        epi_loss = (R_pred * 0.0).sum()
        if self.w_epipolar > 1e-6:
            epi_loss = self.epipolar_loss(
                predictions['keypoints1'],
                predictions['keypoints2'],
                predictions['matches'],
                R_gt, t_gt, K
            )
        
        contrastive_loss = (R_pred * 0.0).sum()
        if self.w_contrastive > 1e-6:
            contrastive_loss = self.contrastive_loss(
                predictions['descriptors1'],
                predictions['descriptors2'],
                predictions['matches']
            )
        
        total_loss = (self.w_pose * pose_loss + 
                     self.w_match * match_loss + 
                     self.w_epipolar * epi_loss +
                     self.w_contrastive * contrastive_loss)
        
        stats = {
            'loss': total_loss.item(),
            'pose_loss': pose_loss.item(),
            'rot_loss': rot_loss.item(),
            'trans_loss': trans_loss.item(),
            'match_loss': match_loss.item(),
            'epi_loss': epi_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }
        
        return total_loss, stats
    
    def update_weights(self, epoch, total_epochs):
        if epoch < 20:
            self.w_pose = 1.0
            self.w_match = 0.0
            self.w_epipolar = 0.0
            self.w_contrastive = 0.0
        else:
            self.w_pose = 1.0
            self.w_match = 0.05 
            self.w_epipolar = 0.02  
            self.w_contrastive = 0.01 