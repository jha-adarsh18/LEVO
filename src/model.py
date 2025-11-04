import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class EventBackbone(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(4, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, events, mask):
        B, N, _ = events.shape
        x = self.input_proj(events)
        x = x + self.pos_encoding[:, :N, :]
        attn_mask = ~mask.bool()
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        return x


class EventSampler(nn.Module):
    def __init__(self, d_model=256, num_samples=500):
        super().__init__()
        self.num_samples = num_samples
        self.score_head = nn.Linear(d_model, 1)
    
    def forward(self, features, positions, mask):
        B, N, D = features.shape
        scores = self.score_head(features).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
    
        valid_counts = mask.sum(dim=1).int()
        k_values = torch.clamp(valid_counts, min=10, max=self.num_samples)
        max_k = k_values.max().item()
    
        topk_scores, topk_indices = torch.topk(scores, max_k, dim=1)
        batch_indices = torch.arange(B, device=features.device).unsqueeze(1).expand(-1, max_k)
    
        padded_pos = positions[batch_indices, topk_indices]
        padded_feat = features[batch_indices, topk_indices]
        padded_scores = torch.sigmoid(topk_scores)
    
        valid_mask = torch.arange(max_k, device=features.device).unsqueeze(0) < k_values.unsqueeze(1)
        padded_pos = padded_pos * valid_mask.unsqueeze(-1).float()
        padded_feat = padded_feat * valid_mask.unsqueeze(-1).float()
        padded_scores = padded_scores * valid_mask.float()
    
        return padded_pos, padded_feat, padded_scores


class CrossAttentionMatcher(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.desc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, feat1, feat2):
        feat1_cross, _ = self.cross_attn1(feat1, feat2, feat2)
        feat1 = self.norm1(feat1 + feat1_cross)
        
        feat2_cross, _ = self.cross_attn2(feat2, feat1, feat1)
        feat2 = self.norm2(feat2 + feat2_cross)
        
        desc1 = F.normalize(self.desc_head(feat1), p=2, dim=-1)
        desc2 = F.normalize(self.desc_head(feat2), p=2, dim=-1)
        
        return desc1, desc2


class OptimalTransportMatcher(nn.Module):
    def __init__(self, d_desc=256, num_iterations=5):
        super().__init__()
        self.num_iterations = num_iterations
        self.dustbin = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, desc1, desc2):
        B, N, D = desc1.shape
        M = desc2.shape[1]
        
        scores = torch.bmm(desc1, desc2.transpose(1, 2))
        
        dustbin_row = self.dustbin.expand(B, N, 1)
        dustbin_col = self.dustbin.expand(B, 1, M)
        dustbin_corner = self.dustbin.expand(B, 1, 1)
        
        scores = torch.cat([scores, dustbin_row], dim=2)
        scores = torch.cat([scores, torch.cat([dustbin_col, dustbin_corner], dim=2)], dim=1)
        
        scores = F.log_softmax(scores, dim=-1) + F.log_softmax(scores, dim=-2)
        
        for _ in range(self.num_iterations):
            scores = F.log_softmax(scores, dim=-1)
            scores = F.log_softmax(scores, dim=-2)
        
        scores = scores.exp()
        return scores[:, :N, :M]


class GeometricPoseEstimator(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.corr_attn = nn.MultiheadAttention(d_model*2, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model*2)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 9)
        )
    
    def forward(self, kp1, kp2, feat1, feat2, match_scores, K):
        B = feat1.shape[0]
        device = feat1.device
        
        threshold = match_scores.max(dim=2, keepdim=True)[0] * 0.3
        confident_mask = match_scores > threshold
        
        R_batch = []
        t_batch = []
        
        for b in range(B):
            mask_b = confident_mask[b]
            if mask_b.sum() < 8:
                R_batch.append(torch.eye(3, device=device))
                t_batch.append(torch.tensor([0., 0., 1.], device=device))
                continue
            
            match_idx = mask_b.nonzero(as_tuple=False)
            pts1 = kp1[b, match_idx[:, 0]].cpu().numpy()
            pts2 = kp2[b, match_idx[:, 1]].cpu().numpy()
            K_np = K[b].cpu().numpy()
            
            pts1_px = pts1.copy()
            pts2_px = pts2.copy()
            
            if pts1.max() <= 1.0:
                H, W = 260, 346
                pts1_px[:, 0] *= W
                pts1_px[:, 1] *= H
                pts2_px[:, 0] *= W
                pts2_px[:, 1] *= H
            
            try:
                E, mask_inliers = cv2.findEssentialMat(pts1_px, pts2_px, K_np, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is None:
                    raise ValueError("Essential matrix is None")
                _, R, t, _ = cv2.recoverPose(E, pts1_px, pts2_px, K_np)
                R_batch.append(torch.from_numpy(R).float().to(device))
                t_batch.append(torch.from_numpy(t.squeeze()).float().to(device))
            except:
                corr_feats = torch.cat([
                    feat1[b:b+1, match_idx[:, 0]],
                    feat2[b:b+1, match_idx[:, 1]]
                ], dim=-1)
                
                attn_out, _ = self.corr_attn(corr_feats, corr_feats, corr_feats)
                corr_feats = self.norm(corr_feats + attn_out)
                agg = corr_feats.mean(dim=1)
                pose = self.mlp(agg)
                
                rot_6d = pose[:, :6]
                trans = F.normalize(pose[:, 6:9], p=2, dim=1)
                R = self.rotation_6d_to_matrix(rot_6d)
                
                R_batch.append(R[0])
                t_batch.append(trans[0])
        
        return torch.stack(R_batch), torch.stack(t_batch)
    
    def rotation_6d_to_matrix(self, x):
        B = x.shape[0]
        x = x.view(B, 2, 3)
        
        a1 = x[:, 0]
        a2 = x[:, 1]
        
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.linalg.cross(b1, b2, dim=1)
        
        return torch.stack([b1, b2, b3], dim=1)


class EventVO(nn.Module):
    def __init__(self, d_model=256, num_samples=500):
        super().__init__()
        self.backbone = EventBackbone(d_model=d_model, num_layers=6)
        self.sampler = EventSampler(d_model=d_model, num_samples=num_samples)
        self.cross_matcher = CrossAttentionMatcher(d_model=d_model)
        self.matcher = OptimalTransportMatcher(d_desc=d_model)
        self.pose_estimator = GeometricPoseEstimator(d_model=d_model)
    
    def forward(self, events1, mask1, events2, mask2, K=None):
        pos1 = events1[:, :, :2]
        pos2 = events2[:, :, :2]
        
        feats1 = self.backbone(events1, mask1)
        feats2 = self.backbone(events2, mask2)
        
        kp1, kp_feats1, scores1 = self.sampler(feats1, pos1, mask1)
        kp2, kp_feats2, scores2 = self.sampler(feats2, pos2, mask2)
        
        desc1, desc2 = self.cross_matcher(kp_feats1, kp_feats2)
        matches = self.matcher(desc1, desc2)
        
        if K is None:
            K = torch.eye(3, device=events1.device).unsqueeze(0).expand(events1.shape[0], -1, -1)
        
        R, t = self.pose_estimator(kp1, kp2, kp_feats1, kp_feats2, matches, K)
        
        return {
            'R_pred': R,
            't_pred': t,
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': desc1,
            'descriptors2': desc2,
            'matches': matches,
            'scores1': scores1,
            'scores2': scores2
        }