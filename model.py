import torch
import torch.nn as nn
import torch.nn.functional as F


class EventBackbone(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(4, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
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
        
        k = min(self.num_samples, mask.sum(dim=1).min().item())
        if k == 0:
            return torch.zeros(B, 1, 2, device=features.device), torch.zeros(B, 1, D, device=features.device), torch.zeros(B, 1, device=features.device)
        
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        sampled_positions = torch.gather(positions, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 2))
        sampled_features = torch.gather(features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))
        
        return sampled_positions, sampled_features, torch.sigmoid(topk_scores)


class CrossAttentionMatcher(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.desc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, feat1, feat2):
        feat1_cross, _ = self.cross_attn(feat1, feat2, feat2)
        feat1 = self.norm1(feat1 + feat1_cross)
        
        feat2_cross, _ = self.cross_attn(feat2, feat1, feat1)
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
    def __init__(self):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pos1, pos2, match_scores, K):
        B = pos1.shape[0]
        device = pos1.device
        
        max_matches = match_scores.max(dim=2)[0].max(dim=1)[0]
        threshold = max_matches * 0.1
        
        R_pred = []
        t_pred = []
        
        for b in range(B):
            mask = match_scores[b] > threshold[b]
            i_idx, j_idx = torch.where(mask)
            
            if len(i_idx) < 5:
                R_pred.append(torch.eye(3, device=device))
                t_pred.append(torch.tensor([0., 0., 1.], device=device))
                continue
            
            pts1 = pos1[b, i_idx]
            pts2 = pos2[b, j_idx]
            weights = match_scores[b, i_idx, j_idx]
            
            weights = self.weight_net(weights.unsqueeze(-1)).squeeze(-1)
            weights = weights / (weights.sum() + 1e-8)
            
            centroid1 = (pts1 * weights.unsqueeze(-1)).sum(dim=0)
            centroid2 = (pts2 * weights.unsqueeze(-1)).sum(dim=0)
            
            pts1_centered = pts1 - centroid1
            pts2_centered = pts2 - centroid2
            
            H = (pts1_centered.T * weights) @ pts2_centered
            
            U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T
            
            if torch.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            t = centroid2 - R @ centroid1
            t = F.normalize(t, p=2, dim=0)
            
            R_pred.append(R)
            t_pred.append(t)
        
        R_pred = torch.stack(R_pred)
        t_pred = torch.stack(t_pred)
        
        return R_pred, t_pred


class EventVO(nn.Module):
    def __init__(self, d_model=256, num_samples=500):
        super().__init__()
        self.backbone = EventBackbone(d_model=d_model)
        self.sampler = EventSampler(d_model=d_model, num_samples=num_samples)
        self.cross_matcher = CrossAttentionMatcher(d_model=d_model)
        self.matcher = OptimalTransportMatcher(d_desc=d_model)
        self.pose_estimator = GeometricPoseEstimator()
    
    def forward(self, events1, mask1, events2, mask2):
        pos1 = events1[:, :, :2]
        pos2 = events2[:, :, :2]
        
        feats1 = self.backbone(events1, mask1)
        feats2 = self.backbone(events2, mask2)
        
        kp1, kp_feats1, scores1 = self.sampler(feats1, pos1, mask1)
        kp2, kp_feats2, scores2 = self.sampler(feats2, pos2, mask2)
        
        desc1, desc2 = self.cross_matcher(kp_feats1, kp_feats2)
        
        matches = self.matcher(desc1, desc2)
        
        K = torch.eye(3, device=events1.device).unsqueeze(0).expand(events1.shape[0], -1, -1)
        R, t = self.pose_estimator(kp1, kp2, matches, K)
        
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