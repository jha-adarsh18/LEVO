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


class KeypointDetector(nn.Module):
    def __init__(self, d_model=256, grid_h=90, grid_w=160):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.conv1 = nn.Conv2d(d_model, d_model, 3, padding=1)
        self.conv2 = nn.Conv2d(d_model, 65, 3, padding=1)
    
    def spatial_binning(self, event_features, positions, mask):
        B, N, D = event_features.shape
        device = event_features.device
        
        x_bins = (positions[:, :, 0] * (self.grid_w - 1)).long().clamp(0, self.grid_w - 1)
        y_bins = (positions[:, :, 1] * (self.grid_h - 1)).long().clamp(0, self.grid_h - 1)
        
        feature_grid = torch.zeros(B, D, self.grid_h, self.grid_w, device=device)
        count_grid = torch.zeros(B, 1, self.grid_h, self.grid_w, device=device)
        
        mask_expanded = mask.unsqueeze(-1)
        masked_features = event_features * mask_expanded
        
        for b in range(B):
            valid_mask = mask[b].bool()
            if valid_mask.sum() == 0:
                continue
            
            x_b = x_bins[b, valid_mask]
            y_b = y_bins[b, valid_mask]
            feats_b = masked_features[b, valid_mask]
            
            indices = y_b * self.grid_w + x_b
            
            for d in range(D):
                feature_grid[b, d].view(-1).scatter_add_(0, indices, feats_b[:, d])
            
            count_grid[b, 0].view(-1).scatter_add_(0, indices, torch.ones_like(x_b, dtype=torch.float32))
        
        count_grid = count_grid.clamp(min=1)
        feature_grid = feature_grid / count_grid
        
        return feature_grid
    
    def forward(self, event_features, positions, mask, top_k=500):
        feature_grid = self.spatial_binning(event_features, positions, mask)
        
        x = F.relu(self.conv1(feature_grid))
        x = self.conv2(x)
        
        B, C, H, W = x.shape
        x = x.view(B, 65, H, W)
        
        heatmap = F.softmax(x[:, :64].view(B, 8, 8, H, W), dim=1)
        heatmap = heatmap.permute(0, 3, 1, 4, 2).reshape(B, H*8, W*8)
        dustbin = x[:, 64:65, :, :]
        
        scores = heatmap.view(B, -1)
        topk_scores, topk_indices = torch.topk(scores, top_k, dim=1)
        
        keypoints_y = (topk_indices // (W * 8)).float() / (H * 8)
        keypoints_x = (topk_indices % (W * 8)).float() / (W * 8)
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)
        
        return keypoints, topk_scores, feature_grid


class DescriptorExtractor(nn.Module):
    def __init__(self, d_model=256, d_desc=256, radius=0.05, min_events=3):
        super().__init__()
        self.radius = radius
        self.min_events = min_events
        self.query_proj = nn.Linear(2, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.desc_head = nn.Sequential(
            nn.Linear(d_model, d_desc),
            nn.LayerNorm(d_desc)
        )
    
    def forward(self, event_features, positions, mask, keypoints):
        B, K, _ = keypoints.shape
        N, D = event_features.shape[1], event_features.shape[2]
        
        kp_expanded = keypoints.unsqueeze(2)
        pos_expanded = positions.unsqueeze(1)
        
        dists = torch.norm(pos_expanded - kp_expanded, dim=-1)
        local_masks = (dists < self.radius) & mask.unsqueeze(1)
        
        queries = self.query_proj(keypoints)
        
        max_local = local_masks.sum(dim=2).max().item()
        if max_local < self.min_events:
            return torch.zeros(B, K, self.desc_head[-1].normalized_shape[0], device=event_features.device)
        
        max_local = min(max_local, 256)
        
        batch_indices = torch.arange(B, device=event_features.device).view(B, 1, 1).expand(B, K, N)
        kp_indices = torch.arange(K, device=event_features.device).view(1, K, 1).expand(B, K, N)
        event_indices = torch.arange(N, device=event_features.device).view(1, 1, N).expand(B, K, N)
        
        flat_batch = batch_indices[local_masks]
        flat_kp = kp_indices[local_masks]
        flat_event = event_indices[local_masks]
        
        local_feats = torch.zeros(B, K, max_local, D, device=event_features.device)
        attn_mask = torch.ones(B, K, max_local, device=event_features.device, dtype=torch.bool)
        
        counts = torch.zeros(B, K, device=event_features.device, dtype=torch.long)
        counts.scatter_add_(0, flat_batch * K + flat_kp, torch.ones_like(flat_batch))
        counts = counts.view(B, K).clamp(max=max_local)
        
        sorted_idx = torch.argsort(flat_batch * K + flat_kp)
        flat_batch = flat_batch[sorted_idx]
        flat_kp = flat_kp[sorted_idx]
        flat_event = flat_event[sorted_idx]
        
        unique_bk = flat_batch * K + flat_kp
        change_idx = torch.cat([torch.tensor([0], device=event_features.device), 
                                torch.where(unique_bk[1:] != unique_bk[:-1])[0] + 1,
                                torch.tensor([len(unique_bk)], device=event_features.device)])
        
        for i in range(len(change_idx) - 1):
            start = change_idx[i]
            end = change_idx[i + 1]
            n_local = min(end - start, max_local)
            
            b = flat_batch[start].item()
            k = flat_kp[start].item()
            
            event_idx = flat_event[start:start + n_local]
            local_feats[b, k, :n_local] = event_features[b, event_idx]
            attn_mask[b, k, :n_local] = False
        
        queries_flat = queries.view(B * K, 1, D)
        local_feats_flat = local_feats.view(B * K, max_local, D)
        attn_mask_flat = attn_mask.view(B * K, max_local)
        
        attn_out, _ = self.attn(queries_flat, local_feats_flat, local_feats_flat, 
                                key_padding_mask=attn_mask_flat)
        descriptors = self.desc_head(attn_out.view(B, K, D))
        descriptors = F.normalize(descriptors, p=2, dim=-1)
        
        return descriptors


class FeatureMatcher(nn.Module):
    def __init__(self, d_desc=256):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(-2.3))
    
    def forward(self, desc1, desc2):
        temperature = self.log_temperature.exp().clamp(min=0.01, max=10.0)
        sim = torch.bmm(desc1, desc2.transpose(1, 2)) / temperature
        scores1 = F.softmax(sim, dim=2)
        scores2 = F.softmax(sim, dim=1)
        mutual_scores = scores1 * scores2.transpose(1, 2)
        return mutual_scores


class PoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
    
    def forward(self, desc1, desc2, match_scores):
        B = desc1.shape[0]
        
        match_weights1 = match_scores.sum(dim=2, keepdim=True) + 1e-8
        match_weights2 = match_scores.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8
        
        desc1_pooled = (desc1 * match_weights1).sum(dim=1) / match_weights1.sum(dim=1)
        desc2_pooled = (desc2 * match_weights2).sum(dim=1) / match_weights2.sum(dim=1)
        
        features = torch.cat([desc1_pooled, desc2_pooled], dim=1)
        pose = self.mlp(features)
        
        rot_6d = pose[:, :6]
        trans = F.normalize(pose[:, 6:9], p=2, dim=1)
        
        R = self.rotation_6d_to_matrix(rot_6d)
        
        return R, trans
    
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
    def __init__(self, d_model=256, d_desc=256):
        super().__init__()
        self.backbone = EventBackbone(d_model=d_model)
        self.keypoint_detector = KeypointDetector(d_model=d_model)
        self.descriptor_extractor = DescriptorExtractor(d_model=d_model, d_desc=d_desc)
        self.matcher = FeatureMatcher(d_desc=d_desc)
        self.pose_estimator = PoseEstimator()
    
    def forward(self, events1, mask1, events2, mask2):
        pos1 = events1[:, :, :2]
        pos2 = events2[:, :, :2]
        
        feats1 = self.backbone(events1, mask1)
        feats2 = self.backbone(events2, mask2)
        
        kp1, scores1, grid1 = self.keypoint_detector(feats1, pos1, mask1)
        kp2, scores2, grid2 = self.keypoint_detector(feats2, pos2, mask2)
        
        desc1 = self.descriptor_extractor(feats1, pos1, mask1, kp1)
        desc2 = self.descriptor_extractor(feats2, pos2, mask2, kp2)
        
        matches = self.matcher(desc1, desc2)
        
        R, trans = self.pose_estimator(desc1, desc2, matches)
        
        return {
            'R_pred': R,
            't_pred': trans,
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': desc1,
            'descriptors2': desc2,
            'matches': matches,
            'scores1': scores1,
            'scores2': scores2
        }