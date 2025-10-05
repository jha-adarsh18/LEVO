import torch
import torch.nn as nn
import numpy as np

"""
To implement the following algorithm from Ren et al. CVPR 2024:

7: Hierarchy structure
8: for stage in range(Snum) do
9: Grouping and Sampling(P N)
10: Get P GS ∈ [B, Nstage, K, 2 * Dstage-1]
11: Local Extractor(P GS)
12: Get Flocal ∈ [B, Nstage, K, Dstage]
13: Attentive Aggregate(Flocal)
14: Get Faggre ∈ [B, Nstage, Dstage]
15: Global Extractor(Faggre)
16: Get P N = Fglobal ∈ [B, Nstage, Dstage]
17: end for
"""

class HierarchyStructure(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp,  group_all=False):
        super(HierarchyStructure, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # Feature extractor with residual connections(bottleneck design)
        # I(x) = f(BN(MLP1(x)))

        self.input_transform = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Conv1d(in_channel, mlp[0], 1),
            nn.ReLU()
        )

        # O(x) = BN(MLP2(x))

        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(mlp[0]),
            nn.Conv1d(mlp[0], mlp[1], 1)
        )

        # Projection layer for residual connection if dimensions don't match

        self.residual_projection = None
        if in_channel != mlp[1]:
            self.residual_projection = nn. Conv1d(in_channel, mlp[1], 1)

        # Attention mechanism for temporal aggregation
        # MLP for computing attention weights

        self.attention_mlp = nn.Sequential(
            nn.Conv1d(mlp[1], mlp[1] // 4, 1),
            nn.ReLU(),
            nn.Conv1d(mlp[1] // 4, 1, 1)
        )

        # Global feature extractor (same structure as local)

        self.global_transform = nn.Sequential(
            nn.BatchNorm1d(mlp[1]),
            nn.Conv1d(mlp[1], mlp[-1], 1),
            nn.ReLU()
        )

        self.final_dim = mlp[-1]

    def standardize_groups(self, grouped_points):
        # P_GSi = (P_Gi - P_Si) / Std(P_Gi)

        B, C, npoint, nsample = grouped_points.shape

        reshaped = grouped_points.permute(0, 2, 1, 3).contiguous() # change dimension to [B, npoint, C, nsample]
        reshaped = reshaped.view(B, npoint, -1) # [B, npoint, C*nsample]
        
        group_mean = torch.mean(reshaped, dim=2, keepdim=True)
        group_std = torch.std(reshaped, dim=2, keepdim=True, unbiased=False)
        group_std = torch.clamp(group_std, min=1e-6)

        standardized = (reshaped - group_mean) / group_std

        standardized = standardized.view(B, npoint, C, nsample)
        standardized = standardized.permute(0, 2, 1, 3).contiguous() # back to [B, C, npoint, nsample]

        return standardized
    
    def local_feature_extractor(self, grouped_points):
        """
        Local feature extraction with residual connection (Equations 8-10):
        I(x) = f(BN(MLP1(x)))
        O(x) = BN(MLP2(x))
        Ext(x) = f(x + O(I(x)))
        """

        B, C, npoint, nsample = grouped_points.shape
        x = grouped_points.view(B, C, npoint * nsample)
        identity = x

        intermediate = self.input_transform(x)
        output = self.bottleneck(intermediate)

        if self.residual_projection is not None:
            identity = self.residual_projection(identity)

        result = torch.relu(identity + output)

        result = result.view(B, -1, npoint, nsample)

        return result
    
    def attention_aggregation(self, features):
        """
        Attention mechanism for temporal aggregation (Equations 11-13):
        Flocal = Ext(x) = (Ft1, Ft2, ..., Ftk)
        A = SoftMax(MLP(Flocal)) = (at1, at2, ..., atk)
        Faggre = A·Flocal = Ft1·at1 + Ft2·at2 + ... + Ftk·atk
        """

        B, C, npoint, nsample = features.shape
        features_reshaped = features.view(B, C, npoint * nsample)

        attention_input = features_reshaped.view(B * npoint, C, nsample)
        attention_weights = self.attention_mlp(attention_input)
        attention_weights = torch.softmax(attention_weights, dim=2)
        attention_weights = attention_weights.view(B, npoint, 1, nsample)

        features_for_agg = features.permute(0, 2, 1, 3)
        aggregated = torch.matmul(features_for_agg, attention_weights.transpose(-1, -2))
        aggregated = aggregated.squeeze(-1).permute(0, 2, 1)

        return aggregated
    
    def forward(self, xyz, points, mask=None):
        # Forward pass for one stage of hierarchy

        xyz = xyz.permute(0, 2, 1)
        if self.group_all:
            new_xyz, grouped_points = self.sample_and_group_all(xyz, points, mask)
        else:
            new_xyz, grouped_points = self.sample_and_group(xyz, points, mask)

        grouped_points = self.standardize_groups(grouped_points)

        local_features = self.local_feature_extractor(grouped_points)
        aggregated_features = self.attention_aggregation(local_features)
        global_features = self.global_transform(aggregated_features)

        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, global_features
    
    def sample_and_group(self, xyz, points, mask=None):
        B, C, N = xyz.shape
        centroids = self.farthest_point_sample_masked(xyz, self.npoint, mask)
        new_xyz = self.index_points(xyz, centroids)

        grouped_xyz, grouped_points = self.knn_group_masked(xyz, new_xyz, points, self.nsample, mask)

        return new_xyz, grouped_points
    
    def sample_and_group_all(self, xyz, points, mask=None):
        B, C, N = xyz.shape
        new_xyz = torch.zeros(B, 3, 1).to(xyz.device)
        grouped_xyz = xyz.view(B, C, 1, N)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, -1, 1, N)], dim=1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points
    
    def farthest_point_sample_masked(self, xyz, npoint, mask=None):
        B, C, N = xyz.shape
        if mask is None:
            centroids = torch.randint(0, N, (B, npoint), device=xyz.device)
        else:
            centroids =  []
            for b in range(B):
                valid_indices = torch.where(mask[b])[0]
                if len(valid_indices) == 0:
                    batch_centroids = torch.randint(0, N, (npoint,), device=xyz.device)
                elif len(valid_indices) < npoint:
                    batch_centroids = valid_indices[torch.randint(0, len(valid_indices),  (npoint,),  device=xyz.device)]
                else:
                    sampled_indices = torch.randperm(len(valid_indices))[:npoint]
                    batch_centroids = valid_indices[sampled_indices]
                centroids.append(batch_centroids)
            centroids = torch.stack(centroids)

        return centroids
    
    def index_points(self, points, idx):
        B, C, N = points.shape
        npoint = idx.shape[1]
        idx_expanded =idx.unsqueeze(1).expand(B, C, npoint)
        new_points = torch.gather(points, 2, idx_expanded)
        return new_points
    
    def knn_group_masked(self, xyz, new_xyz, points, nsample, mask=None):
        B, C_xyz, N = xyz.shape
        npoint = new_xyz.shape[2]

        # compute pairwise distances
        xyz_expanded = xyz.unsqueeze(3) # [B, 3, N, 1]
        new_xyz_expanded = new_xyz.unsqueeze(2) # [B, 3, 1, npoint]
        dists = torch.sum((xyz_expanded - new_xyz_expanded) ** 2, dim=1) # [B, N, npoint]

        if mask is not None:
            mask_expanded = mask.unsqueeze(2).expand(B, N, npoint)
            dists = dists.masked_fill(~mask_expanded, float('inf'))

        # find k nearest neighbors
        dists = dists.permute(0, 2, 1) # [B, npoint, N]
        _, idx = torch.topk(dists, nsample, dim=2, largest=False, sorted=True)

        # handle insufficient valid neighbors
        for b in range(B):
            for p in range(npoint):
                valid_mask = dists[b, p] != float('inf')
                if valid_mask.sum() < nsample:
                    idx[b, p, :] = 0
                elif valid_mask.sum() <nsample:
                    valid_idx = torch.where(valid_mask)[0]
                    repeat_times = (nsample + len(valid_idx) - 1) // len(valid_idx)
                    repeated_idx = valid_idx.repeat(repeat_times)[:nsample]
                    idx[b, p, :] = repeated_idx

        # gather grouped points            
        idx_expanded = idx.view(B, npoint * nsample)
        
        xyz_grouped = self.index_points(xyz, idx_expanded)
        xyz_grouped = xyz_grouped.view(B, C_xyz, npoint, nsample)

        # compute relative positions
        new_xyz_expanded = new_xyz.unsqueeze(3).expand(B, C_xyz, npoint, nsample)
        grouped_xyz = xyz_grouped - new_xyz_expanded

        # index point features
        if points is not None:
            points_grouped = self.index_points(points, idx_expanded)
            points_grouped = points_grouped.view(B, -1, npoint, nsample)
            grouped_points = torch.cat([grouped_xyz, points_grouped], dim=1)
        else:
            grouped_points = grouped_xyz
        
        return grouped_xyz, grouped_points
    