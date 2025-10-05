import torch
import torch.nn as nn
import torch.nn.functional as F

from eventcloud import HierarchyStructure

class FeaturePropagation(nn.Module):
    """
    For upsampling features to a higher resolution
    uses inverse distance weighted interpolation
    """

    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp.append(nn.BatchNorm1d(out_channel))
            self.mlp.append(nn.ReLU())
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape

        if N2 == 1:
            # global feature, just repeat
            interpolated = points2.repeat
        else:
            # Inverse distance weighted interpolation
            xyz1_expanded = xyz1.unsqueeze(2) # [B, N1, 1, 3]
            xyz2_expanded = xyz2.unsqueeze(1) # [B, 1, N2, 3]

            dists = torch.sum((xyz1_expanded - xyz2_expanded) ** 2, dim=-1) # [B, N1, N2]
            dists = torch.clamp(dists, min=1e-10)

            # find 3 nearest neighbors
            k = min(3, N2)
            dist, idx = torch.topk(dists, k, dim=2, largest=False) # [B, N1, k]

            # inverse distance weights
            weights = 1.0 / dist # [B, N1, k]
            weights = weights / torch.sum(weights, dim=2, keepdim=True) # Normalize

            # gather features
            idx_expanded = idx.unsqueeze(1).expand(B, points2.shape[1], N1, k) # [B, C2, N1, k]
            points2_expanded = points2.unsqueeze(2).expand(B, points2.shape[1], N1, N2) # [B, C2, N1, N2]

            gathered = torch.gather(points2_expanded, 3, idx_expanded) # [B, C2, N1, k]

            # weighted sum
            weights_expanded = weights.unsqueeze(1) # [B, 1, N1, k]
            interpolated = torch.sum(gathered * weights_expanded, dim=3) # [B, C2, N1]

        # concatenate with skip connection
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=1) # [B, C1+C2, N1]
        else:
            new_points = interpolated

        # apply mlp
        for layer in self.mlp:
            new_points = layer(new_points)

        return new_points
    
class PEVSLAM(nn.Module):
    """
    Multi-task network for event-based SLAM:
    - Per - event descriptors (128-d) for matching
    - Keypoint scores (which events are important)
    - Uncertainity estimates (matching confidence)
    """

    def __init__(self, base_channel=4):
        super(PEVSLAM, self).__init__()

        # Hierarchial feature extraction (Pointnet++ backbone)
        self.stage1 = HierarchyStructure(
            npoint=512,
            radius=0.1,
            nsample=32,
            in_channel=base_channel, # x, y, t, p
            mlp = [64, 64, 128]
        )

        self.stage2 = HierarchyStructure(
            npoint=256,
            radius=0.2,
            nsample=64,
            in_channel=128,
            mlp=[128, 128, 256]
        )

        # feature propagation layers (to get per-event features back)
        self.fp2 = FeaturePropagation(256 + 128, [256, 128])
        self.fp1 = FeaturePropagation(128 + base_channel, [128, 128, 128])

        # multi-task heads
        # 1. per-event descriptors(128-d)
        self.descriptor_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1)
        )

        # 2. Keypoint scores (0 - 1, higher = more important)
        self.keypoint_head = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid() # Output in [0, 1]
        )

        # 3. Uncertainity estimates (0 - 1, higher = less certain)
        self.uncertainity_head = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid() # Output in [0, 1]
        )

    def forward(self, events, mask=None):
        B, N, C = events.shape

        # split coordinates and features
        xyz0 = events[:, :, :3] # [B, N, 3] (x, y, t)
        features0 = events.permute(0, 2, 1) # [B, 4, N]

        # Hierarchial encoding
        xyz1, features1 = self.stage1(xyz0, features0, mask) # [B, 512, 3], [B, 128, 512]
        xyz2, features2 = self.stage2(xyz1, features1, None) # [B, 256, 3], [B, 256, 256]

        # feature propagation (upsample back to original resolution)
        features1_upsampled = self.fp2(xyz1, xyz2, features1, features2) # [B, 128, 512]
        features0_upsampled = self.fp1(xyz0, xyz1, features0, features1_upsampled) # [B, 128, N]

        # multi-tasks head
        descriptors = self.descriptor_head(features0_upsampled) # [B, 128, N]
        descriptors = F.normalize(descriptors, p=2, dim=1) # L2 normalize for cosine similarity

        keypoint_scores = self.keypoint_head(features0_upsampled) # [B, 1, N]
        uncertainties = self.uncertainity_head(features0_upsampled) # [B, 1, N]

        # apply mask to outputs
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).float() # [B, 1, N]
            keypoint_scores = keypoint_scores * mask_expanded
            uncertainties = uncertainties * mask_expanded + (1 - mask_expanded) # Masked = Uncertain

        return descriptors, keypoint_scores, uncertainties
    
# Testing
if __name__ == "__main__":
    batch_size = 2
    num_events = 1024

    # fake event data
    events = torch.randn(batch_size, num_events, 4)
    mask = torch.ones(batch_size, num_events, dtype=torch.bool)

    # set some events as invalid
    mask[:, -100:] = False
    events[:, -100:, :] = -1.0

    # create network
    net = PEVSLAM()

    # forward pass
    descriptors, keypoint_scores, uncertainties = net(events, mask)

    print(f"Input shape: {events.shape}")
    print(f"Descriptors shape: {descriptors.shape}")  # [B, 128, N]
    print(f"Keypoint scores shape: {keypoint_scores.shape}")  # [B, 1, N]
    print(f"Uncertainties shape: {uncertainties.shape}")  # [B, 1, N]
    print(f"\nNetwork parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # check outputs
    print(f"\nDescriptor norm (should be ~1): {descriptors.norm(dim=1).mean():.3f}")
    print(f"Keypoint scores range: [{keypoint_scores.min():.3f}, {keypoint_scores.max():.3f}]")
    print(f"Uncertainties range: [{uncertainties.min():.3f}, {uncertainties.max():.3f}]")
    
    # top keypoints
    top_k = 100
    top_keypoints = torch.topk(keypoint_scores.squeeze(1), top_k, dim=1)
    print(f"\nTop {top_k} keypoint scores: {top_keypoints.values[0, :5]}")