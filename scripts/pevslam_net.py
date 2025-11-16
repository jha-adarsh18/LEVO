import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for event sequences
    Better for temporal patterns than PointNet++
    """
    
    def __init__(self, input_dim=4, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, 4] - event features (x, y, t, p)
            mask: [B, N] - validity mask (True = valid, False = padded)
        Returns:
            features: [B, embed_dim] - global descriptor
        """
        B, N, C = x.shape
        
        # Project input
        x = self.input_proj(x)  # [B, N, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :N, :]
        
        # Create attention mask (True = ignore, False = attend)
        if mask is not None:
            attn_mask = ~mask  # Invert mask for transformer
        else:
            attn_mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, N, embed_dim]
        
        # Global pooling (mean over valid events)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
            x_masked = x * mask_expanded
            x_global = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x_global = x.mean(dim=1)  # [B, embed_dim]
        
        # Output projection
        features = self.output_proj(x_global)  # [B, embed_dim]
        
        return features


class PEVSLAM(nn.Module):
    """
    Simplified network for event-based place recognition:
    - Global descriptor (256-d) for loop closure detection
    """

    def __init__(self, base_channel=4, descriptor_dim=256):
        super(PEVSLAM, self).__init__()
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            input_dim=base_channel,
            embed_dim=descriptor_dim,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
        
        self.descriptor_dim = descriptor_dim

    def forward(self, events, mask=None):
        """
        Args:
            events: [B, N, 4] - event data (x, y, t, p)
            mask: [B, N] - validity mask
            
        Returns:
            descriptors: [B, descriptor_dim] - L2-normalized global descriptor
        """
        B, N, C = events.shape
        
        # Extract global features
        descriptors = self.encoder(events, mask)  # [B, descriptor_dim]
        
        # L2 normalize for cosine similarity
        descriptors = F.normalize(descriptors, p=2, dim=1)
        
        return descriptors


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
    descriptors = net(events, mask)

    print(f"Input shape: {events.shape}")
    print(f"Descriptors shape: {descriptors.shape}")  # [B, 256]
    print(f"\nNetwork parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # check outputs
    print(f"\nDescriptor norm (should be ~1): {descriptors.norm(dim=1).mean():.3f}")
    print(f"Descriptor range: [{descriptors.min():.3f}, {descriptors.max():.3f}]")