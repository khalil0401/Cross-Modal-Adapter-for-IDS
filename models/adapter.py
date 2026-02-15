"""
Adapter Module for Learnable TS-to-Image Transformation

Implementation from Section III.C, Fig. 2:
- Converts latent code into 224×224×3 image
- Progressive upsampling from 7×7 seed to 224×224
- Designed to be input-compliant with pretrained vision models

Reference: Lines 301-321 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAdapter(nn.Module):
    """
    Learnable adapter that transforms latent TS representation into images.
    
    From paper Lines 301-321 (Section III.C):
    "The adapter treats the latent code as a 2-D map with time along one axis
    and channels along the other... successive upsampling and convolution stages
    refine this seed into a 224×224×3 image that is directly compatible with
    vision backbones."
    
    Architecture:
    - Input: (batch, latent_dim, time_steps) where latent_dim=128
    - Treat as 2D map: expand to (batch, latent_dim, time_steps, 1)
    - Project to (batch, time_steps, latent_dim, C_p) via 2D conv
    - Resize to 7×7 seed
    - Progressive upsampling: 7×7 → 14×14 → 28×28 → 56×56 → 112×112 → 224×224
    - Output: (batch, 3, 224, 224) RGB image
    """
    
    def __init__(self, latent_dim, config):
        """
        Args:
            latent_dim (int): Latent dimension from encoder (128)
            config (dict): Configuration dictionary
        """
        super(ImageAdapter, self).__init__()
        
        self.latent_dim = latent_dim
        self.seed_size = config['adapter']['adapter_branch']['seed_size']  # [7, 7]
        self.projection_channels = config['adapter']['adapter_branch']['projection_channels']  # 128
        self.output_size = config['adapter']['adapter_branch']['output_size']  # [224, 224, 3]
        
        # Initial projection: expand latent to 2D and project channels
        # Lines 304-306: "(T×C) tensor is expanded to (T×C×1), projected via 2D conv to (T×C×C_p)"
        self.initial_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.projection_channels,
            kernel_size=3,
            padding=1
        )
        
        # Progressive upsampling stages (Lines 307-309)
        # Target sizes: 7×7 → 14×14 → 28×28 → 56×56 → 112×112 → 224×224
        self.upsample_blocks = nn.ModuleList([
            self._make_upsample_block(self.projection_channels, 256),  # 7→14
            self._make_upsample_block(256, 128),  # 14→28
            self._make_upsample_block(128, 64),   # 28→56
            self._make_upsample_block(64, 32),    # 56→112
            self._make_upsample_block(32, 16),    # 112→224
        ])
        
        # Final convolution to produce 3 RGB channels
        self.final_conv = nn.Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
            padding=1
        )
        
        # Sigmoid to ensure pixel values in [0, 1]
        self.sigmoid = nn.Sigmoid()
    
    def _make_upsample_block(self, in_channels, out_channels):
        """
        Create an upsampling block: Upsample + Conv2D + BatchNorm + LeakyReLU
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, z):
        """
        Forward pass: latent → 7×7 seed → 224×224×3 image
        
        Args:
            z (torch.Tensor): Latent representation of shape (batch, latent_dim, time_steps)
        
        Returns:
            torch.Tensor: Generated image of shape (batch, 3, 224, 224)
        """
        batch_size, latent_dim, time_steps = z.shape
        
        # Step 1: Treat latent as 2D map (Lines 301-304)
        # Permute to (batch, time_steps, latent_dim) and add channel dimension
        z_2d = z.transpose(1, 2).unsqueeze(-1)  # (batch, time_steps, latent_dim, 1)
        z_2d = z_2d.permute(0, 3, 1, 2)  # (batch, 1, time_steps, latent_dim)
        
        # Step 2: Project via 2D convolution (Lines 304-306)
        z_proj = self.initial_proj(z_2d)  # (batch, C_p, time_steps, latent_dim)
        
        # Step 3: Resize to 7×7 seed (Lines 307)
        seed_h, seed_w = self.seed_size
        z_seed = F.adaptive_avg_pool2d(z_proj, (seed_h, seed_w))  # (batch, C_p, 7, 7)
        
        # Step 4: Progressive upsampling (Lines 307-309)
        x = z_seed
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        
        # Step 5: Final convolution to 3 channels
        image = self.final_conv(x)  # (batch, 3, 224, 224)
        
        # Step 6: Ensure pixel values in [0, 1]
        image = self.sigmoid(image)
        
        return image


if __name__ == "__main__":
    # Test adapter
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    latent_dim = 128
    batch_size = 4
    time_steps = 18  # After encoder: 288 / 16 = 18
    
    # Create adapter
    adapter = ImageAdapter(latent_dim, config)
    
    # Test with sample latent representation
    z = torch.randn(batch_size, latent_dim, time_steps)
    
    # Forward pass
    image = adapter(z)
    
    print(f"Latent shape: {z.shape}")
    print(f"Generated image shape: {image.shape}")
    print(f"Expected shape: ({batch_size}, 3, 224, 224)")
    print(f"Pixel value range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Image is input-compliant with pretrained models: {image.shape == torch.Size([batch_size, 3, 224, 224])}")
