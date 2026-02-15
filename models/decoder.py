"""
Decoder Module for Time-Series Reconstruction

Implementation from Section III.C, Fig. 2:
- Mirrors the encoder architecture
- Upsamples latent code back to original TS dimensions
- Uses transpose convolutions for upsampling

Reference: Lines 297-300 of the paper
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    TS Reconstruction Decoder that mirrors the encoder.
    
    From paper Lines 297-300:
    "The reconstruction branch (decoder) mirrors the encoder to upsample
    the latent representation back to the original TS dimensions, ensuring
    that the latent features preserve the essential information."
    
    Architecture:
    - Input: (batch, 128, T/16) latent representation
    - 4 transpose conv layers with filters [128, 64, 32, 16]
    - Each layer upsamples by factor of 2: T/16 → T/8 → T/4 → T/2 → T
    - Output: (batch, output_channels, T) reconstructed time-series
    """
    
    def __init__(self, latent_dim, output_channels, config):
        """
        Args:
            latent_dim (int): Dimension of latent representation (128)
            output_channels (int): Number of output channels (original input channels)
            config (dict): Configuration dictionary
        """
        super(Decoder, self).__init__()
        
        # Mirror encoder filters in reverse order
        encoder_filters = config['adapter']['encoder']['filters']  # [16, 32, 64, 128]
        self.filters = list(reversed(encoder_filters))  # [128, 64, 32, 16]
        
        self.kernel_size = config['adapter']['encoder']['kernel_size']
        self.stride = config['adapter']['encoder']['stride']
        self.dropout = config['adapter']['encoder']['dropout']
        
        # Build decoder layers
        self.layers = nn.ModuleList()
        in_channels = latent_dim
        
        for i, out_channels in enumerate(self.filters):
            # Transpose convolution for upsampling
            conv_transpose = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                output_padding=1
            )
            
            batch_norm = nn.BatchNorm1d(out_channels)
            dropout = nn.Dropout(p=self.dropout)
            activation = nn.LeakyReLU(negative_slope=0.2)
            
            self.layers.append(nn.Sequential(
                conv_transpose,
                batch_norm,
                dropout,
                activation
            ))
            
            in_channels = out_channels
        
        # Final layer to match output channels
        self.final_conv = nn.Conv1d(
            in_channels=self.filters[-1],
            out_channels=output_channels,
            kernel_size=1  # 1x1 conv to adjust channels
        )
    
    def forward(self, z):
        """
        Forward pass through decoder.
        
        Args:
            z (torch.Tensor): Latent representation of shape (batch, 128, T/16)
        
        Returns:
            torch.Tensor: Reconstructed time-series of shape (batch, output_channels, T)
        """
        x = z
        
        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Final convolution to match original channel count
        x_reconstructed = self.final_conv(x)
        
        return x_reconstructed


if __name__ == "__main__":
    # Test decoder
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    latent_dim = 128
    output_channels = 10  # Original input channels
    batch_size = 4
    time_steps = 288
    latent_time_steps = time_steps // 16  # After encoder downsampling
    
    # Create decoder
    decoder = Decoder(latent_dim, output_channels, config)
    
    # Test with sample latent representation
    z = torch.randn(batch_size, latent_dim, latent_time_steps)
    
    # Forward pass
    x_reconstructed = decoder(z)
    
    print(f"Latent shape: {z.shape}")
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    print(f"Expected output shape: ({batch_size}, {output_channels}, {time_steps})")
