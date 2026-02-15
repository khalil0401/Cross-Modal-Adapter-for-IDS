"""
Encoder Module for Cross-Modal Adapter

Implementation of the encoder from Section III.C, Fig. 2:
- 4-layer 1D CNN with progressive downsampling
- Filters: [16, 32, 64, 128]
- Stride: 2 (temporal resolution reduced by half each layer)
- BatchNorm + Dropout(0.3) + LeakyReLU after each conv

Reference: Lines 247-254 of the paper
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    1D CNN Encoder that compresses input time-series into latent representation.
    
    Architecture (from paper Section III.C):
    - Input: (batch, channels, time_steps)
    - 4 sequential conv layers with filters [16, 32, 64, 128]
    - Each layer: Conv1D → BatchNorm → Dropout → LeakyReLU
    - Stride=2 progressively reduces temporal dimension: T → T/2 → T/4 → T/8 → T/16
    - Output: (batch, 128, T/16) latent representation
    """
    
    def __init__(self, input_channels, config):
        """
        Args:
            input_channels (int): Number of input channels (features)
            config (dict): Configuration dictionary containing encoder hyperparameters
        """
        super(Encoder, self).__init__()
        
        # Extract hyperparameters from config (Section V, Lines 407-412)
        self.filters = config['adapter']['encoder']['filters']  # [16, 32, 64, 128]
        self.kernel_size = config['adapter']['encoder']['kernel_size']  # 3
        self.stride = config['adapter']['encoder']['stride']  # 2
        self.dropout = config['adapter']['encoder']['dropout']  # 0.3
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, out_channels in enumerate(self.filters):
            # Convolutional layer with stride=2 for downsampling
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2
            )
            
            # Regularization: BatchNorm + Dropout (Lines 251-254)
            batch_norm = nn.BatchNorm1d(out_channels)
            dropout = nn.Dropout(p=self.dropout)
            activation = nn.LeakyReLU(negative_slope=0.2)
            
            # Add layer block
            self.layers.append(nn.Sequential(
                conv,
                batch_norm,
                dropout,
                activation
            ))
            
            in_channels = out_channels
        
        self.latent_dim = self.filters[-1]  # 128
        
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input time-series of shape (batch, channels, time_steps)
        
        Returns:
            torch.Tensor: Latent representation of shape (batch, 128, time_steps/16)
        """
        # Pass through all encoder layers sequentially
        for layer in self.layers:
            x = layer(x)
        
        return x


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for latent features.
    
    From paper Lines 291-294:
    "Self-attention calculates pairwise similarity between latent elements,
    producing output emphasizing most informative temporal patterns."
    """
    
    def __init__(self, latent_dim):
        super(SelfAttention, self).__init__()
        
        # Query, Key, Value projections
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        
        self.scale = latent_dim ** 0.5
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch, latent_dim, time_steps)
        
        Returns:
            torch.Tensor: Attention-refined features of same shape
        """
        # Transpose for attention: (batch, time_steps, latent_dim)
        x = x.transpose(1, 2)
        
        batch_size, seq_len, latent_dim = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch, seq_len, latent_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores: Q @ K^T / sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Residual connection (Lines 291-294)
        output = x + attended
        
        # Transpose back: (batch, latent_dim, time_steps)
        output = output.transpose(1, 2)
        
        return output


if __name__ == "__main__":
    # Test encoder
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create encoder
    input_channels = 10  # Example: 10 IoT features
    encoder = Encoder(input_channels, config)
    
    # Test with sample input
    batch_size = 4
    time_steps = 288  # Daily window
    x = torch.randn(batch_size, input_channels, time_steps)
    
    # Forward pass
    latent = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Expected latent time dimension: {time_steps // (2**4)} = {time_steps // 16}")
    
    # Test self-attention
    attention = SelfAttention(config['adapter']['encoder']['filters'][-1])
    attended_latent = attention(latent)
    print(f"Attended latent shape: {attended_latent.shape}")
