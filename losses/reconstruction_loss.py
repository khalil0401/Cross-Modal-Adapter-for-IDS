"""
Reconstruction Loss (L_rec)

Implementation of Equation 4 from the paper:
L_rec = (1 / (N * D)) * ||x - x_hat||_2^2

Reference: Lines 263-268
"""

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    Mean Squared Error (MSE) reconstruction loss.
    
    From paper Equation 4 (Lines 263-268):
    "The reconstruction loss optimizes the mean-squared error, where x ∈ R^(N×D)
    is the input window (length N, D channels) and x_hat its decoder reconstruction."
    
    L_rec = (1 / (N * D)) * ||x - x_hat||_2^2
    """
    
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, x_reconstructed, x_original):
        """
        Compute reconstruction loss.
        
        Args:
            x_reconstructed (torch.Tensor): Reconstructed TS from decoder (batch, D, N)
            x_original (torch.Tensor): Original input TS (batch, D, N)
        
        Returns:
            torch.Tensor: Scalar reconstruction loss
        """
        return self.mse(x_reconstructed, x_original)


if __name__ == "__main__":
    # Test reconstruction loss
    batch_size = 4
    channels = 10
    time_steps = 288
    
    # Create loss
    loss_fn = ReconstructionLoss()
    
    # Generate sample data
    x_original = torch.randn(batch_size, channels, time_steps)
    x_reconstructed = x_original + torch.randn_like(x_original) * 0.1  # Add noise
    
    # Compute loss
    loss = loss_fn(x_reconstructed, x_original)
    
    print(f"Original shape: {x_original.shape}")
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    print(f"Reconstruction loss: {loss.item():.6f}")
