"""
Total Variation Loss (L_tv)

Implementation of Equation 5 from the paper:
Encourages spatial smoothness in generated images

Reference: Lines 269-280
"""

import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss for spatial smoothness of generated images.
    
    From paper Equation 5 (Lines 269-280):
    "The total-variation loss encourages spatial smoothness. Let I ∈ R^(H×W×K)
    be the adapter-generated image before it is fed to the vision backbone."
    
    L_tv = (1 / |Ω|K) * Σ_{i,j,k} |I_{i+1,j,k} - I_{i,j,k}| + |I_{i,j+1,k} - I_{i,j,k}|
    
    where:
    - H, W, K: height, width, channels
    - Ω = {1,...,H-1} × {1,...,W-1}
    - |Ω| = (H-1) * (W-1)
    """
    
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, images):
        """
        Compute total variation loss.
        
        Args:
            images (torch.Tensor): Generated images of shape (batch, K, H, W)
        
        Returns:
            torch.Tensor: Scalar total variation loss
        """
        batch_size, K, H, W = images.shape
        
        # Compute differences (Equation 5, Lines 272-275)
        # Vertical differences: |I_{i+1,j,k} - I_{i,j,k}|
        diff_i = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        
        # Horizontal differences: |I_{i,j+1,k} - I_{i,j,k}|
        diff_j = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        
        # Sum of differences
        tv_sum = diff_i.sum() + diff_j.sum()
        
        # Normalize by |Ω| * K (Lines 272, 280)
        omega_size = (H - 1) * (W - 1)
        normalization = omega_size * K * batch_size
        
        tv_loss = tv_sum / normalization
        
        return tv_loss


if __name__ == "__main__":
    # Test total variation loss
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    
    # Create loss
    loss_fn = TotalVariationLoss()
    
    # Test 1: Smooth image (low TV)
    smooth_image = torch.ones(batch_size, channels, height, width) * 0.5
    tv_smooth = loss_fn(smooth_image)
    print(f"Smooth image TV loss: {tv_smooth.item():.6f}")
    
    # Test 2: Noisy image (high TV)
    noisy_image = torch.rand(batch_size, channels, height, width)
    tv_noisy = loss_fn(noisy_image)
    print(f"Noisy image TV loss: {tv_noisy.item():.6f}")
    
    # Test 3: Gradient image (medium TV)
    gradient_image = torch.linspace(0, 1, height * width).view(1, 1, height, width).repeat(batch_size, channels, 1, 1)
    tv_gradient = loss_fn(gradient_image)
    print(f"Gradient image TV loss: {tv_gradient.item():.6f}")
    
    print(f"\nExpected: TV(smooth) < TV(gradient) < TV(noisy)")
    print(f"Actual: {tv_smooth.item():.6f} < {tv_gradient.item():.6f} < {tv_noisy.item():.6f}")
