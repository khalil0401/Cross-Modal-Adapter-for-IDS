"""
Composite Multitask Loss

Implementation of Equation 3 from the paper:
L_total = λ_rec * L_rec + λ_tv * L_tv + λ_cls * L_cls

Reference: Lines 256-262
"""

import torch
import torch.nn as nn
from .reconstruction_loss import ReconstructionLoss
from .total_variation_loss import TotalVariationLoss
from .classification_loss import ClassificationLoss


class CompositeLoss(nn.Module):
    """
    Composite multitask loss for adapter training.
    
    From paper Equation 3 (Lines 256-262):
    "The adapter deep encoder uses a multitask training objective,
    which is a composite loss described as follows:
    
    L_total = λ_rec * L_rec + λ_tv * L_tv + λ_cls * L_cls
    
    where L_rec, L_tv, and L_cls denote the reconstruction, total-variation,
    and classification losses, respectively, while λ_rec, λ_tv, and λ_cls are
    scalar weighting coefficients controlling their relative contributions."
    
    From Lines 410-412:
    λ_rec = 0.5
    λ_tv = 0.1
    λ_cls = 0.4
    """
    
    def __init__(self, config, task='binary'):
        """
        Args:
            config (dict): Configuration dictionary with loss weights
            task (str): 'binary' or 'multiclass'
        """
        super(CompositeLoss, self).__init__()
        
        # Extract loss weights from config (Lines 410-412)
        self.lambda_rec = config['loss']['lambda_rec']  # 0.5
        self.lambda_tv = config['loss']['lambda_tv']    # 0.1
        self.lambda_cls = config['loss']['lambda_cls']  # 0.4
        
        # Initialize individual loss functions
        self.reconstruction_loss = ReconstructionLoss()
        self.total_variation_loss = TotalVariationLoss()
        self.classification_loss = ClassificationLoss(task=task)
    
    def forward(self, outputs, x_original, targets):
        """
        Compute composite multitask loss.
        
        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'reconstruction': Reconstructed TS (batch, D, N)
                - 'image': Generated image (batch, 3, 224, 224)
                - 'classification': Classification predictions
            x_original (torch.Tensor): Original input TS (batch, D, N)
            targets (torch.Tensor): Classification targets
        
        Returns:
            dict: Dictionary containing:
                - 'total': Total composite loss
                - 'reconstruction': Individual reconstruction loss
                - 'total_variation': Individual TV loss
                - 'classification': Individual classification loss
        """
        # Compute individual losses
        L_rec = self.reconstruction_loss(outputs['reconstruction'], x_original)
        L_tv = self.total_variation_loss(outputs['image'])
        L_cls = self.classification_loss(outputs['classification'], targets)
        
        # Compute weighted composite loss (Equation 3)
        L_total = (self.lambda_rec * L_rec +
                   self.lambda_tv * L_tv +
                   self.lambda_cls * L_cls)
        
        # Return all losses for logging
        return {
            'total': L_total,
            'reconstruction': L_rec,
            'total_variation': L_tv,
            'classification': L_cls
        }


if __name__ == "__main__":
    # Test composite loss
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create loss
    loss_fn = CompositeLoss(config, task='binary')
    
    # Generate sample data
    batch_size = 4
    channels = 10
    time_steps = 288
    
    x_original = torch.randn(batch_size, channels, time_steps)
    x_reconstructed = x_original + torch.randn_like(x_original) * 0.1
    image = torch.rand(batch_size, 3, 224, 224)
    classification = torch.randn(batch_size, 1)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    outputs = {
        'reconstruction': x_reconstructed,
        'image': image,
        'classification': classification
    }
    
    # Compute loss
    losses = loss_fn(outputs, x_original, targets)
    
    print("=== Composite Loss ===")
    print(f"Total loss: {losses['total'].item():.6f}")
    print(f"\nIndividual losses:")
    print(f"  Reconstruction (λ={loss_fn.lambda_rec}): {losses['reconstruction'].item():.6f}")
    print(f"  Total Variation (λ={loss_fn.lambda_tv}): {losses['total_variation'].item():.6f}")
    print(f"  Classification (λ={loss_fn.lambda_cls}): {losses['classification'].item():.6f}")
    print(f"\nWeighted contributions:")
    print(f"  Reconstruction: {(loss_fn.lambda_rec * losses['reconstruction']).item():.6f}")
    print(f"  Total Variation: {(loss_fn.lambda_tv * losses['total_variation']).item():.6f}")
    print(f"  Classification: {(loss_fn.lambda_cls * losses['classification']).item():.6f}")
    print(f"  Sum: {losses['total'].item():.6f}")
