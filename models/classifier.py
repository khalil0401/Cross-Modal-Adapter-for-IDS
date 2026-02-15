"""
Latent Classification Head

Implementation from Section III.C, Fig. 2:
- Classification from latent representation
- FC layers: 64 → 32 → output
- Uses global average pooling on latent features

Reference: Lines 294-297 of the paper
"""

import torch
import torch.nn as nn


class LatentClassifier(nn.Module):
    """
    Classification head that operates on latent representation.
    
    From paper Lines 294-297:
    "The refined latent representation is aggregated via global average pooling
    and fed into the classification head. It comprises a sequential stack of
    fully connected layers (64 and 32) to produce a classification probability."
    
    Architecture:
    - Global Average Pooling on latent features
    - FC(latent_dim, 64) → ReLU → Dropout
    - FC(64, 32) → ReLU → Dropout
    - FC(32, num_classes) → Sigmoid/Softmax
    """
    
    def __init__(self, latent_dim, num_classes, task='binary', dropout=0.3):
        """
        Args:
            latent_dim (int): Latent dimension (128)
            num_classes (int): Number of output classes (1 for binary, >1 for multiclass)
            task (str): 'binary' for fault detection, 'multiclass' for fault diagnosis
            dropout (float): Dropout probability
        """
        super(LatentClassifier, self).__init__()
        
        self.task = task
        self.num_classes = num_classes
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # FC layers: 64 → 32 (Lines 295-297)
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes if task == 'multiclass' else 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Output activation
        if task == 'binary':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Softmax(dim=1)
    
    def forward(self, z):
        """
        Forward pass through classifier.
        
        Args:
            z (torch.Tensor): Latent representation of shape (batch, latent_dim, time_steps)
        
        Returns:
            torch.Tensor: Classification probabilities
                - Binary: (batch, 1)
                - Multiclass: (batch, num_classes)
        """
        # Global average pooling (Lines 294-295)
        z_pooled = self.gap(z).squeeze(-1)  # (batch, latent_dim)
        
        # FC layers
        x = self.fc1(z_pooled)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # Output activation
        if self.training:
            # During training, return logits for loss computation
            return x
        else:
            # During inference, return probabilities
            return self.output_activation(x)


if __name__ == "__main__":
    # Test classifier
    latent_dim = 128
    batch_size = 4
    time_steps = 18
    
    # Binary classification
    print("=== Binary Classification ===")
    classifier_binary = LatentClassifier(latent_dim, num_classes=1, task='binary')
    z = torch.randn(batch_size, latent_dim, time_steps)
    
    classifier_binary.eval()
    output = classifier_binary(z)
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Multiclass classification
    print("\n=== Multiclass Classification ===")
    num_attack_types = 5
    classifier_multi = LatentClassifier(latent_dim, num_classes=num_attack_types, task='multiclass')
    
    classifier_multi.eval()
    output_multi = classifier_multi(z)
    print(f"Output shape: {output_multi.shape}")
    print(f"Probabilities sum: {output_multi.sum(dim=1)}")  # Should be ~1 for each sample
