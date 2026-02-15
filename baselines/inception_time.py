"""
InceptionTime Baseline

Deep learning architecture for time-series classification.
Uses Inception modules with multi-scale temporal convolutions.

Reference: Paper baselines (Table II)
From: Ismail Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification", 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Inception module with parallel convolutions at different scales.
    """
    
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionModule, self).__init__()
        
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        
        # Parallel convolutions with different kernel sizes
        self.conv_list = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, n_filters, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        
        # MaxPooling branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(n_filters * len(kernel_sizes) + n_filters)
        
    def forward(self, x):
        # Bottleneck
        x_bottleneck = self.bottleneck(x)
        
        # Parallel convolutions
        conv_outputs = [conv(x_bottleneck) for conv in self.conv_list]
        
        # MaxPool branch
        x_maxpool = self.maxpool(x)
        x_maxpool = self.maxpool_conv(x_maxpool)
        
        # Concatenate all branches
        out = torch.cat(conv_outputs + [x_maxpool], dim=1)
        out = self.bn(out)
        out = F.relu(out)
        
        return out


class InceptionBlock(nn.Module):
    """
    Inception block with residual connection.
    """
    
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionBlock, self).__init__()
        
        self.inception = InceptionModule(in_channels, n_filters, kernel_sizes, bottleneck_channels)
        
        # Output channels after concatenation
        out_channels = n_filters * (len(kernel_sizes) + 1)
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.out_channels = out_channels
        
    def forward(self, x):
        inception_out = self.inception(x)
        residual_out = self.residual(x)
        
        out = inception_out + residual_out
        out = F.relu(out)
        
        return out


class InceptionTime(nn.Module):
    """
    InceptionTime for time-series classification.
    
    From paper baselines (Table II):
    "InceptionTime achieves F1 ~88.6% on average across datasets,
    serving as the strongest time-series baseline."
    """
    
    def __init__(self, input_channels, num_classes=1, task='binary', 
                 n_filters=32, depth=6, kernel_sizes=[9, 19, 39]):
        """
        Args:
            input_channels (int): Number of input features
            num_classes (int): Number of output classes
            task (str): 'binary' or 'multiclass'
            n_filters (int): Number of filters per conv layer
            depth (int): Number of Inception blocks
            kernel_sizes (list): Kernel sizes for parallel convolutions
        """
        super(InceptionTime, self).__init__()
        
        self.task = task
        self.num_classes = num_classes
        
        # Stack of Inception blocks
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        
        for i in range(depth):
            block = InceptionBlock(in_ch, n_filters, kernel_sizes)
            self.blocks.append(block)
            in_ch = block.out_channels
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Linear(in_ch, num_classes if task == 'multiclass' else 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input time-series (batch, channels, time_steps)
        
        Returns:
            torch.Tensor: Logits
        """
        # Pass through Inception blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        
        # Classification
        out = self.fc(x)
        
        return out


if __name__ == "__main__":
    # Test InceptionTime
    batch_size = 4
    input_channels = 10
    time_steps = 288
    
    model = InceptionTime(input_channels, num_classes=1, task='binary')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"InceptionTime parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_channels, time_steps)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output: ({batch_size}, 1)")
