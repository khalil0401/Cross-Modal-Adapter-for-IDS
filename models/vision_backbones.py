"""
Vision Backbone Models

Integration of pretrained vision models from torchvision:
- DenseNet121 (primary)
- ResNet18
- EfficientNet-B0
- MobileNetV3
- ViT-B16

Reference: Section IV, Lines 336-345
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights
)


class VisionBackbone(nn.Module):
    """
    Wrapper for pretrained vision models with custom classification head.
    
    From paper Lines 336-345:
    "We evaluate the adapter with several pretrained vision backbones:
    DenseNet121, ResNet18, EfficientNet-B0, MobileNetV3-Small, and ViT-B/16.
    All models are pretrained on ImageNet-1K. The final classification layer
    is replaced to match the task (binary or multiclass)."
    
    Key principle: NO INTERNAL MODIFICATIONS to backbone architecture.
    """
    
    def __init__(self, backbone_name='densenet121', num_classes=1, task='binary', freeze_backbone=False):
        """
        Args:
            backbone_name (str): Name of pretrained model
            num_classes (int): Number of output classes
            task (str): 'binary' or 'multiclass'
            freeze_backbone (bool): If True, freeze all layers except classifier
        """
        super(VisionBackbone, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.task = task
        
        # Load pretrained model
        if backbone_name == 'densenet121':
            self.backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes if task == 'multiclass' else 1)
            
        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes if task == 'multiclass' else 1)
            
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes if task == 'multiclass' else 1)
            
        elif backbone_name == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(in_features, num_classes if task == 'multiclass' else 1)
            
        elif backbone_name == 'vit_b16':
            self.backbone = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes if task == 'multiclass' else 1)
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Freeze backbone if specified (Lines 354-357)
        if freeze_backbone:
            self.freeze_backbone_layers()
    
    def freeze_backbone_layers(self):
        """
        Freeze all layers except the final classification head.
        Used during initial fine-tuning phase.
        """
        for name, param in self.backbone.named_parameters():
            # Only keep classifier trainable
            if 'classifier' not in name and 'fc' not in name and 'head' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all layers for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through vision backbone.
        
        Args:
            x (torch.Tensor): Input images (batch, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits (batch, num_classes) or (batch, 1) for binary
        """
        return self.backbone(x)


def create_vision_model(config, num_classes=1, task='binary'):
    """
    Factory function to create vision backbone from config.
    
    Args:
        config (dict): Configuration dictionary
        num_classes (int): Number of classes
        task (str): 'binary' or 'multiclass'
    
    Returns:
        VisionBackbone: Pretrained vision model
    """
    backbone_name = config['vision_backbone']['model']
    freeze_backbone = config['training']['backbone_finetuning'].get('freeze_backbone_initially', True)
    
    model = VisionBackbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        task=task,
        freeze_backbone=freeze_backbone
    )
    
    return model


if __name__ == "__main__":
    # Test vision backbones
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("Testing Vision Backbones")
    print("=" * 70)
    
    # Test all backbones
    backbones = ['densenet121', 'resnet18', 'efficientnet_b0', 'mobilenet_v3', 'vit_b16']
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    for backbone_name in backbones:
        print(f"\n[{backbone_name.upper()}]")
        
        model = VisionBackbone(backbone_name=backbone_name, num_classes=1, task='binary')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(images)
        
        print(f"Input shape: {images.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test freezing
        model.freeze_backbone_layers()
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable after freeze: {trainable_after_freeze:,}")
