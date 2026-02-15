"""
Complete Adapter Deep Encoder System

Integrates all components:
- Encoder
- Decoder (TS reconstruction)
- Image Adapter
- Latent Classifier
- Self-Attention

Reference: Section III.C, Fig. 2
"""

import torch
import torch.nn as nn
from .encoder import Encoder, SelfAttention
from .decoder import Decoder
from .adapter import ImageAdapter
from .classifier import LatentClassifier


class AdapterDeepEncoder(nn.Module):
    """
    Complete cross-modal adapter system for IoT IDS.
    
    From paper Fig. 2 and Lines 244-321:
    "The encoder maps the input TS to a latent code z, from which three branches
    operate in parallel:
    (1) A decoder mirrors the encoder for TS reconstruction
    (2) An adapter that maps z to a 7×7×128 seed, progressively upsampled to 224×224×3
    (3) A latent classification head enforcing discriminative structure"
    
    This architecture enables:
    - Learning task-optimal TS representations
    - Generating vision-model-compatible images
    - Multitask optimization (reconstruction + smoothness + classification)
    """
    
    def __init__(self, input_channels, num_classes, config, task='binary'):
        """
        Args:
            input_channels (int): Number of input TS features
            num_classes (int): Number of output classes
            config (dict): Configuration dictionary
            task (str): 'binary' or 'multiclass'
        """
        super(AdapterDeepEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.task = task
        self.config = config
        
        latent_dim = config['adapter']['encoder']['filters'][-1]  # 128
        
        # Component 1: Encoder (Lines 247-254)
        self.encoder = Encoder(input_channels, config)
        
        # Component 2: Self-Attention (Lines 291-294)
        self.attention = SelfAttention(latent_dim)
        
        # Component 3: TS Reconstruction Decoder (Lines 297-300)
        self.decoder = Decoder(latent_dim, input_channels, config)
        
        # Component 4: Image Adapter (Lines 301-321)
        self.adapter = ImageAdapter(latent_dim, config)
        
        # Component 5: Latent Classification Head (Lines 294-297)
        self.classifier = LatentClassifier(latent_dim, num_classes, task)
    
    def forward(self, x, return_all=False):
        """
        Forward pass through complete adapter system.
        
        Args:
            x (torch.Tensor): Input time-series of shape (batch, channels, time_steps)
            return_all (bool): If True, return all intermediate outputs
        
        Returns:
            If return_all=False:
                dict: {
                    'image': Generated image (batch, 3, 224, 224),
                    'classification': Classification output
                }
            If return_all=True:
                dict: {
                    'latent': Latent representation,
                    'latent_attended': Attention-refined latent,
                    'reconstruction': Reconstructed TS,
                    'image': Generated image,
                    'classification': Classification output
                }
        """
        # Step 1: Encode input TS to latent representation
        latent = self.encoder(x)  # (batch, 128, time_steps/16)
        
        # Step 2: Apply self-attention
        latent_attended = self.attention(latent)
        
        # Step 3: Three parallel branches
        
        # Branch 1: TS Reconstruction
        x_reconstructed = self.decoder(latent)
        
        # Branch 2: Image Generation
        image = self.adapter(latent)
        
        # Branch 3: Latent Classification
        classification = self.classifier(latent_attended)
        
        if return_all:
            return {
                'latent': latent,
                'latent_attended': latent_attended,
                'reconstruction': x_reconstructed,
                'image': image,
                'classification': classification
            }
        else:
            return {
                'image': image,
                'classification': classification,
                'reconstruction': x_reconstructed
            }
    
    def freeze_adapter(self):
        """
        Freeze adapter weights after training.
        Used before fine-tuning vision backbone (Lines 316-318).
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_adapter(self):
        """Unfreeze adapter for further training."""
        for param in self.parameters():
            param.requires_grad = True
    
    def generate_images(self, x):
        """
        Generate images from input TS (inference mode).
        
        Args:
            x (torch.Tensor): Input time-series (batch, channels, time_steps)
        
        Returns:
            torch.Tensor: Generated images (batch, 3, 224, 224)
        """
        with torch.no_grad():
            latent = self.encoder(x)
            image = self.adapter(latent)
        return image


if __name__ == "__main__":
    # Test complete adapter system
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    input_channels = 10  # IoT features
    num_classes = 1  # Binary classification
    batch_size = 4
    time_steps = 288
    
    # Create adapter system
    print("=== Creating Adapter Deep Encoder ===")
    adapter_system = AdapterDeepEncoder(input_channels, num_classes, config, task='binary')
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter_system.parameters())
    trainable_params = sum(p.numel() for p in adapter_system.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    x = torch.randn(batch_size, input_channels, time_steps)
    
    outputs = adapter_system(x, return_all=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {outputs['latent'].shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"Image shape: {outputs['image'].shape}")
    print(f"Classification shape: {outputs['classification'].shape}")
    
    # Verify image is input-compliant
    expected_image_shape = torch.Size([batch_size, 3, 224, 224])
    print(f"\nImage is input-compliant with pretrained models: {outputs['image'].shape == expected_image_shape}")
    
    # Test image generation (inference)
    print("\n=== Testing Image Generation (Inference) ===")
    adapter_system.eval()
    images = adapter_system.generate_images(x)
    print(f"Generated images shape: {images.shape}")
