"""
Main Training Script for IoT IDS Adapter

Execute this script to train the complete adapter system.

Usage:
    python scripts/train_adapter.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from models import AdapterDeepEncoder
from losses import CompositeLoss
from data import create_dataloaders
from training.trainer import AdapterTrainer


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("=" * 80)
    print("IoT IDS - LEARNABLE CROSS-MODAL ADAPTER TRAINING")
    print("=" * 80)
    print("\nBased on:")
    print("'A Learnable Cross-Modal Adapter for Industrial Fault Detection")
    print("Using Pretrained Vision Models' - IEEE TII 2026")
    print("=" * 80)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n[CONFIG] Dataset: {config['dataset']['name']}")
    print(f"[CONFIG] Window size: {config['dataset']['window_size']}")
    print(f"[CONFIG] Batch size: {config['training']['adapter_training']['batch_size']}")
    print(f"[CONFIG] Learning rate: {config['training']['adapter_training']['learning_rate']}")
    print(f"[CONFIG] Max epochs: {config['training']['adapter_training']['epochs']}")
    print(f"[CONFIG] Early stopping patience: {config['training']['adapter_training']['early_stopping_patience']}")
    
    # Set random seed for reproducibility (Lines 407-412)
    seed = config['random_seed']
    set_seed(seed)
    print(f"\n[SEED] Random seed set to: {seed}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[DEVICE] Using: {device}")
    if device == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_loader, val_loader, test_loader, num_features = create_dataloaders(config)
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    
    model = AdapterDeepEncoder(
        input_channels=num_features,
        num_classes=1,  # Binary classification
        config=config,
        task='binary'
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    
    # Create loss
    criterion = CompositeLoss(config, task='binary')
    print(f"\n[LOSS] Composite loss weights:")
    print(f"  - lambda_rec (reconstruction): {criterion.lambda_rec}")
    print(f"  - lambda_tv (total variation): {criterion.lambda_tv}")
    print(f"  - lambda_cls (classification): {criterion.lambda_cls}")
    
    # Create trainer
    trainer = AdapterTrainer(model, criterion, config, device=device)
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    history = trainer.train(train_loader, val_loader, checkpoint_dir=checkpoint_dir)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'adapter_final.pth')
    trainer.save_model(final_model_path)
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(history['train_rec_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_rec_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss (L_rec)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total variation loss
        axes[1, 0].plot(history['train_tv_loss'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_tv_loss'], label='Val', linewidth=2)
        axes[1, 0].set_title('Total Variation Loss (L_tv)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Classification loss
        axes[1, 1].plot(history['train_cls_loss'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_cls_loss'], label='Val', linewidth=2)
        axes[1, 1].set_title('Classification Loss (L_cls)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        print(f"\n[PLOT] Training curves saved to results/training_curves.png")
        
    except Exception as e:
        print(f"\n[WARNING] Could not plot training curves: {e}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✓ Best model saved to: {os.path.join(checkpoint_dir, 'adapter_best.pth')}")
    print(f"✓ Final model saved to: {final_model_path}")
    print(f"✓ Best validation loss: {trainer.best_val_loss:.6f}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Fine-tune vision backbone (DenseNet/ResNet) on generated images")
    print("2. Evaluate on test set")
    print("3. Compare with baselines (InceptionTime, MiniRocket, etc.)")
    print("4. Generate visualizations of adapter outputs")
    
    return history


if __name__ == "__main__":
    history = main()
