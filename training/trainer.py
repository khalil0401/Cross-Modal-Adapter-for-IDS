"""
Training Module for Adapter Deep Encoder

Implements the complete training pipeline from Section V:
- Adam optimizer with lr=0.001
- Batch size=32, epochs=100
- Early stopping with patience=8
- Composite loss optimization

Reference: Lines 407-412
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import sys


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdapterTrainer:
    """
    Trainer for Adapter Deep Encoder system.
    
    From paper Section V (Lines 407-412):
    "The training configuration uses Adam with initial learning rate=0.001,
    batch size=32, epochs=100, with early stopping (patience=8), and seed=42."
    """
    
    def __init__(self, model, criterion, config, device='cuda'):
        """
        Args:
            model: AdapterDeepEncoder model
            criterion: CompositeLoss
            config (dict): Configuration dictionary
            device (str): 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # Training hyperparameters (Lines 407-412)
        self.learning_rate = config['training']['adapter_training']['learning_rate']
        self.epochs = config['training']['adapter_training']['epochs']
        self.patience = config['training']['adapter_training']['early_stopping_patience']
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_tv_loss': [],
            'train_cls_loss': [],
            'val_loss': [],
            'val_rec_loss': [],
            'val_tv_loss': [],
            'val_cls_loss': [],
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            dict: Average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'total_variation': 0.0,
            'classification': 0.0,
        }
        
        num_batches = 0
        
        for batch in tqdm(train_loader, desc='Training', leave=False):
            # Move data to device
            x = batch['data'].to(self.device)  # (batch, features, window_size)
            targets = batch['label'].to(self.device).squeeze()  # (batch,)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x, return_all=True)
            
            # Compute composite loss (Equation 3)
            losses = self.criterion(outputs, x, targets)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping (optional, for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader):
        """
        Validate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            dict: Average losses for validation
        """
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'total_variation': 0.0,
            'classification': 0.0,
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                x = batch['data'].to(self.device)
                targets = batch['label'].to(self.device).squeeze()
                
                # Forward pass
                outputs = self.model(x, return_all=True)
                
                # Compute loss
                losses = self.criterion(outputs, x, targets)
                
                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, train_loader, val_loader, checkpoint_dir='checkpoints'):
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            checkpoint_dir (str): Directory to save checkpoints
        
        Returns:
            dict: Training history
        """
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Early stopping patience: {self.patience}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_rec_loss'].append(train_losses['reconstruction'])
            self.history['train_tv_loss'].append(train_losses['total_variation'])
            self.history['train_cls_loss'].append(train_losses['classification'])
            
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_rec_loss'].append(val_losses['reconstruction'])
            self.history['val_tv_loss'].append(val_losses['total_variation'])
            self.history['val_cls_loss'].append(val_losses['classification'])
            
            # Print losses
            print(f"\nTrain Loss: {train_losses['total']:.6f} "
                  f"(Rec: {train_losses['reconstruction']:.6f}, "
                  f"TV: {train_losses['total_variation']:.6f}, "
                  f"Cls: {train_losses['classification']:.6f})")
            
            print(f"Val Loss:   {val_losses['total']:.6f} "
                  f"(Rec: {val_losses['reconstruction']:.6f}, "
                  f"TV: {val_losses['total_variation']:.6f}, "
                  f"Cls: {val_losses['classification']:.6f})")
            
            # Early stopping check
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, 'adapter_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'history': self.history,
                }, checkpoint_path)
                
                print(f"✓ New best model saved! Val loss: {val_losses['total']:.6f}")
            else:
                self.epochs_without_improvement += 1
                print(f"✗ No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with val loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def save_model(self, filepath):
        """Save model state."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model state."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test training setup
    import yaml
    from models import AdapterDeepEncoder
    from losses import CompositeLoss
    from data import create_dataloaders
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Create dataloaders  
    print("Loading data...")
    train_loader, val_loader, test_loader, num_features = create_dataloaders(config)
    
    # Create model
    print("\nCreating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AdapterDeepEncoder(
        input_channels=num_features,
        num_classes=1,
        config=config,
        task='binary'
    )
    
    # Create loss
    criterion = CompositeLoss(config, task='binary')
    
    # Create trainer
    trainer = AdapterTrainer(model, criterion, config, device=device)
    
    # Train
    print("\n" + "="*70)
    print("STARTING ADAPTER TRAINING")
    print("="*70)
    
    history = trainer.train(train_loader, val_loader)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
