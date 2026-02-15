"""
Vision Backbone Fine-tuning Pipeline

Fine-tunes pretrained vision models on adapter-generated images.

From paper Lines 354-361:
"The adapter is first trained end-to-end with composite loss.
Then, the adapter weights are frozen, and we generate images for
the entire dataset. These images are used to fine-tune the vision
backbone's classification layer (lr=0.001, 20 epochs)."

Reference: Section V, Algorithm 2 (implicit)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GeneratedImageDataset(Dataset):
    """
    Dataset of adapter-generated images for vision backbone fine-tuning.
    """
    
    def __init__(self, images, labels):
        """
        Args:
            images (torch.Tensor): Generated images (N, 3, 224, 224)
            labels (torch.Tensor): Labels (N,)
        """
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx]
        }


class BackboneFinetuner:
    """
    Fine-tuning trainer for vision backbones.
    
    From paper Lines 354-361:
    "Fine-tuning uses Adam optimizer with lr=0.001 for 20 epochs.
    Only the classification head is updated initially, then full
    fine-tuning if needed."
    """
    
    def __init__(self, vision_model, config, device='cuda'):
        """
        Args:
            vision_model: VisionBackbone model
            config (dict): Configuration dictionary
            device (str): 'cuda' or 'cpu'
        """
        self.model = vision_model.to(device)
        self.config = config
        self.device = device
        
        # Fine-tuning hyperparameters
        self.learning_rate = config['training']['backbone_finetuning']['learning_rate']
        self.epochs = config['training']['backbone_finetuning']['epochs']
        self.batch_size = config['training']['backbone_finetuning']['batch_size']
        
        # Loss function
        if self.model.task == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc='Fine-tuning', leave=False):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            if self.model.task == 'binary':
                labels = labels.float()
                if outputs.dim() == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, labels)
                
                # Predictions
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                labels = labels.long()
                loss = self.criterion(outputs, labels)
                
                # Predictions
                _, preds = torch.max(outputs, 1)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                if self.model.task == 'binary':
                    labels = labels.float()
                    if outputs.dim() == 2 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)
                    loss = self.criterion(outputs, labels)
                    
                    # Predictions
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                else:
                    labels = labels.long()
                    loss = self.criterion(outputs, labels)
                    
                    # Predictions
                    _, preds = torch.max(outputs, 1)
                
                # Accumulate metrics
                epoch_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = epoch_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def finetune(self, train_images, train_labels, val_images, val_labels, checkpoint_dir='checkpoints'):
        """
        Complete fine-tuning loop.
        
        Args:
            train_images (torch.Tensor): Training images (N, 3, 224, 224)
            train_labels (torch.Tensor): Training labels (N,)
            val_images (torch.Tensor): Validation images (M, 3, 224, 224)
            val_labels (torch.Tensor): Validation labels (M,)
            checkpoint_dir (str): Directory to save checkpoints
        
        Returns:
            dict: Training history
        """
        print(f"Starting fine-tuning for {self.epochs} epochs...")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        
        # Create datasets
        train_dataset = GeneratedImageDataset(train_images, train_labels)
        val_dataset = GeneratedImageDataset(val_images, val_labels)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.6f}, Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, f'{self.model.backbone_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history,
                }, checkpoint_path)
                print(f"✓ New best model saved! Val acc: {val_acc:.4f}")
        
        print(f"\nFine-tuning complete! Best val acc: {best_val_acc:.4f}")
        
        return self.history


def generate_images_from_adapter(adapter_model, dataloader, device='cuda'):
    """
    Generate images for entire dataset using trained adapter.
    
    Args:
        adapter_model: Trained AdapterDeepEncoder
        dataloader: DataLoader with original time-series
        device (str): 'cuda' or 'cpu'
    
    Returns:
        tuple: (images, labels)
    """
    adapter_model.eval()
    adapter_model.to(device)
    
    all_images = []
    all_labels = []
    
    print("Generating images from adapter...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating images'):
            x = batch['data'].to(device)
            labels = batch['label']
            
            # Generate images
            images = adapter_model.generate_images(x)
            
            all_images.append(images.cpu())
            all_labels.append(labels)
    
    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0).squeeze()
    
    print(f"Generated {len(all_images)} images")
    
    return all_images, all_labels


if __name__ == "__main__":
    # Test fine-tuning
    import yaml
    from models import VisionBackbone
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create vision model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vision_model = VisionBackbone('densenet121', num_classes=1, task='binary', freeze_backbone=True)
    
    # Generate dummy images and labels
    train_images = torch.randn(100, 3, 224, 224)
    train_labels = torch.randint(0, 2, (100,)).float()
    val_images = torch.randn(20, 3, 224, 224)
    val_labels = torch.randint(0, 2, (20,)).float()
    
    # Fine-tune
    finetuner = BackboneFinetuner(vision_model, config, device=device)
    history = finetuner.finetune(train_images, train_labels, val_images, val_labels)
    
    print("\nTraining history:")
    print(f"Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"Final val acc: {history['val_acc'][-1]:.4f}")
