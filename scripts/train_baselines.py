"""
Baseline Model Training Script

Trains all baseline models for comparison with the adapter:
- InceptionTime (TS baseline)
- LSTM with Attention (TS baseline)
- GAF + DenseNet (fixed transformation)
- RP + DenseNet (fixed transformation)
- MTF + DenseNet (fixed transformation)

Usage:
    python scripts/train_baselines.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import create_dataloaders
from baselines import InceptionTime, LSTMAttention, transform_batch_to_images
from models import VisionBackbone
from evaluation import evaluate_model, print_evaluation_results


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_ts_baseline(model, train_loader, val_loader, config, device, model_name):
    """
    Train a time-series baseline model (InceptionTime, LSTM).
    
    Args:
        model: Baseline model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config (dict): Configuration
        device (str): Device
        model_name (str): Name of the model
    
    Returns:
        model: Trained model
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['adapter_training']['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = config['training']['adapter_training']['epochs']
    patience = config['training']['adapter_training']['early_stopping_patience']
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            x = batch['data'].to(device)
            targets = batch['label'].to(device).squeeze().float()
            
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['data'].to(device)
                targets = batch['label'].to(device).squeeze().float()
                
                outputs = model(x).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model)
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    
    return model


def main():
    print("=" * 80)
    print("TRAINING ALL BASELINE MODELS")
    print("=" * 80)
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed = config['random_seed']
    set_seed(seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[DEVICE] Using: {device}")
    
    # Directories
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_loader, val_loader, test_loader, num_features = create_dataloaders(config)
    
    baseline_results = {}
    
    # ========================================================================
    # 1. InceptionTime
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE 1: InceptionTime")
    print("=" * 80)
    
    inception_model = InceptionTime(num_features, num_classes=1, task='binary')
    inception_model = train_ts_baseline(inception_model, train_loader, val_loader, 
                                       config, device, 'InceptionTime')
    
    # Evaluate
    results_inception = evaluate_model(inception_model, test_loader, device=device, task='binary')
    print_evaluation_results(results_inception, model_name='InceptionTime')
    baseline_results['InceptionTime'] = results_inception
    
    # Save
    torch.save(inception_model.state_dict(), 
              os.path.join(checkpoints_dir, 'inception_time.pth'))
    
    # ========================================================================
    # 2. LSTM with Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE 2: LSTM with Attention")
    print("=" * 80)
    
    lstm_model = LSTMAttention(num_features, num_classes=1, task='binary')
    lstm_model = train_ts_baseline(lstm_model, train_loader, val_loader,
                                   config, device, 'LSTM-Attention')
    
    # Evaluate
    results_lstm = evaluate_model(lstm_model, test_loader, device=device, task='binary')
    print_evaluation_results(results_lstm, model_name='LSTM-Attention')
    baseline_results['LSTM-Attention'] = results_lstm
    
    # Save
    torch.save(lstm_model.state_dict(),
              os.path.join(checkpoints_dir, 'lstm_attention.pth'))
    
    # ========================================================================
    # 3. Fixed Transformations + DenseNet
    # ========================================================================
    for method in ['gaf', 'rp', 'mtf']:
        print("\n" + "=" * 80)
        print(f"BASELINE: {method.upper()} + DenseNet121")
        print("=" * 80)
        
        # Generate images
        print(f"\nGenerating {method.upper()} images...")
        train_images, train_labels = transform_batch_to_images(train_loader, method=method)
        val_images, val_labels = transform_batch_to_images(val_loader, method=method)
        test_images, test_labels = transform_batch_to_images(test_loader, method=method)
        
        # Create vision model
        vision_model = VisionBackbone('densenet121', num_classes=1, task='binary', freeze_backbone=True)
        
        # Fine-tune (simplified version)
        from training import BackboneFinetuner
        
        finetuner = BackboneFinetuner(vision_model, config, device=device)
        history = finetuner.finetune(train_images, train_labels, val_images, val_labels,
                                     checkpoint_dir=checkpoints_dir)
        
        # Load best model
        best_checkpoint = os.path.join(checkpoints_dir, 'densenet121_best.pth')
        checkpoint = torch.load(best_checkpoint, map_location=device)
        vision_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_dataset = TensorDataset(test_images, test_labels)
        test_image_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        class ImageDataLoaderWrapper:
            def __init__(self, loader):
                self.loader = loader
            
            def __iter__(self):
                for images, labels in self.loader:
                    yield {'image': images, 'label': labels.unsqueeze(1)}
            
            def __len__(self):
                return len(self.loader)
        
        wrapped_loader = ImageDataLoaderWrapper(test_image_loader)
        results = evaluate_model(vision_model, wrapped_loader, device=device, task='binary')
        print_evaluation_results(results, model_name=f'{method.upper()} + DenseNet121')
        baseline_results[f'{method.upper()} + DenseNet121'] = results
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for model_name, results in baseline_results.items():
        print(f"\n{model_name}:")
        print(f"  F1 Score:  {results['metrics']['f1']:.4f}")
        print(f"  AUC-ROC:   {results['metrics']['auc_roc']:.4f}")
        print(f"  Precision: {results['metrics']['precision']:.4f}")
        print(f"  Recall:    {results['metrics']['recall']:.4f}")
    
    # Save results
    import pickle
    with open(os.path.join(results_dir, 'baseline_results.pkl'), 'wb') as f:
        pickle.dump(baseline_results, f)
    
    print(f"\nAll baseline results saved to: {results_dir}/baseline_results.pkl")
    
    return baseline_results


if __name__ == "__main__":
    results = main()
