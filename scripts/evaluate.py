"""
Complete End-to-End Evaluation Script

Evaluates the full IoT IDS pipeline:
1. Load trained adapter
2. Generate images from test set
3. Fine-tune vision backbone (DenseNet)
4. Evaluate on test set
5. Compute metrics with confidence intervals
6. Generate all visualizations

Usage:
    python scripts/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from models import AdapterDeepEncoder, VisionBackbone
from data import create_dataloaders
from training import generate_images_from_adapter, BackboneFinetuner
from evaluation import (
    evaluate_model,
    print_evaluation_results,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrix,
    visualize_generated_images,
    plot_comparison_table
)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    print("=" * 80)
    print("IoT IDS - COMPLETE EVALUATION PIPELINE")
    print("=" * 80)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed = config['random_seed']
    set_seed(seed)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[DEVICE] Using: {device}")
    
    # Create output directories
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    train_loader, val_loader, test_loader, num_features = create_dataloaders(config)
    
    # ========================================================================
    # STEP 2: Load Trained Adapter
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: LOADING TRAINED ADAPTER")
    print("=" * 80)
    
    adapter_checkpoint = os.path.join(checkpoints_dir, 'adapter_best.pth')
    
    if not os.path.exists(adapter_checkpoint):
        print(f"\n[ERROR] Adapter checkpoint not found: {adapter_checkpoint}")
        print("Please train the adapter first using: python scripts/train_adapter.py")
        return
    
    adapter_model = AdapterDeepEncoder(
        input_channels=num_features,
        num_classes=1,
        config=config,
        task='binary'
    )
    
    # Load checkpoint (handle both raw state_dict and full checkpoint)
    checkpoint = torch.load(adapter_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        adapter_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        adapter_model.load_state_dict(checkpoint)
    
    adapter_model.freeze_adapter()  # Freeze for image generation
    
    print(f"[SUCCESS] Loaded adapter from: {adapter_checkpoint}")
    
    # ========================================================================
    # STEP 3: Generate Images from Test Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING IMAGES FROM TEST SET")
    print("=" * 80)
    
    # Generate for train, val, test
    print("\nGenerating training images...")
    train_images, train_labels = generate_images_from_adapter(adapter_model, train_loader, device)
    
    print("Generating validation images...")
    val_images, val_labels = generate_images_from_adapter(adapter_model, val_loader, device)
    
    print("Generating test images...")
    test_images, test_labels = generate_images_from_adapter(adapter_model, test_loader, device)
    
    print(f"\n[SUCCESS] Generated images:")
    print(f"  Train: {train_images.shape}")
    print(f"  Val: {val_images.shape}")
    print(f"  Test: {test_images.shape}")
    
    # ========================================================================
    # STEP 4: Fine-tune Vision Backbone
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FINE-TUNING VISION BACKBONE (DenseNet121)")
    print("=" * 80)
    
    vision_model = VisionBackbone(
        backbone_name='densenet121',
        num_classes=1,
        task='binary',
        freeze_backbone=True
    )
    
    finetuner = BackboneFinetuner(vision_model, config, device=device)
    history = finetuner.finetune(train_images, train_labels, val_images, val_labels, 
                                 checkpoint_dir=checkpoints_dir)
    
    # Load best model
    best_checkpoint = os.path.join(checkpoints_dir, 'densenet121_best.pth')
    checkpoint = torch.load(best_checkpoint, map_location=device)
    vision_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n[SUCCESS] Fine-tuning complete!")
    print(f"  Best val accuracy: {checkpoint['val_acc']:.4f}")
    
    # ========================================================================
    # STEP 5: Evaluate on Test Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATING ON TEST SET")
    print("=" * 80)
    
    # Create test dataloader with generated images
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(test_images, test_labels)
    test_image_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate adapter + DenseNet
    class ImageDataLoaderWrapper:
        """Wrapper to match expected format."""
        def __init__(self, loader):
            self.loader = loader
        
        def __iter__(self):
            for images, labels in self.loader:
                yield {'image': images, 'label': labels.unsqueeze(1)}
        
        def __len__(self):
            return len(self.loader)
    
    wrapped_loader = ImageDataLoaderWrapper(test_image_loader)
    
    print("\nEvaluating Adapter + DenseNet121...")
    results_adapter = evaluate_model(vision_model, wrapped_loader, device=device, task='binary')
    
    print_evaluation_results(results_adapter, model_name='Adapter + DenseNet121')
    
    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Save all results
    results_dict = {
        'Adapter + DenseNet121': results_adapter
    }
    
    # ROC curve
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curves(results_dict, save_path=roc_path)
    
    # Precision-Recall curve
    prc_path = os.path.join(results_dir, 'precision_recall_curve.png')
    plot_precision_recall_curves(results_dict, save_path=prc_path)
    
    # Confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results_adapter['metrics']['confusion_matrix'], save_path=cm_path)
    
    # Generated images
    img_path = os.path.join(results_dir, 'generated_images.png')
    visualize_generated_images(adapter_model, test_loader, device=device, 
                               num_samples=16, save_path=img_path)
    
    # Comparison table
    table_path = os.path.join(results_dir, 'performance_table.png')
    plot_comparison_table(results_dict, save_path=table_path)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n✓ All results saved to: {results_dir}")
    print(f"\nKey Metrics:")
    print(f"  F1 Score:  {results_adapter['metrics']['f1']:.4f}")
    print(f"  AUC-ROC:   {results_adapter['metrics']['auc_roc']:.4f}")
    print(f"  AUC-PRC:   {results_adapter['metrics']['auc_prc']:.4f}")
    print(f"  Precision: {results_adapter['metrics']['precision']:.4f}")
    print(f"  Recall:    {results_adapter['metrics']['recall']:.4f}")
    
    print(f"\nGenerated Visualizations:")
    print(f"  - ROC curve: {roc_path}")
    print(f"  - Precision-Recall curve: {prc_path}")
    print(f"  - Confusion matrix: {cm_path}")
    print(f"  - Generated images: {img_path}")
    print(f"  - Performance table: {table_path}")
    
    print("\n" + "=" * 80)
    
    return results_adapter


if __name__ == "__main__":
    results = main()
