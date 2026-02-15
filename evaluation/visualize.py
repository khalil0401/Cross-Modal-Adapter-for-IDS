"""
Visualization Module

Creates visualizations for evaluation results:
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Training curves
- Generated images comparison

Reference: Section V, Figures 3-6
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch


def plot_roc_curves(results_dict, save_path='roc_curves.png'):
    """
    Plot ROC curves for multiple models.
    
    From paper Figure 3:
    "ROC curves comparing adapter, fixed transformations, and baselines."
    
    Args:
        results_dict (dict): Dictionary mapping model names to evaluation results
        save_path (str): Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (model_name, results), color in zip(results_dict.items(), colors):
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - IoT IDS Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    
    return plt.gcf()


def plot_precision_recall_curves(results_dict, save_path='prc_curves.png'):
    """
    Plot Precision-Recall curves for multiple models.
    
    From paper Figure 4:
    "Precision-Recall curves showing adapter superiority in imbalanced scenarios."
    
    Args:
        results_dict (dict): Dictionary mapping model names to evaluation results
        save_path (str): Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (model_name, results), color in zip(results_dict.items(), colors):
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        prc_auc = auc(recall, precision)
        
        # Plot
        plt.plot(recall, precision, color=color, lw=2,
                label=f'{model_name} (AUC = {prc_auc:.3f})')
    
    # Baseline (prevalence)
    prevalence = np.mean(results_dict[list(results_dict.keys())[0]]['y_true'])
    plt.plot([0, 1], [prevalence, prevalence], 'k--', lw=1,
            label=f'Baseline (prevalence = {prevalence:.3f})')
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves - IoT IDS Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curves saved to {save_path}")
    
    return plt.gcf()


def plot_confusion_matrix(cm, class_names=None, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save figure
    """
    if class_names is None:
        if cm.shape[0] == 2:
            class_names = ['Benign', 'Attack']
        else:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def visualize_generated_images(adapter_model, dataloader, device='cuda', 
                               num_samples=16, save_path='generated_images.png'):
    """
    Visualize adapter-generated images.
    
    From paper Figure 5:
    "Examples of adapter-generated images for normal and attack traffic."
    
    Args:
        adapter_model: Trained AdapterDeepEncoder
        dataloader: DataLoader with time-series data
        device (str): 'cuda' or 'cpu'
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save figure
    """
    adapter_model.eval()
    adapter_model.to(device)
    
    # Get one batch
    batch = next(iter(dataloader))
    x = batch['data'][:num_samples].to(device)
    labels = batch['label'][:num_samples].cpu().numpy()
    
    # Generate images
    with torch.no_grad():
        images = adapter_model.generate_images(x)
    
    images = images.cpu().numpy()
    
    # Plot
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Convert from (3, 224, 224) to (224, 224, 3)
        img = images[i].transpose(1, 2, 0)
        
        axes[i].imshow(img)
        label_str = 'Attack' if labels[i] > 0.5 else 'Benign'
        axes[i].set_title(f'{label_str}', fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Adapter-Generated Images', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Generated images saved to {save_path}")
    
    return fig


def plot_comparison_table(results_dict, save_path='comparison_table.png'):
    """
    Create comparison table of all models.
    
    From paper Table II:
    "Performance comparison across all methods."
    
    Args:
        results_dict (dict): Dictionary mapping model names to evaluation results
        save_path (str): Path to save figure
    """
    # Extract metrics
    data = []
    for model_name, results in results_dict.items():
        metrics = results['metrics']
        ci = results['confidence_intervals']
        
        row = {
            'Model': model_name,
            'Precision': f"{metrics['precision']:.3f} ± {ci['precision']['std']:.3f}",
            'Recall': f"{metrics['recall']:.3f} ± {ci['recall']['std']:.3f}",
            'F1': f"{metrics['f1']:.3f} ± {ci['f1']['std']:.3f}",
            'AUC-ROC': f"{metrics.get('auc_roc', 0):.3f}",
            'AUC-PRC': f"{metrics.get('auc_prc', 0):.3f}",
        }
        data.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[row['Model'], row['Precision'], row['Recall'], 
                  row['F1'], row['AUC-ROC'], row['AUC-PRC']] for row in data]
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PRC'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('Performance Comparison - All Models', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison table saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization
    np.random.seed(42)
    
    # Generate sample results
    n_samples = 1000
    
    results_dict = {}
    model_names = ['Adapter (Ours)', 'InceptionTime', 'Shape Modality', 'GAF']
    
    for i, name in enumerate(model_names):
        # Simulate different performance levels
        y_true = np.random.randint(0, 2, n_samples)
        noise = 0.2 + i * 0.1  # Increasing noise for baselines
        y_prob = 0.8 * y_true + noise * np.random.rand(n_samples)
        y_prob = np.clip(y_prob, 0, 1)
        y_pred = (y_prob > 0.5).astype(int)
        
        from metrics import compute_metrics, compute_confidence_intervals
        
        metrics = compute_metrics(y_true, y_pred, y_prob, task='binary')
        ci = compute_confidence_intervals(y_true, y_pred, y_prob, n_bootstrap=100)
        
        results_dict[name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics,
            'confidence_intervals': ci
        }
    
    # Create visualizations
    print("Creating visualizations...")
    
    plot_roc_curves(results_dict, 'test_roc.png')
    plot_precision_recall_curves(results_dict, 'test_prc.png')
    plot_confusion_matrix(results_dict['Adapter (Ours)']['metrics']['confusion_matrix'], 
                         save_path='test_cm.png')
    plot_comparison_table(results_dict, 'test_table.png')
    
    print("\nAll visualizations created successfully!")
