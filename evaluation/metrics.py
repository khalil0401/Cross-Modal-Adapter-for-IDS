"""
Evaluation Metrics Module

Implements metrics from paper Section V:
- Precision, Recall, F1 Score
- AUC-ROC, AUC-PRC
- 95% Confidence Intervals
- Confusion Matrix

Reference: Lines 465-472
"""

import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, y_prob=None, task='binary'):
    """
    Compute comprehensive evaluation metrics.
    
    From paper Lines 465-472:
    "We evaluate using precision, recall, F1-score, AUC-ROC, and AUC-PRC.
    For binary classification, we report metrics for the positive class (attack).
    Confidence intervals are computed using bootstrap resampling (n=1000, α=0.05)."
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray): Predicted probabilities (for AUC)
        task (str): 'binary' or 'multiclass'
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    if task == 'binary':
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    else:
        # Macro-average for multiclass
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC metrics (require probabilities)
    if y_prob is not None:
        try:
            if task == 'binary':
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['auc_prc'] = average_precision_score(y_true, y_prob)
            else:
                # One-vs-rest for multiclass
                from sklearn.preprocessing import label_binarize
                n_classes = len(np.unique(y_true))
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                metrics['auc_roc'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
                metrics['auc_prc'] = average_precision_score(y_true_bin, y_prob, average='macro')
        except Exception as e:
            print(f"Warning: Could not compute AUC metrics: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_prc'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Accuracy
    metrics['accuracy'] = (y_pred == y_true).mean()
    
    return metrics


def compute_confidence_intervals(y_true, y_pred, y_prob=None, task='binary', n_bootstrap=1000, alpha=0.05):
    """
    Compute 95% confidence intervals using bootstrap resampling.
    
    From paper Lines 469-472:
    "95% confidence intervals are estimated via bootstrap resampling
    with 1000 iterations."
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray): Predicted probabilities
        task (str): 'binary' or 'multiclass'
        n_bootstrap (int): Number of bootstrap samples
        alpha (float): Significance level (0.05 for 95% CI)
    
    Returns:
        dict: Metrics with confidence intervals
    """
    n_samples = len(y_true)
    
    # Store bootstrap samples
    bootstrap_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
    }
    
    if y_prob is not None:
        bootstrap_metrics['auc_roc'] = []
        bootstrap_metrics['auc_prc'] = []
    
    # Bootstrap resampling
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices] if y_prob is not None else None
        
        # Compute metrics for this bootstrap sample
        metrics_boot = compute_metrics(y_true_boot, y_pred_boot, y_prob_boot, task)
        
        for key in bootstrap_metrics:
            if key in metrics_boot:
                bootstrap_metrics[key].append(metrics_boot[key])
    
    # Compute confidence intervals
    ci_results = {}
    for key, values in bootstrap_metrics.items():
        values = np.array(values)
        mean_val = values.mean()
        ci_low = np.percentile(values, 100 * alpha / 2)
        ci_high = np.percentile(values, 100 * (1 - alpha / 2))
        
        ci_results[key] = {
            'mean': mean_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'std': values.std()
        }
    
    return ci_results


def evaluate_model(model, dataloader, device='cuda', task='binary'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model (AdapterDeepEncoder or VisionBackbone)
        dataloader: DataLoader for evaluation
        device (str): 'cuda' or 'cpu'
        task (str): 'binary' or 'multiclass'
    
    Returns:
        dict: Evaluation metrics with confidence intervals
    """
    model.eval()
    model.to(device)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different batch formats
            if 'data' in batch:
                x = batch['data'].to(device)
            elif 'image' in batch:
                x = batch['image'].to(device)
            else:
                x = batch[0].to(device)
            
            labels = batch['label'].cpu().numpy()
            
            # Forward pass
            outputs = model(x)
            
            # Extract logits/probabilities
            if isinstance(outputs, dict):
                # AdapterDeepEncoder returns dict
                logits = outputs['classification']
            else:
                # VisionBackbone returns tensor
                logits = outputs
            
            # Convert to probabilities
            if task == 'binary':
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 2 and probs.shape[1] == 1:
                    probs = probs.squeeze(1)
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
            
            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(probs)
    
    # Concatenate
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    # Ensure correct shapes
    if y_true.ndim > 1:
        y_true = y_true.squeeze()
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, task)
    
    # Compute confidence intervals
    ci_results = compute_confidence_intervals(y_true, y_pred, y_prob, task)
    
    # Combine
    results = {
        'metrics': metrics,
        'confidence_intervals': ci_results,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return results


def print_evaluation_results(results, model_name='Model'):
    """Print formatted evaluation results."""
    print(f"\n{'='*70}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*70}")
    
    metrics = results['metrics']
    ci = results['confidence_intervals']
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} (95% CI: [{ci['precision']['ci_low']:.4f}, {ci['precision']['ci_high']:.4f}])")
    print(f"  Recall:    {metrics['recall']:.4f} (95% CI: [{ci['recall']['ci_low']:.4f}, {ci['recall']['ci_high']:.4f}])")
    print(f"  F1 Score:  {metrics['f1']:.4f} (95% CI: [{ci['f1']['ci_low']:.4f}, {ci['f1']['ci_high']:.4f}])")
    
    if 'auc_roc' in metrics:
        print(f"\nAUC Metrics:")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f} (95% CI: [{ci['auc_roc']['ci_low']:.4f}, {ci['auc_roc']['ci_high']:.4f}])")
        print(f"  AUC-PRC:   {metrics['auc_prc']:.4f} (95% CI: [{ci['auc_prc']['ci_low']:.4f}, {ci['auc_prc']['ci_high']:.4f}])")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Generate sample predictions
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Add some correlation
    y_prob = 0.7 * y_true + 0.3 * y_prob
    y_pred = (y_prob > 0.5).astype(int)
    
    print("Testing metrics computation...")
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, task='binary')
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PRC: {metrics['auc_prc']:.4f}")
    
    # Compute confidence intervals
    print("\nComputing confidence intervals (this may take a moment)...")
    ci_results = compute_confidence_intervals(y_true, y_pred, y_prob, n_bootstrap=100)
    
    print(f"\nF1 Score with 95% CI:")
    print(f"  {ci_results['f1']['mean']:.4f} [{ci_results['f1']['ci_low']:.4f}, {ci_results['f1']['ci_high']:.4f}]")
