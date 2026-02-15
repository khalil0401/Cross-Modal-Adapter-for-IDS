"""
Complete Comparison Script

Compares all methods:
1. Adapter + DenseNet (Our Method)
2. InceptionTime (TS Baseline)
3. LSTM-Attention (TS Baseline)
4. GAF + DenseNet (Fixed Transform)
5. RP + DenseNet (Fixed Transform)
6. MTF + DenseNet (Fixed Transform)
7. Shape Modality + DenseNet (Secondary Branch)

Generates publication-ready comparison table and visualizations.

Usage:
    python scripts/compare_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
import pickle
from data import create_dataloaders
from evaluation import plot_roc_curves, plot_precision_recall_curves, plot_comparison_table


def main():
    print("=" * 80)
    print("COMPLETE PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Directories
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    # Load all results
    print("\n[LOADING RESULTS]")
    
    all_results = {}
    
    # Load adapter results
    adapter_results_path = os.path.join(results_dir, 'adapter_results.pkl')
    if os.path.exists(adapter_results_path):
        with open(adapter_results_path, 'rb') as f:
            adapter_results = pickle.load(f)
        all_results['Adapter + DenseNet (Ours)'] = adapter_results
        print("✓ Loaded adapter results")
    else:
        print("✗ Adapter results not found. Run: python scripts/evaluate.py")
    
    # Load baseline results
    baseline_results_path = os.path.join(results_dir, 'baseline_results.pkl')
    if os.path.exists(baseline_results_path):
        with open(baseline_results_path, 'rb') as f:
            baseline_results = pickle.load(f)
        all_results.update(baseline_results)
        print(f"✓ Loaded {len(baseline_results)} baseline results")
    else:
        print("✗ Baseline results not found. Run: python scripts/train_baselines.py")
    
    if len(all_results) == 0:
        print("\n[ERROR] No results found. Please run training and evaluation scripts first.")
        return
    
    # Sort by F1 score (descending)
    sorted_results = dict(sorted(all_results.items(), 
                                 key=lambda x: x[1]['metrics']['f1'], 
                                 reverse=True))
    
    # ========================================================================
    # PRINT COMPARISON TABLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON - ALL METHODS")
    print("=" * 80)
    
    print(f"\n{'Method':<35} {'F1':<10} {'Precision':<12} {'Recall':<10} {'AUC-ROC':<10} {'AUC-PRC':<10}")
    print("-" * 87)
    
    for method_name, results in sorted_results.items():
        metrics = results['metrics']
        ci = results['confidence_intervals']
        
        f1_str = f"{metrics['f1']:.4f} ± {ci['f1']['std']:.4f}"
        prec_str = f"{metrics['precision']:.4f}"
        rec_str = f"{metrics['recall']:.4f}"
        auc_roc_str = f"{metrics.get('auc_roc', 0):.4f}"
        auc_prc_str = f"{metrics.get('auc_prc', 0):.4f}"
        
        print(f"{method_name:<35} {f1_str:<10} {prec_str:<12} {rec_str:<10} {auc_roc_str:<10} {auc_prc_str:<10}")
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # ROC curves
    roc_path = os.path.join(results_dir, 'comparison_roc.png')
    plot_roc_curves(sorted_results, save_path=roc_path)
    
    # Precision-Recall curves
    prc_path = os.path.join(results_dir, 'comparison_prc.png')
    plot_precision_recall_curves(sorted_results, save_path=prc_path)
    
    # Comparison table
    table_path = os.path.join(results_dir, 'comparison_table.png')
    plot_comparison_table(sorted_results, save_path=table_path)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    
    # Best model
    best_method = list(sorted_results.keys())[0]
    best_f1 = sorted_results[best_method]['metrics']['f1']
    
    print(f"\n🏆 Best Method: {best_method}")
    print(f"   F1 Score: {best_f1:.4f}")
    
    # Improvement over best baseline
    if 'Adapter + DenseNet (Ours)' in all_results:
        adapter_f1 = all_results['Adapter + DenseNet (Ours)']['metrics']['f1']
        
        # Find best baseline
        baseline_methods = [k for k in all_results.keys() if 'Ours' not in k]
        if baseline_methods:
            best_baseline_f1 = max([all_results[k]['metrics']['f1'] for k in baseline_methods])
            improvement = ((adapter_f1 - best_baseline_f1) / best_baseline_f1) * 100
            
            print(f"\n📈 Adapter Improvement over Best Baseline:")
            print(f"   Adapter F1: {adapter_f1:.4f}")
            print(f"   Best Baseline F1: {best_baseline_f1:.4f}")
            print(f"   Relative Improvement: {improvement:.2f}%")
    
    print(f"\n📊 Visualizations saved to:")
    print(f"   - ROC curves: {roc_path}")
    print(f"   - Precision-Recall curves: {prc_path}")
    print(f"   - Comparison table: {table_path}")
    
    # Expected results from paper
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 80)
    
    print("\nExpected Performance (from paper Table II):")
    print("  Adapter (Ours):    F1 ~94.4%")
    print("  Shape Modality:    F1 ~92.4%")
    print("  InceptionTime:     F1 ~88.6%")
    print("  MiniRocket:        F1 ~87.7%")
    print("  RP (best fixed):   F1 ~86.6%")
    
    print("\nYour Results:")
    for method_name in ['Adapter + DenseNet (Ours)', 'InceptionTime', 'RP + DenseNet121']:
        if method_name in all_results:
            f1 = all_results[method_name]['metrics']['f1']
            print(f"  {method_name:<25} F1 {f1*100:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
