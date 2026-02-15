"""
Quick Start Script

Runs the complete pipeline:
1. Train adapter
2. Evaluate adapter + DenseNet
3. Train baselines (optional)
4. Compare all methods

Usage:
    python scripts/run_all.py [--skip-baselines]
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run complete IoT IDS pipeline')
    parser.add_argument('--skip-baselines', action='store_true',
                      help='Skip baseline training (faster, but no comparison)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("IoT IDS COMPLETE PIPELINE")
    print("=" * 80)
    
    # Change to scripts directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(scripts_dir)
    
    # Step 1: Train adapter
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING ADAPTER")
    print("=" * 80)
    
    ret = os.system('python train_adapter.py')
    if ret != 0:
        print("\n[ERROR] Adapter training failed!")
        return
    
    # Step 2: Evaluate adapter
    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING ADAPTER")
    print("=" * 80)
    
    ret = os.system('python evaluate.py')
    if ret != 0:
        print("\n[ERROR] Adapter evaluation failed!")
        return
    
    # Step 3: Train baselines (optional)
    if not args.skip_baselines:
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING BASELINES")
        print("=" * 80)
        
        ret = os.system('python train_baselines.py')
        if ret != 0:
            print("\n[WARNING] Baseline training failed, continuing...")
    else:
        print("\n[INFO] Skipping baseline training (--skip-baselines flag)")
    
    # Step 4: Compare all methods
    print("\n" + "=" * 80)
    print("STEP 4: COMPARING ALL METHODS")
    print("=" * 80)
    
    ret = os.system('python compare_all.py')
    if ret != 0:
        print("\n[WARNING] Comparison failed!")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nCheck the 'results/' directory for all outputs.")


if __name__ == "__main__":
    main()
