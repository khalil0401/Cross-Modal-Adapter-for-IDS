"""
IoT Dataset Preprocessing and Loading

Handles NSL-KDD and UNSW-NB15 datasets with:
- Sliding window segmentation
- Normalization
- Train/val/test splits
- Feature selection for paired signals

Reference: Section V, Lines 459-466
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


class IoTDataset(Dataset):
    """
    PyTorch Dataset for IoT IDS with sliding window segmentation.
    
    From paper Lines 459-466:
    "All TS inputs are processed using standard preprocessing practices.
    The signals are segmented into fixed-length windows (e.g., 288 samples),
    ensuring equal-length inputs. Missing values are handled using linear
    interpolation. Min-max normalization is applied jointly to paired TS."
    """
    
    def __init__(self, data, labels, window_size=288, feature_pairs=None):
        """
        Args:
            data (np.ndarray): Features (num_samples, num_features)
            labels (np.ndarray): Labels (num_samples,)
            window_size (int): Size of sliding window
            feature_pairs (list of tuples): Pairs of feature indices for shape modality
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.feature_pairs = feature_pairs
        
        # Create windows
        self.windows, self.window_labels = self._create_windows()
    
    def _create_windows(self):
        """
        Segment data into fixed-length windows with sliding.
        """
        num_samples = len(self.data)
        
        if num_samples < self.window_size:
            # Pad if needed
            padding = self.window_size - num_samples
            self.data = np.pad(self.data, ((0, padding), (0, 0)), mode='edge')
            self.labels = np.pad(self.labels, (0, padding), mode='edge')
            num_samples = len(self.data)
        
        windows = []
        window_labels = []
        
        # Sliding window with stride (non-overlapping for simplicity)
        stride = self.window_size  # Can be adjusted
        
        for i in range(0, num_samples - self.window_size + 1, stride):
            window = self.data[i:i + self.window_size]
            # Label: majority vote or last label in window
            label = np.bincount(self.labels[i:i + self.window_size].astype(int)).argmax()
            
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'data': (window_size, num_features),
                'label': scalar,
                'paired_signals': [(window_size,), (window_size,)] if feature_pairs defined
            }
        """
        window = self.windows[idx]  # (window_size, num_features)
        label = self.window_labels[idx]
        
        # Transpose to (num_features, window_size) for Conv1D
        window_t = torch.FloatTensor(window.T)
        
        item = {
            'data': window_t,
            'label': torch.FloatTensor([label])
        }
        
        # Extract paired signals for shape modality if specified
        if self.feature_pairs:
            paired_signals = []
            for feat1_idx, feat2_idx in self.feature_pairs:
                signal1 = window[:, feat1_idx]
                signal2 = window[:, feat2_idx]
                paired_signals.append((signal1, signal2))
            item['paired_signals'] = paired_signals
        
        return item


def load_nsl_kdd(data_path, config):
    """
    Load and preprocess NSL-KDD dataset.
    
    Args:
        data_path (str): Path to NSL-KDD directory
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels, feature_names)
    """
    print("Loading NSL-KDD dataset...")
    
    # Load train and test files
    train_df = pd.read_csv(os.path.join(data_path, 'PKDDTrain+.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'PKDDTest+.csv'))
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Assuming 'labels2' or 'labels5' column contains binary/multi-class labels
    # and remaining columns are features
    label_col = 'labels2'  # Binary: normal vs anomaly
    if label_col not in train_df.columns:
        # Try other label columns
        label_cols = [col for col in train_df.columns if 'label' in col.lower()]
        if label_cols:
            label_col = label_cols[0]
        else:
            raise ValueError("No label column found in NSL-KDD data")
    
    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col != label_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values
    
    # Convert labels to binary (0: normal, 1: attack)
    y_train = (y_train != 'normal').astype(int)
    y_test = (y_test != 'normal').astype(int)
    
    # Handle missing values with linear interpolation
    X_train = pd.DataFrame(X_train).interpolate(method='linear', axis=0).fillna(0).values
    X_test = pd.DataFrame(X_test).interpolate(method='linear', axis=0).fillna(0).values
    
    # Min-max normalization (applied jointly)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Features: {len(feature_cols)}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train attack ratio: {y_train.mean():.2%}, Test attack ratio: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def load_unsw_nb15(data_path, config):
    """
    Load and preprocess UNSW-NB15 dataset.
    
    Args:
        data_path (str): Path to UNSW-NB15 directory
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels, feature_names)
    """
    print("Loading UNSW-NB15 dataset...")
    
    # Load train and test files
    train_df = pd.read_csv(os.path.join(data_path, 'UNSWTrain.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'UNSWTest.csv'))
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Label column
    label_col = 'label'
    
    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col != label_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values
    
    # Handle missing values
    X_train = pd.DataFrame(X_train).interpolate(method='linear', axis=0).fillna(0).values
    X_test = pd.DataFrame(X_test).interpolate(method='linear', axis=0).fillna(0).values
    
    # Min-max normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Features: {len(feature_cols)}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train attack ratio: {y_train.mean():.2%}, Test attack ratio: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_features)
    """
    # Load dataset based on config
    dataset_name = config['dataset']['name']
    
    if dataset_name == 'NSL-KDD':
        data_path = config['dataset']['nsl_kdd_path']
        X_train, y_train, X_test, y_test, feature_cols = load_nsl_kdd(data_path, config)
    elif dataset_name == 'UNSW-NB15':
        data_path = config['dataset']['unsw_nb15_path']
        X_train, y_train, X_test, y_test, feature_cols = load_unsw_nb15(data_path, config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split train into train/val
    val_size = config['dataset']['val_size']
    seed = config['dataset']['seed']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    
    # Define feature pairs for shape modality (examples for UNSW-NB15)
    # User can customize based on domain knowledge
    feature_pairs = [
        (2, 3),  # (spkts, dpkts) - source/dest packets
        (3, 4),  # (dpkts, sbytes) - dest packets vs source bytes
    ] if len(feature_cols) >= 5 else None
    
    # Create datasets
    window_size = config['dataset']['window_size']
    
    train_dataset = IoTDataset(X_train, y_train, window_size, feature_pairs)
    val_dataset = IoTDataset(X_val, y_val, window_size, feature_pairs)
    test_dataset = IoTDataset(X_test, y_test, window_size, feature_pairs)
    
    # Create dataloaders
    batch_size = config['training']['adapter_training']['batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_features = len(feature_cols)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")
    print(f"Test windows: {len(test_dataset)}")
    print(f"Window size: {window_size}")
    print(f"Number of features: {num_features}")
    
    return train_loader, val_loader, test_loader, num_features


if __name__ == "__main__":
    # Test data loading
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with UNSW-NB15
    config['dataset']['name'] = 'UNSW-NB15'
    
    train_loader, val_loader, test_loader, num_features = create_dataloaders(config)
    
    # Test batch
    print("\n=== Testing Batch ===")
    for batch in train_loader:
        print(f"Data shape: {batch['data'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Label values: {batch['label'][:5].squeeze()}")
        
        if 'paired_signals' in batch:
            print(f"Number of feature pairs: {len(batch['paired_signals'])}")
        
        break
