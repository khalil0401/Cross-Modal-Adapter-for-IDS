"""
Fixed TS-to-Image Transformations

Implementations of:
- Gramian Angular Field (GAF)
- Recurrence Plot (RP)
- Markov Transition Field (MTF)

These serve as baselines for comparison with the learnable adapter.

Reference: Paper baselines (Table II)
From: "Imaging Time-Series to Improve Classification and Imputation", 2015
"""

import numpy as np
import torch
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField


def transform_to_gaf(time_series, image_size=224, method='summation'):
    """
    Transform time-series to Gramian Angular Field (GAF).
    
    From paper baselines:
    "GAF encodes time-series as polar coordinates and computes
    the cosine of angular sums, creating texture-rich images."
    
    Args:
        time_series (np.ndarray): Input time-series (batch, features, time_steps)
        image_size (int): Output image size
        method (str): 'summation' (GASF) or 'difference' (GADF)
    
    Returns:
        torch.Tensor: GAF images (batch, 3, image_size, image_size)
    """
    batch_size, n_features, time_steps = time_series.shape
    
    # Create GAF transformer
    gaf = GramianAngularField(image_size=image_size, method=method)
    
    images = []
    
    for i in range(batch_size):
        # For multivariate TS, use first 3 features as RGB channels
        # or aggregate to single series
        if n_features >= 3:
            # Use first 3 features as RGB
            feature_indices = [0, 1, 2]
        else:
            # Repeat features to create 3 channels
            feature_indices = [0] * 3
        
        channels = []
        for feat_idx in feature_indices:
            ts = time_series[i, feat_idx % n_features]
            
            # Transform to GAF
            gaf_img = gaf.fit_transform(ts.reshape(1, -1))[0]
            
            # Normalize to [0, 1]
            gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
            
            channels.append(gaf_img)
        
        # Stack as RGB
        img_rgb = np.stack(channels, axis=0)  # (3, image_size, image_size)
        images.append(img_rgb)
    
    images = np.array(images)
    return torch.from_numpy(images).float()


def transform_to_rp(time_series, image_size=224):
    """
    Transform time-series to Recurrence Plot (RP).
    
    From paper baselines:
    "Recurrence plots visualize recurrent states in phase space,
    revealing periodic and chaotic behavior patterns."
    
    Args:
        time_series (np.ndarray): Input time-series (batch, features, time_steps)
        image_size (int): Output image size
    
    Returns:
        torch.Tensor: RP images (batch, 3, image_size, image_size)
    """
    batch_size, n_features, time_steps = time_series.shape
    
    # Create RP transformer
    rp = RecurrencePlot(dimension=1, threshold='point', percentage=10)
    
    images = []
    
    for i in range(batch_size):
        # For multivariate TS, use first 3 features as RGB channels
        if n_features >= 3:
            feature_indices = [0, 1, 2]
        else:
            feature_indices = [0] * 3
        
        channels = []
        for feat_idx in feature_indices:
            ts = time_series[i, feat_idx % n_features]
            
            # Transform to RP
            rp_img = rp.fit_transform(ts.reshape(1, -1))[0]
            
            # Resize to target size
            from scipy.ndimage import zoom
            zoom_factor = image_size / rp_img.shape[0]
            rp_img = zoom(rp_img, zoom_factor, order=1)
            
            # Normalize to [0, 1]
            rp_img = rp_img.astype(float)
            
            channels.append(rp_img)
        
        # Stack as RGB
        img_rgb = np.stack(channels, axis=0)  # (3, image_size, image_size)
        images.append(img_rgb)
    
    images = np.array(images)
    return torch.from_numpy(images).float()


def transform_to_mtf(time_series, image_size=224, n_bins=8):
    """
    Transform time-series to Markov Transition Field (MTF).
    
    From paper baselines:
    "MTF encodes temporal transition probabilities between
    quantized states, capturing dynamics and trends."
    
    Args:
        time_series (np.ndarray): Input time-series (batch, features, time_steps)
        image_size (int): Output image size
        n_bins (int): Number of quantization bins
    
    Returns:
        torch.Tensor: MTF images (batch, 3, image_size, image_size)
    """
    batch_size, n_features, time_steps = time_series.shape
    
    # Create MTF transformer
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    
    images = []
    
    for i in range(batch_size):
        # For multivariate TS, use first 3 features as RGB channels
        if n_features >= 3:
            feature_indices = [0, 1, 2]
        else:
            feature_indices = [0] * 3
        
        channels = []
        for feat_idx in feature_indices:
            ts = time_series[i, feat_idx % n_features]
            
            # Transform to MTF
            mtf_img = mtf.fit_transform(ts.reshape(1, -1))[0]
            
            # Normalize to [0, 1]
            mtf_img = (mtf_img - mtf_img.min()) / (mtf_img.max() - mtf_img.min() + 1e-8)
            
            channels.append(mtf_img)
        
        # Stack as RGB
        img_rgb = np.stack(channels, axis=0)  # (3, image_size, image_size)
        images.append(img_rgb)
    
    images = np.array(images)
    return torch.from_numpy(images).float()


def transform_batch_to_images(dataloader, method='gaf', image_size=224):
    """
    Transform entire dataset to images using specified method.
    
    Args:
        dataloader: DataLoader with time-series data
        method (str): 'gaf', 'rp', or 'mtf'
        image_size (int): Output image size
    
    Returns:
        tuple: (images, labels)
    """
    all_images = []
    all_labels = []
    
    print(f"Transforming data to {method.upper()} images...")
    
    for batch in dataloader:
        x = batch['data'].numpy()  # (batch, features, time_steps)
        labels = batch['label']
        
        # Transform
        if method == 'gaf':
            images = transform_to_gaf(x, image_size)
        elif method == 'rp':
            images = transform_to_rp(x, image_size)
        elif method == 'mtf':
            images = transform_to_mtf(x, image_size)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        all_images.append(images)
        all_labels.append(labels)
    
    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0).squeeze()
    
    print(f"Generated {len(all_images)} {method.upper()} images")
    
    return all_images, all_labels


if __name__ == "__main__":
    # Test transformations
    batch_size = 4
    n_features = 10
    time_steps = 288
    
    # Generate sample data
    time_series = np.random.randn(batch_size, n_features, time_steps)
    
    print("Testing GAF transformation...")
    gaf_images = transform_to_gaf(time_series)
    print(f"GAF output shape: {gaf_images.shape}")
    print(f"Expected: ({batch_size}, 3, 224, 224)")
    
    print("\nTesting RP transformation...")
    rp_images = transform_to_rp(time_series)
    print(f"RP output shape: {rp_images.shape}")
    
    print("\nTesting MTF transformation...")
    mtf_images = transform_to_mtf(time_series)
    print(f"MTF output shape: {mtf_images.shape}")
