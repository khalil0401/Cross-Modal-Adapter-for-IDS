"""
Shape Modality - Geometric Representation from Paired Time-Series

Implementation of Algorithm 1 from the paper (Lines 195-216):
Transforms paired time-series into closed geometric shapes

Reference: Section III.A, Algorithm 1
"""

import numpy as np
import torch
from PIL import Image, ImageDraw


def generate_shape_image(signal1, signal2, H=224, W=224, return_tensor=True):
    """
    Generate geometric shape from paired time-series.
    
    Algorithm 1 from paper (Lines 195-216):
    "The shape representation is an enhanced line-plot visualization technique
    that encodes the relationship between two paired TS as a closed geometric
    shape to explicitly capture dynamic interactions within a system."
    
    Steps:
    1. Define horizontal coordinates: x_i = (i-1)/(N-1) * (W-1)
    2. Compute value bounds: v_min, v_max from both signals
    3. Define vertical mapping: y(v) = (v - v_min) / (v_max - v_min) * (H-1)
    4. Create vertex list P:
       - Forward pass: append (x_i, y(s_i)) for i=1..N
       - Backward pass: append (x_i, y(r_i)) for i=N..1
    5. Rasterize filled polygon P into binary image I ∈ {0,1}^(H×W)
    
    Args:
        signal1 (np.ndarray): First time-series (e.g., inbound bytes)
        signal2 (np.ndarray): Second time-series (e.g., outbound bytes)
        H (int): Image height (default: 224)
        W (int): Image width (default: 224)
        return_tensor (bool): If True, return torch.Tensor, else numpy array
    
    Returns:
        torch.Tensor or np.ndarray: Shape image of size (3, H, W) or (H, W, 3)
    
    IoT IDS Examples:
    - Inbound vs. Outbound bytes
    - SYN rate vs. ACK rate
    - Packet rate vs. Energy consumption
    - Forward packet length vs. Backward packet length
    """
    # Ensure inputs are numpy arrays
    signal1 = np.array(signal1)
    signal2 = np.array(signal2)
    
    N = len(signal1)
    assert len(signal2) == N, "Signals must have same length"
    
    # Step 1: Define horizontal coordinates (Line 201)
    x_coords = np.linspace(0, W - 1, N)
    
    # Step 2: Compute value bounds (Lines 202-206)
    v_min = min(signal1.min(), signal2.min())
    v_max = max(signal1.max(), signal2.max())
    
    # Avoid division by zero
    if v_max == v_min:
        v_max = v_min + 1e-6
    
    # Step 3: Define vertical mapping (Lines 207-208)
    def y_mapping(v):
        return (v - v_min) / (v_max - v_min) * (H - 1)
    
    # Step 4: Initialize vertex list P (Line 209)
    vertices = []
    
    # Forward pass: append (x_i, y(signal1_i)) (Lines 210-212)
    for i in range(N):
        x = x_coords[i]
        y = H - 1 - y_mapping(signal1[i])  # Invert y-axis for image coordinates
        vertices.append((x, y))
    
    # Backward pass: append (x_i, y(signal2_i)) (Lines 213-215)
    for i in range(N - 1, -1, -1):
        x = x_coords[i]
        y = H - 1 - y_mapping(signal2[i])
        vertices.append((x, y))
    
    # Step 5: Rasterize filled polygon (Line 216)
    # Create image
    img = Image.new('L', (W, H), 0)  # Binary image, black background
    draw = ImageDraw.Draw(img)
    
    # Draw filled polygon
    draw.polygon(vertices, fill=255, outline=255)
    
    # Convert to numpy array
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Convert to RGB (3 channels)
    img_rgb = np.stack([img_array] * 3, axis=0)  # (3, H, W)
    
    if return_tensor:
        return torch.from_numpy(img_rgb).float()
    else:
        return img_rgb.transpose(1, 2, 0)  # (H, W, 3) for visualization


def generate_shape_batch(signal1_batch, signal2_batch, H=224, W=224):
    """
    Generate batch of shape images from paired time-series batches.
    
    Args:
        signal1_batch (np.ndarray or torch.Tensor): Batch of first signals (batch, time_steps)
        signal2_batch (np.ndarray or torch.Tensor): Batch of second signals (batch, time_steps)
        H (int): Image height
        W (int): Image width
    
    Returns:
        torch.Tensor: Batch of shape images (batch, 3, H, W)
    """
    if isinstance(signal1_batch, torch.Tensor):
        signal1_batch = signal1_batch.cpu().numpy()
        signal2_batch = signal2_batch.cpu().numpy()
    
    batch_size = signal1_batch.shape[0]
    images = []
    
    for i in range(batch_size):
        img = generate_shape_image(signal1_batch[i], signal2_batch[i], H, W, return_tensor=True)
        images.append(img)
    
    return torch.stack(images)


if __name__ == "__main__":
    # Test shape generation
    import matplotlib.pyplot as plt
    
    # Generate sample IoT signals
    time_steps = 288
    
    # Example 1: Normal pattern
    inbound_normal = np.sin(np.linspace(0, 4 * np.pi, time_steps)) * 100 + 500
    outbound_normal = np.cos(np.linspace(0, 4 * np.pi, time_steps)) * 80 + 450
    
    # Example 2: Attack pattern (anomalous)
    inbound_attack = np.concatenate([
        np.random.normal(500, 50, time_steps // 2),
        np.random.normal(1500, 200, time_steps // 2)  # Spike
    ])
    outbound_attack = np.random.normal(300, 100, time_steps)
    
    # Generate shapes
    shape_normal = generate_shape_image(inbound_normal, outbound_normal, return_tensor=False)
    shape_attack = generate_shape_image(inbound_attack, outbound_attack, return_tensor=False)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot time-series
    axes[0, 0].plot(inbound_normal, label='Inbound', color='blue')
    axes[0, 0].plot(outbound_normal, label='Outbound', color='orange')
    axes[0, 0].set_title('Normal Traffic - Time Series')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Bytes')
    
    axes[0, 1].imshow(shape_normal, cmap='gray')
    axes[0, 1].set_title('Normal Traffic - Shape Modality')
    axes[0, 1].axis('off')
    
    axes[1, 0].plot(inbound_attack, label='Inbound', color='blue')
    axes[1, 0].plot(outbound_attack, label='Outbound', color='orange')
    axes[1, 0].set_title('Attack Traffic - Time Series')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Bytes')
    
    axes[1, 1].imshow(shape_attack, cmap='gray')
    axes[1, 1].set_title('Attack Traffic - Shape Modality')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('shape_modality_example.png', dpi=150, bbox_inches='tight')
    print("Saved shape modality example to 'shape_modality_example.png'")
    
    # Test batch generation
    print("\n=== Testing Batch Generation ===")
    batch_size = 4
    signal1_batch = np.random.randn(batch_size, time_steps) * 100 + 500
    signal2_batch = np.random.randn(batch_size, time_steps) * 80 + 450
    
    shape_batch = generate_shape_batch(signal1_batch, signal2_batch)
    print(f"Batch shape: {shape_batch.shape}")
    print(f"Expected: ({batch_size}, 3, 224, 224)")
    print(f"Pixel value range: [{shape_batch.min():.3f}, {shape_batch.max():.3f}]")
