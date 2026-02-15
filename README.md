# IoT Intrusion Detection System using Learnable Cross-Modal Adapter

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**End-to-end implementation of a learnable cross-modal adapter for IoT intrusion detection**, strictly adapted from:

> **"A Learnable Cross-Modal Adapter for Industrial Fault Detection Using Pretrained Vision Models"**  
> *IEEE Transactions on Industrial Informatics*, 2026

This project transforms the paper's industrial fault detection methodology into a **complete IoT IDS system** without changing the core learnable adapter philosophy.

---

## 🎯 Overview

### Key Innovation

Unlike fixed TS-to-image transformations (GAF, RP, MTF, spectrograms), this system uses a **learnable adapter deep encoder** that:
1. **Learns optimal task-specific representations** from raw network traffic time-series
2. **Generates 224×224×3 RGB images** input-compliant with pretrained vision models (DenseNet, ResNet, ViT)
3. **Optimizes through multitask learning**: reconstruction + total variation + classification

### Architecture

```
Input IoT Traffic → Encoder → Latent Code (z)
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    Decoder       Adapter      Classifier
        ↓             ↓             ↓
Reconstructed  224×224×3 Image  Attack Prob
  Time-Series  → DenseNet/ResNet
```

**Core Components**:
- **Encoder**: 4 conv layers [16,32,64,128], stride=2, BatchNorm+Dropout(0.3)+LeakyReLU
- **Decoder**: Mirrors encoder for TS reconstruction  
- **Image Adapter**: 7×7 seed → 224×224×3 via progressive upsampling
- **Latent Classifier**: FC[64,32] with self-attention
- **Composite Loss**: L_total = 0.5×L_rec + 0.1×L_tv + 0.4×L_cls

---

## 📊 Datasets Supported

- ✅ **NSL-KDD**: Network intrusion dataset
- ✅ **UNSW-NB15**: IoT/network traffic with comprehensive attacks (197 features)

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd iot_ids_adapter
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- tsai (InceptionTime, MiniRocket baselines)
- pyts (GAF, RP, MTF baselines)
- librosa (spectrograms)
- PyYAML, pandas, scikit-learn, matplotlib

---

## 🚀 Quick Start

### 1. Configure Dataset Paths

Edit `config/config.yaml`:

```yaml
dataset:
  name: "UNSW-NB15"  # or "NSL-KDD"
  unsw_nb15_path: "C:/path/to/UNSW_pre_data"
  nsl_kdd_path: "C:/path/to/NSL_pre_data"
```

### 2. Train Adapter

```bash
cd iot_ids_adapter
python scripts/train_adapter.py
```

This will:
- Load and preprocess data (sliding windows, normalization)
- Create train/val/test splits (80/20 + 20% holdout)
- Train adapter with composite loss
- Save best model to `checkpoints/adapter_best.pth`
- Plot training curves

### 3. Complete Evaluation Pipeline

```bash
python scripts/evaluate.py
```

This will:
- Load trained adapter
- Generate images for train/val/test sets
- Fine-tune DenseNet121 on generated images
- Evaluate on test set with metrics (F1, AUC-ROC, AUC-PRC)
- Compute 95% confidence intervals
- Generate all visualizations (ROC, PRC, confusion matrix)

### 4. Train Baseline Models (Optional)

For scientific comparison with the paper:

```bash
python scripts/train_baselines.py
```

This will train:
- InceptionTime (TS baseline)
- LSTM with Attention (TS baseline)
- GAF + DenseNet (fixed transformation)
- RP + DenseNet (fixed transformation)
- MTF + DenseNet (fixed transformation)

### 5. Generate Images from Trained Adapter

```python
import torch
import yaml
from models import AdapterDeepEncoder
from data import create_dataloaders

# Load config and model
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = AdapterDeepEncoder(num_features, 1, config, 'binary')
model.load_state_dict(torch.load('checkpoints/adapter_best.pth'))
model.eval()

# Generate images
train_loader, _, _, _ = create_dataloaders(config)
batch = next(iter(train_loader))
images = model.generate_images(batch['data'])

# images: (batch, 3, 224, 224) - ready for DenseNet/ResNet
```

---

## 📁 Project Structure

```
iot_ids_adapter/
├── config/
│   └── config.yaml           # All hyperparameters from paper
├── models/
│   ├── encoder.py            # 4-layer encoder + self-attention
│   ├── decoder.py            # TS reconstruction
│   ├── adapter.py            # Image adapter (7×7 → 224×224×3)
│   ├── classifier.py         # Latent classification head
│   ├── adapter_full.py       # Complete system
│   └── shape_modality.py     # Geometric shapes (Algorithm 1)
├── losses/
│   ├── reconstruction_loss.py  # L_rec (MSE)
│   ├── total_variation_loss.py # L_tv (smoothness)
│   ├── classification_loss.py  # L_cls (BCE/CE)
│   └── composite_loss.py       # L_total
├── data/
│   └── dataset_loader.py     # NSL-KDD & UNSW-NB15 loaders
├── training/
│   └── trainer.py            # Training loop with early stopping
├── scripts/
├── evaluation/
├── baselines/
└── requirements.txt
```

---

## 🔬 Methodology (from Paper)

### 1. Encoder Architecture (Section III.C)

```python
# 4 sequential 1D conv layers
filters = [16, 32, 64, 128]
kernel_size = 3
stride = 2  # Progressive downsampling: T → T/2 → T/4 → T/8 → T/16

# Each layer:
Conv1D → BatchNorm → Dropout(0.3) → LeakyReLU
```

**Output**: Latent code `z` of shape `(batch, 128, T/16)`

### 2. Image Adapter (Lines 301-321)

```python
# Step 1: Treat latent as 2D map (time × channels)
# Step 2: Expand to (T, C, 1) → project via 2D conv to (T, C, C_p)
# Step 3: Resize to 7×7 seed
# Step 4: Progressive upsampling:
#   7×7 → 14×14 → 28×28 → 56×56 → 112×112 → 224×224
# Step 5: Final conv to 3 RGB channels
```

**Output**: Image `I` of shape `(batch, 3, 224, 224)`

### 3. Composite Loss (Equation 3)

```python
L_total = 0.5 * L_rec + 0.1 * L_tv + 0.4 * L_cls

# where:
# - L_rec: MSE between original and reconstructed TS
# - L_tv: Total variation for image smoothness
# - L_cls: Binary/categorical cross-entropy
```

### 4. Training (Section V)

- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping patience=8)
- **Seed**: 42 (reproducibility)

---

## 📈 Expected Results (from Paper)

| Method | F1 Score | AUC-ROC |
|--------|----------|---------|
| InceptionTime (baseline) | 88.6% | 0.95+ |
| RP (best fixed method) | 86.6% | 0.93+ |
| **Shape Modality** | **92.4%** | **0.97+** |
| **Adapter (Ours)** | **94.4%** | **0.98+** |

### Success Criteria

1. ✅ Adapter F1 > InceptionTime F1 by ≥3-5%
2. ✅ Adapter AUC-ROC > 0.95
3. ✅ Training converges within 100 epochs
4. ✅ No overfitting (val loss tracks train loss)

---

## 🛠️ Advanced Usage

### Shape Modality (Algorithm 1)

Generate geometric shapes from paired IoT signals:

```python
from models import generate_shape_image
import numpy as np

# Example: Inbound vs. Outbound bytes
inbound_bytes = np.random.randn(288) * 100 + 500
outbound_bytes = np.random.randn(288) * 80 + 450

# Generate 224×224 shape image
shape_img = generate_shape_image(inbound_bytes, outbound_bytes)
# shape_img: (3, 224, 224) tensor
```

### Custom Feature Pairs

Edit `data/dataset_loader.py`:

```python
# Define IoT-relevant pairs
feature_pairs = [
    (2, 3),   # (source packets, dest packets)
    (4, 5),   # (source bytes, dest bytes)
    (10, 11), # (SYN rate, ACK rate)
]
```

---

## 📝 Citation

If you use this code, please cite the original paper:

```bibtex
@article{vandreven2026learnable,
  title={A Learnable Cross-Modal Adapter for Industrial Fault Detection Using Pretrained Vision Models},
  author={van Dreven, Jonne and Cheddad, Abbas and Alawadi, Sadi and Ghazi, A.N. and Al Koussa, J. and Vanhoudt, Dirk},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  publisher={IEEE}
}
```

---

## 🔍 Key Differences from Paper

| **Original (Fault Detection)** | **Adapted (Intrusion Detection)** |
|---------------------------------|------------------------------------|
| District heating sensors (Ts, Tr, Q, E) | Network traffic (packets, bytes, flags, duration) |
| Fault vs. Normal operation | Attack vs. Benign traffic |
| 5 fault types (HHC, WSP, LV, etc.) | Attack types (DoS, DDoS, Mirai, etc.) |
| Daily windows (288 samples @ 5 min) | Traffic windows (288 samples) |
| Paired: (supply temp, return temp) | Paired: (inbound, outbound), (SYN, ACK) |

**Core methodology unchanged**: Learnable adapter, composite loss, vision backbone integration.

---

## ⚠️ Important Notes

### ✅ Learnable Adapter (NOT Fixed)

The adapter weights are **trained end-to-end** through backpropagation. This is NOT a fixed transformation like GAF or spectrogram.

### ✅ Vision Backbone Unmodified

DenseNet/ResNet architectures are used **without internal modifications**. Only the final classification layer is replaced.

### ✅ Reproducibility

All random seeds are fixed (seed=42). Results should be identical across runs with same data.

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows paper's architecture strictly
- All hyperparameters documented with paper references
- Tests pass: `pytest tests/`

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

## Acknowledgments

This implementation is based on:
> J. van Dreven et al., "A Learnable Cross-Modal Adapter for Industrial Fault Detection Using Pretrained Vision Models," IEEE TII, 2026.

All architectural decisions, hyperparameters, and training strategies are extracted directly from the paper to ensure scientific accuracy and reproducibility.
