"""Losses package for IoT IDS Adapter"""

from .reconstruction_loss import ReconstructionLoss
from .total_variation_loss import TotalVariationLoss
from .classification_loss import ClassificationLoss
from .composite_loss import CompositeLoss

__all__ = [
    'ReconstructionLoss',
    'TotalVariationLoss',
    'ClassificationLoss',
    'CompositeLoss',
]
