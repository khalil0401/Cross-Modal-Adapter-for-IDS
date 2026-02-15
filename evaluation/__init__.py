"""Evaluation package for IoT IDS Adapter"""

from .metrics import (
    compute_metrics,
    compute_confidence_intervals,
    evaluate_model,
    print_evaluation_results
)

from .visualize import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrix,
    visualize_generated_images,
    plot_comparison_table
)

__all__ = [
    'compute_metrics',
    'compute_confidence_intervals',
    'evaluate_model',
    'print_evaluation_results',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_confusion_matrix',
    'visualize_generated_images',
    'plot_comparison_table',
]
