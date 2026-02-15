"""
Classification Loss (L_cls)

Implementation of Equation 6 from the paper:
Binary cross-entropy for fault detection
Categorical cross-entropy for fault diagnosis

Reference: Lines 281-288
"""

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """
    Classification loss for fault detection or diagnosis.
    
    From paper Equation 6 and Lines 287-288:
    Binary Detection (Lines 281-286):
    "The classification loss minimizes the binary cross-entropy between
    the predicted probability y_hat ∈ (0,1) and ground-truth label y ∈ {0,1}"
    
    L_cls = -[y * log(y_hat) + (1-y) * log(1-y_hat)]
    
    Multiclass Diagnosis (Lines 287-288):
    "For multiclass problems, such as fault diagnosis, it is replaced by
    the categorical cross-entropy over the softmax output."
    """
    
    def __init__(self, task='binary'):
        """
        Args:
            task (str): 'binary' for detection, 'multiclass' for diagnosis
        """
        super(ClassificationLoss, self).__init__()
        
        self.task = task
        
        if task == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Compute classification loss.
        
        Args:
            predictions (torch.Tensor): Model predictions (logits)
                - Binary: (batch, 1)
                - Multiclass: (batch, num_classes)
            targets (torch.Tensor): Ground truth labels
                - Binary: (batch,) or (batch, 1) with values in {0, 1}
                - Multiclass: (batch,) with class indices
        
        Returns:
            torch.Tensor: Scalar classification loss
        """
        if self.task == 'binary':
            # Ensure targets are float and have correct shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            
            # Squeeze predictions if needed
            if predictions.dim() == 2 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(1)
            if targets.dim() == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            
            return self.loss_fn(predictions, targets)
        else:
            # Multiclass: targets should be long tensor with class indices
            targets = targets.long()
            return self.loss_fn(predictions, targets)


if __name__ == "__main__":
    # Test binary classification loss
    print("=== Binary Classification Loss ===")
    loss_binary = ClassificationLoss(task='binary')
    
    batch_size = 4
    predictions_binary = torch.randn(batch_size, 1)  # Logits
    targets_binary = torch.tensor([1, 0, 1, 0]).float()
    
    loss_b = loss_binary(predictions_binary, targets_binary)
    print(f"Predictions (logits): {predictions_binary.squeeze()}")
    print(f"Targets: {targets_binary}")
    print(f"Loss: {loss_b.item():.6f}")
    
    # Test multiclass classification loss
    print("\n=== Multiclass Classification Loss ===")
    num_classes = 5
    loss_multiclass = ClassificationLoss(task='multiclass')
    
    predictions_multi = torch.randn(batch_size, num_classes)  # Logits
    targets_multi = torch.tensor([0, 2, 1, 4])  # Class indices
    
    loss_m = loss_multiclass(predictions_multi, targets_multi)
    print(f"Predictions shape: {predictions_multi.shape}")
    print(f"Targets: {targets_multi}")
    print(f"Loss: {loss_m.item():.6f}")
