"""
Loss functions, metrics, and training utilities for USCNet.

Improvements over v1:
  - Class-weighted CrossEntropy: weights inversely proportional to class frequency.
  - Label smoothing in CrossEntropy to reduce overconfidence.
  - Class-weighted FocalLoss: per-sample alpha from class weights tensor.
  - compute_class_weights() helper used in train.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list, num_classes: int = 4,
                           device: torch.device = None) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a list of integer labels.
    Weight for class c = total_samples / (num_classes * count_c).
    Returns a float tensor of shape (num_classes,).
    """
    labels = np.array(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    total = len(labels)
    for c in range(num_classes):
        count = (labels == c).sum()
        weights[c] = total / (num_classes * max(count, 1))
    # Normalise so the mean weight is 1.0
    weights = weights / weights.mean()
    t = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification.
    Supports a per-class alpha weight tensor (class_weights).
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean",
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Stored as a buffer so it moves to GPU with .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if self.class_weights is not None else None
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class USCNetLoss(nn.Module):
    """
    Combined loss for USCNet:
      L_total = lambda_dice * L_dice + lambda_bce * L_bce + lambda_focal * L_focal

    Where L_dice is for segmentation, and L_bce + L_focal are for classification.
    Dynamic weight adjustment follows the paper.

    Args:
        num_classes: Number of output classes.
        class_weights: (num_classes,) tensor of inverse-frequency class weights.
                       Applied to both CE and Focal losses.
        label_smoothing: Label smoothing epsilon for CrossEntropyLoss.
    """

    def __init__(
        self,
        num_classes: int = 4,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_focal: float = 1.0,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.focal_loss = FocalLoss(gamma=2.0, class_weights=class_weights)

        self.lambda_dice  = nn.Parameter(torch.tensor(lambda_dice),  requires_grad=False)
        self.lambda_bce   = nn.Parameter(torch.tensor(lambda_bce),   requires_grad=False)
        self.lambda_focal = nn.Parameter(torch.tensor(lambda_focal), requires_grad=False)

    def forward(
        self,
        seg_pred:   torch.Tensor,
        seg_target: torch.Tensor,
        cls_pred:   torch.Tensor,
        cls_target: torch.Tensor,
    ) -> dict:
        l_dice  = self.dice_loss(seg_pred, seg_target)
        l_bce   = self.ce_loss(cls_pred, cls_target)
        l_focal = self.focal_loss(cls_pred, cls_target)

        total = (
            self.lambda_dice  * l_dice
            + self.lambda_bce   * l_bce
            + self.lambda_focal * l_focal
        )

        return {
            "total": total,
            "dice":  l_dice.detach(),
            "ce":    l_bce.detach(),
            "focal": l_focal.detach(),
        }


class DynamicWeightAdjuster:
    """
    Dynamic weight adjustment for multi-task loss balancing.
    Adjusts lambda weights based on loss ratios between tasks.
    """

    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature
        self.prev_losses = None

    @torch.no_grad()
    def step(self, loss_module: USCNetLoss, current_losses: dict):
        losses = torch.tensor([
            current_losses["dice"],
            current_losses["ce"],
            current_losses["focal"],
        ])

        if self.prev_losses is not None:
            ratios = losses / (self.prev_losses + 1e-8)
            weights = F.softmax(ratios / self.temperature, dim=0) * 3.0
            loss_module.lambda_dice.fill_(weights[0].item())
            loss_module.lambda_bce.fill_(weights[1].item())
            loss_module.lambda_focal.fill_(weights[2].item())

        self.prev_losses = losses.clone()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_preds: list, all_labels: list, num_classes: int = 4) -> dict:
    """Compute classification metrics."""
    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    f1        = f1_score(labels, preds, average="macro", zero_division=0)

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    label_names = ["large_cell", "squamous", "adenocarcinoma", "nos"]
    report = classification_report(
        labels, preds, target_names=label_names[:num_classes], zero_division=0
    )

    return {
        "accuracy":     acc,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "per_class_f1": per_class_f1,
        "report":       report,
    }


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor,
                       smooth: float = 1.0) -> float:
    """Compute Dice coefficient for segmentation evaluation."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred_flat    = pred.view(-1)
    target_flat  = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
