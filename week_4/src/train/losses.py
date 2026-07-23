# [Week 4] src/train/losses.py
import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss for binary segmentation.

    Specification:
    - Convert logits to probabilities: p = torch.sigmoid(logits).
    - Soft intersection:  inter = (p * targets).sum()
    - Soft union:         union = p.sum() + targets.sum()
    - Dice coefficient:   (2 * inter + eps) / (union + eps)
    - Return 1 − dice_coefficient  (0 = perfect, 1 = worst)
    - Result must be a scalar tensor with a gradient.

    Args:
        logits  (Tensor): Raw model output, any shape (NOT sigmoid-activated).
        targets (Tensor): Binary float32 ground truth, same shape as logits.
        eps     (float):  Smoothing constant to avoid division by zero.

    Returns:
        Tensor: Scalar soft Dice loss with gradient.
    """
    import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss for binary segmentation.
    
    Specification:
    - Convert logits to probabilities: p = torch.sigmoid(logits).
    - Soft intersection:  inter = (p * targets).sum()
    - Soft union:         union = p.sum() + targets.sum()
    - Dice coefficient:   (2 * inter + eps) / (union + eps)
    - Return 1 - dice_coefficient  (0 = perfect, 1 = worst)
    - Result must be a scalar tensor with a gradient.
    
    Args:
        logits  (Tensor): Raw model output, any shape (NOT sigmoid-activated).
        targets (Tensor): Binary float32 ground truth, same shape as logits.
        eps     (float):  Smoothing constant to avoid division by zero.
        
    Returns:
        Tensor: Scalar soft Dice loss with gradient.
    """
    # 1. Convert logits to probabilities using sigmoid
    p = torch.sigmoid(logits)
    
    # 2. Compute soft intersection and union
    inter = (p * targets).sum()
    union = p.sum() + targets.sum()
    
    # 3. Calculate the Dice coefficient
    dice_coefficient = (2 * inter + eps) / (union + eps)
    
    # 4. Return the loss (1 - Dice coefficient)
    return 1 - dice_coefficient


def total_loss(logits, targets):
    """Combined Dice + Binary Cross-Entropy loss for organ segmentation.

    Specification:
    - loss = dice_loss(logits, targets)
           + F.binary_cross_entropy_with_logits(logits, targets)
    - Dice drives spatial overlap; BCE drives per-pixel calibration.
    - Return the sum as a scalar tensor.

    Args:
        logits  (Tensor): Raw model output (NOT sigmoid-activated).
        targets (Tensor): Binary float32 ground truth, same shape as logits.

    Returns:
        Tensor: Scalar combined loss with gradient.
    """
    d_loss = dice_loss(logits, targets)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    return d_loss + bce_loss
