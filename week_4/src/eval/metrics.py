# [Week 4] src/eval/metrics.py
import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


def dice_score(pred3d, gt3d, eps=1e-6):
    """Volumetric Dice Similarity Coefficient between two 3-D binary masks.

    Formula: DSC = 2|A ∩ B| / (|A| + |B|)
    Range: 0 (no overlap) → 1 (perfect overlap).
    Published SAM 2 baselines: liver ~0.93, spleen ~0.94, pancreas ~0.72.

    Specification:
    - Cast both inputs to bool.
    - intersection = (pred & gt).sum()
    - Return float((2 * intersection + eps) / (pred.sum() + gt.sum() + eps))

    Args:
        pred3d (np.ndarray): 3-D bool/uint8 predicted mask, shape (H, W, Z).
        gt3d   (np.ndarray): 3-D bool/uint8 ground-truth mask, shape (H, W, Z).
        eps    (float):      Smoothing constant (avoids division by zero).

    Returns:
        float: Dice score in [0, 1].
    """
    pre = pred3d.astype(bool)
    gt3 = gt3d.astype(bool)
    intersection = (pre & gt3).sum()
    return float((2*intersection + eps) / (pred3d.sum() + gt3.sum() + eps))


def hd95(pred3d, gt3d, spacing):
    """95th-percentile Hausdorff Distance in millimetres via MONAI.

    Measures boundary accuracy — lower is better.
    Typical values: liver < 5 mm, spleen < 5 mm, pancreas < 20 mm.

    Specification:
    - Convert arrays to float32 torch tensors with batch and channel dims:
          p = torch.tensor(pred3d.astype(np.uint8), dtype=torch.float32)[None, None]
          g = torch.tensor(gt3d.astype(np.uint8),   dtype=torch.float32)[None, None]
          # Shape: (1, 1, H, W, Z)
    - MONAI expects spacing in (z, y, x) order for 3-D volumes.
      Our volume is (H, W, Z) so spacing from load_volume is (sx, sy, sz).
      Reorder: sp_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    - dist = compute_hausdorff_distance(p, g, percentile=95, spacing=sp_zyx)
    - Return float(dist.squeeze())
    - Always pass spacing — without it the result is in voxels, not mm.

    Args:
        pred3d  (np.ndarray): 3-D bool predicted mask, shape (H, W, Z).
        gt3d    (np.ndarray): 3-D bool ground-truth mask, shape (H, W, Z).
        spacing (tuple):      (sx, sy, sz) voxel size in mm from load_volume().

    Returns:
        float: HD95 in millimetres.
    """
    p = torch.tensor(pred3d.astype(np.uint8), dtype=torch.float32)[None, None]
    g = torch.tensor(gt3d.astype(np.uint8), dtype=torch.float32)[None, None]
    sp_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    dist = compute_hausdorff_distance(p, g, percentile=95, spacing=sp_zyx)
    return float(dist.squeeze())
