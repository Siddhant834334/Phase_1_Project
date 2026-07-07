# [Week 2] src/utils/viz.py
import numpy as np
from pathlib import Path
import imageio.v2 as imageio


def save_overlay_gif(vol_u8, pred3d, out_path, gt3d=None):
    """Write an animated GIF overlaying segmentation masks onto the CT volume.

    Specification:
    - Iterate over every axial slice (Z axis) of vol_u8.
    - Convert each greyscale (H, W) slice to RGB by repeating the channel.
    - Where pred3d[:, :, z] is True, apply a RED tint  (blend 50 % with [220,60,60]).
    - Where gt3d[:, :, z] is True (if gt3d is not None), apply a GREEN tint
      (blend 50 % with [60,220,60]).  GT is rendered beneath the prediction.
    - Collect all (H, W, 3) uint8 frames and write them as an animated GIF.
    - Create the parent directory of out_path if it does not exist.
    - Recommended frame duration: 80 ms.

    Args:
        vol_u8   (np.ndarray):      uint8 CT volume, shape (H, W, Z).
        pred3d   (np.ndarray):      bool/uint8 predicted mask, shape (H, W, Z).
        out_path (str | Path):      Destination .gif path.
        gt3d     (np.ndarray|None): Ground-truth mask same shape, or None.
    """
    pass
