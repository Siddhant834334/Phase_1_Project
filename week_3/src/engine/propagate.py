# [Week 3] src/engine/propagate.py
# Dependency: src/engine/predictor.py (Week 3)
import numpy as np
from PIL import Image


def propagate_bidirectional(predictor, state, start_z, bbox, target_hw=None):
    """Run SAM 2 bidirectional propagation and return a 3-D binary mask.

    Specification:
    Step 1 — Inject the bounding-box prompt on start_z:
        predictor.add_new_points_or_box(
            state, frame_idx=start_z, obj_id=1,
            box=bbox   # float32 [x0, y0, x1, y1]
        )

    Step 2 — Forward pass (start_z → last slice):
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=start_z, reverse=False):
            store masks[0, 0] for frame_idx

    Step 3 — Backward pass (start_z → first slice):
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=start_z, reverse=True):
            store masks[0, 0] for frame_idx  (do not overwrite start_z)

    Step 4 — Each per-slice mask is a (h, w) logit tensor on GPU.
        - Threshold at 0: mask_bool = (mask > 0).cpu().numpy()
        - If target_hw is given and differs from (h, w), resize to target_hw
          using PIL NEAREST (preserve binary values).

    Step 5 — Stack all per-slice bool arrays along axis=2 → shape (H, W, Z).

    Args:
        predictor  (SAM2VideoPredictor): Built and initialised predictor.
        state      (dict):               Inference state from init_state.
        start_z    (int):                Z-index of the prompt slice.
        bbox       (np.ndarray):         float32 [x0, y0, x1, y1] bounding box.
        target_hw  (tuple | None):       (H, W) to resize masks to, or None to
                                         keep the predictor's native resolution.

    Returns:
        np.ndarray: bool array of shape (H, W, Z) — the full 3-D organ mask.
    """
    pass
