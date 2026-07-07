# [Week 2] src/data/dataset.py
# Dependency: src/data/nifti_io.py (Week 1), src/data/prompts.py (Week 2)
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.prompts import bbox_from_mask


class BTCVSliceDataset(Dataset):
    """2-D prompted dataset of organ-containing axial slices for LoRA training.

    Specification:
    __init__:
    - Iterate over every case in `cases`.
    - Load image and label volumes from image_dir/img<case>.nii and
      label_dir/label<case>.nii (replace 'img' with 'label' in the stem).
    - For each z-slice where organ_id is present:
        * HU-window and convert the image slice to uint8 RGB (H, W, 3).
        * Resize image to (image_size, image_size) using PIL BILINEAR.
        * Extract the binary GT mask; resize with PIL NEAREST to preserve labels.
        * Compute bbox_from_mask on the resized GT; skip if None.
        * Store (resized_img_array, resized_gt_array, bbox_array) in self.samples.

    __getitem__(idx):
    - Return (img_tensor, gt_tensor, box_tensor) where:
        img_tensor : torch.float32, shape (image_size, image_size, 3), range [0, 1]
        gt_tensor  : torch.float32, shape (image_size, image_size), binary 0/1
        box_tensor : torch.float32, shape (4,), [x0, y0, x1, y1]

    Args:
        cases      (list[str]):  Case stem names, e.g. ['img0001', 'img0002'].
        organ_id   (int):        BTCV integer label for the target organ.
        image_dir  (str | Path): Folder containing img*.nii files.
        label_dir  (str | Path): Folder containing label*.nii files.
        image_size (int):        Square spatial size for resizing (default 1024).
    """

    def __init__(self, cases, organ_id, image_dir, label_dir, image_size=1024):
        self.samples = []
        # TODO: implement as described above
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: unpack self.samples[idx] and return tensors as specified above
        pass
