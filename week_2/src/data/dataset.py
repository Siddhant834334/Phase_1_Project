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
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        for case in cases:
            clean_case = case.replace("img", "")
            img_path = image_dir / f"img{clean_case}.nii"
            if not img_path.exists():
                img_path = image_dir / f"img{clean_case}.nii.gz"

            label_path = label_dir / f"label{clean_case}.nii"
            if not label_path.exists():
                label_path = label_dir / f"label{clean_case}.nii.gz"

            vol_img, _, _ = load_volume(img_path)
            vol_lbl, _, _ = load_volume(label_path)

            vol_img_u8 = apply_hu_window(vol_img, lo=-150, hi=250, as_uint8=True)

            z_depth = vol_img_u8.shape[2]
            for z in range(z_depth):
                lbl_slice = vol_lbl[:, :, z]

                if np.any(lbl_slice == organ_id):
                    gt_mask = (lbl_slice == organ_id).astype(np.uint8)

                    img_slice = to_rgb(vol_img_u8[:, :, z])

                    img_pil = Image.fromarray(img_slice).resize(
                        (image_size, image_size), resample=Image.BILINEAR
                    )
                    resized_img = np.array(img_pil, dtype=np.uint8)

                    gt_pil = Image.fromarray(gt_mask).resize(
                        (image_size, image_size), resample=Image.NEAREST
                    )
                    resized_gt = np.array(gt_pil, dtype=np.uint8)

                    bbox = bbox_from_mask(resized_gt, pad=4)

                    if bbox is None:
                        continue

                    self.samples.append((resized_img, resized_gt, bbox))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: unpack self.samples[idx] and return tensors as specified above
        img_arr, gt_arr, box_arr = self.samples[idx]

        img_tensor = torch.from_numpy(img_arr).to(torch.float32) / 255.0

        gt_tensor = torch.from_numpy(gt_arr).to(torch.float32)

        box_tensor = torch.from_numpy(box_arr).to(torch.float32)

        return img_tensor, gt_tensor, box_tensor
