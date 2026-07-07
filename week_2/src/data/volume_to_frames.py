# [Week 2] src/data/volume_to_frames.py
# Dependency: src/data/nifti_io.py (Week 1)
import os
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb


def write_frames_png(nifti_path, out_dir):
    """Convert every axial slice of a CT volume to a numbered PNG file on disk.

    Specification:
    - Load and HU-window the volume using Week 1 utilities.
    - Convert each (H, W) greyscale slice to (H, W, 3) RGB via to_rgb().
    - Save as PNG named 00000.png, 00001.png, … (5-digit zero-padded index = z).
    - Create out_dir if it does not exist.
    - SAM 2's video predictor reads this exact folder structure.

    Args:
        nifti_path (str | Path): Source .nii or .nii.gz file.
        out_dir    (str | Path): Destination folder for PNG frames.

    Returns:
        int: Number of PNGs written (equals the Z depth of the volume).
    """
    pass


def frames_to_arrays(frames_dir):
    """Load all PNG frames from a folder into a single uint8 numpy array.

    Specification:
    - Read every .png in frames_dir in sorted (filename) order.
    - Stack along a new leading axis → shape (Z, H, W, 3).

    Args:
        frames_dir (str | Path): Folder produced by write_frames_png.

    Returns:
        np.ndarray: uint8 array of shape (Z, H, W, 3).
    """
    pass
