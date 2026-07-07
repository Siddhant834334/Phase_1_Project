import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_volume(path):
    """Load a NIfTI file and return the 3D volume, affine, and voxel spacing.

    TODO:
    1. Load the file:   img = nib.load(path)
    2. Reorient to standard axes so axis-2 is always axial (z):
           img = nib.as_closest_canonical(img)
    3. Get the numpy array:  vol = img.get_fdata().astype(np.float32)
    4. Read voxel spacing in mm from the header:
           spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
       IMPORTANT: CT is anisotropic — x/y spacing ≈ 1 mm, z spacing ≈ 3-5 mm.
       Print spacing when you first load a volume so you can see this.
    5. Return (vol, img.affine, spacing)

    Args:
        path: path to a .nii or .nii.gz file

    Returns:
        vol     : float32 numpy array of shape (H, W, Z)
        affine  : 4x4 affine matrix from the NIfTI header
        spacing : tuple (sx, sy, sz) — voxel size in mm
    """
    img=nib.load(path)
    img=nib.as_closest_canonical(img)
    vol=img.get_fdata().astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])

    return (vol, img.affine, spacing)


def apply_hu_window(vol, lo=-150, hi=250, as_uint8=True):
    """Clip a CT volume to a soft-tissue HU window and rescale to [0, 255] or [0, 1].

    TODO:
    1. Clip:     v = np.clip(vol, lo, hi)
    2. Rescale:  v = (v - lo) / (hi - lo)   →  values now in [0, 1]
    3. If as_uint8 is True:  return (v * 255).astype(np.uint8)
       If as_uint8 is False: return v.astype(np.float32)

    Args:
        vol      : float32 volume array (H, W, Z) in Hounsfield Units
        lo       : lower clip bound  (default -150 HU)
        hi       : upper clip bound  (default  250 HU)
        as_uint8 : True  → return uint8 [0, 255]  (for PNG frames)
                   False → return float [0, 1]     (for direct array feeding)

    Returns:
        Windowed array, same shape as vol
    """
    v = np.clip(vol, lo, hi)
    v = (v - lo) / (hi - lo)
    if (as_uint8==True):
      return (v * 255).astype(np.uint8)
    else:
      return v.astype(np.float32)


def to_rgb(slice2d_u8):
    """Convert a (H, W) uint8 greyscale slice to (H, W, 3) by repeating the channel.

    TODO:
    1. Add a channel dimension: slice2d_u8[..., None]  →  shape (H, W, 1)
    2. Repeat 3 times along axis 2:
           np.repeat(slice2d_u8[..., None], 3, axis=2)  →  shape (H, W, 3)
    3. Return the result

    SAM 2's ViT image encoder expects 3-channel RGB input even for greyscale CT.

    Args:
        slice2d_u8: 2D uint8 array of shape (H, W)

    Returns:
        uint8 array of shape (H, W, 3)
    """
    return np.repeat(slice2d_u8[..., None], 3, axis=2)

def resample_isotropic(vol, spacing, target=1.5, order=1):
    """Resample a volume to isotropic voxel spacing using scipy zoom.

    OPTIONAL — only call this if cfg.preprocess.resample_isotropic is True.

    TODO:
    1. Compute zoom factors:  factors = [s / target for s in spacing]
    2. Apply zoom:  resampled = zoom(vol, factors, order=order)
       - Use order=1 for image volumes (bilinear interpolation)
       - Use order=0 for label/mask volumes (nearest-neighbour — no blurring)
    3. Return (resampled, (target, target, target))

    Args:
        vol     : input volume array (H, W, Z)
        spacing : current voxel spacing in mm — e.g. (0.97, 0.97, 3.0)
        target  : desired isotropic spacing in mm (default 1.5 mm)
        order   : interpolation order (1 for images, 0 for masks)

    Returns:
        resampled volume, new spacing tuple (target, target, target)
    """
    pass
