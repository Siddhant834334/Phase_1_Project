# [Week 3] src/engine/predictor.py
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(model_cfg, ckpt_path):
    """Build a SAM 2 video predictor for bidirectional z-axis propagation.

    Specification:
    - Call build_sam2_video_predictor(model_cfg, ckpt_path, device="cuda").
    - model_cfg is the config filename (e.g. "sam2.1_hiera_t.yaml"), not a path.
    - SAM 2 resolves it relative to its installed sam2/configs/ directory.
    - Return the predictor; do NOT call init_state here.

    Model size → (cfg, ckpt) mapping (from configs/default.yaml):
        tiny   : sam2.1_hiera_t.yaml   / checkpoints/sam2.1_hiera_tiny.pt
        small  : sam2.1_hiera_s.yaml   / checkpoints/sam2.1_hiera_small.pt
        base+  : sam2.1_hiera_b+.yaml  / checkpoints/sam2.1_hiera_base_plus.pt

    Args:
        model_cfg  (str): SAM 2 config filename (not a full path).
        ckpt_path  (str): Absolute or relative path to the .pt checkpoint.

    Returns:
        SAM2VideoPredictor: Predictor ready for init_state.
    """
    pass


def build_image_predictor(model_cfg, ckpt_path):
    """Build a SAM 2 image predictor for single-frame prompted segmentation.

    Specification:
    - Call build_sam2(model_cfg, ckpt_path, device="cuda") to get the base model.
    - Wrap it in SAM2ImagePredictor(model).
    - Used by the LoRA training path (Week 4) instead of the video predictor.

    Args:
        model_cfg (str): SAM 2 config filename.
        ckpt_path (str): Path to the .pt checkpoint.

    Returns:
        SAM2ImagePredictor
    """
    pass


def init_state(predictor, frames_dir):
    """Initialise the SAM 2 video memory state for a frame folder.

    Specification:
    - Call predictor.init_state(
          video_path=frames_dir,
          offload_video_to_cpu=True,   # mandatory on 8 GB VRAM
          offload_state_to_cpu=True,   # mandatory on 8 GB VRAM
      )
    - Return the inference state dict.
    - frames_dir must contain the 00000.png, 00001.png, … files from
      write_frames_png (Week 2).

    Args:
        predictor  (SAM2VideoPredictor): Built by build_predictor.
        frames_dir (str | Path):         Folder of numbered PNG frames.

    Returns:
        dict: SAM 2 inference state (opaque — pass directly to propagate).
    """
    pass
