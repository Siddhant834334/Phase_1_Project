# [Week 5] src/eval/evaluate.py
# Dependencies: all previous weeks (weeks 1-4)
import numpy as np
from pathlib import Path
import torch

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.prompts import bbox_from_mask, best_start_slice
from src.data.volume_to_frames import write_frames_png
from src.engine.predictor import build_predictor, init_state
from src.engine.propagate import propagate_bidirectional
from src.eval.metrics import dice_score, hd95
from src.utils.viz import save_overlay_gif


def _load_predictor(cfg, lora_path=None):
    """Build a SAM 2 video predictor and optionally load a LoRA adapter.

    Specification:
    - predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    - If lora_path is not None and the file exists:
          adapter = torch.load(lora_path, map_location="cuda")
          predictor.model.image_encoder.load_state_dict(adapter, strict=False)
    - Return predictor.

    Args:
        cfg       (dict):      Config dict from configs/default.yaml.
        lora_path (str|None):  Path to a saved LoRA .pt file, or None.

    Returns:
        SAM2VideoPredictor (with LoRA weights loaded if lora_path given).
    """
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    
    if lora_path is not None and Path(lora_path).exists():
        adapter = torch.load(lora_path, map_location="cuda")
        predictor.model.image_encoder.load_state_dict(adapter, strict=False)
        print(f"Loaded LoRA adapter from {lora_path}")
        
    return predictor


def evaluate_organ(cfg, organ_id, lora_path=None):
    """Run zero-shot and optional LoRA inference on all validation cases.

    Specification:
    - val_cases = cfg["split"]["val_cases"]
    - img_dir   = Path(cfg["paths"]["raw_images"])
    - lbl_dir   = Path(cfg["paths"]["raw_labels"])
    - pred_zs   = _load_predictor(cfg)
    - pred_lora = _load_predictor(cfg, lora_path) if lora_path else None

    For each case in val_cases:
      a. Paths:
             img_path   = img_dir / f"{case}.nii"
             lbl_path   = lbl_dir / f"{case.replace('img','label')}.nii"
             frames_dir = Path(cfg["paths"]["processed"]) / case
             out_dir    = Path("outputs") / case ; out_dir.mkdir(parents=True, exist_ok=True)
         Skip if either file does not exist.

      b. Write PNG frames if not already done.

      c. Load volume and labels:
             vol, _, spacing = load_volume(img_path)
             lbl, _, _       = load_volume(lbl_path)
             vol_u8          = apply_hu_window(vol)
             gt3d            = (lbl == organ_id).astype(bool)
         Skip if gt3d has no True pixels (organ absent in this case).

      d. Find prompt:
             start_z  = best_start_slice(lbl, organ_id)
             gt_slice = (lbl[:,:,start_z] == organ_id).astype(np.uint8)
             bbox     = bbox_from_mask(gt_slice, pad=4)
         Skip if bbox is None.

      e. Zero-shot inference:
             state_zs = init_state(pred_zs, str(frames_dir))
             mask_zs  = propagate_bidirectional(pred_zs, state_zs, start_z, bbox,
                                                target_hw=(vol.shape[0], vol.shape[1]))
             dsc_zs = dice_score(mask_zs, gt3d)
             hd_zs  = hd95(mask_zs, gt3d, spacing)
         Save GIF to out_dir / f"zeroshot_{organ_id}.gif".

      f. LoRA inference (only if pred_lora is not None):
             state_lo = init_state(pred_lora, str(frames_dir))
             mask_lo  = propagate_bidirectional(pred_lora, state_lo, start_z, bbox,
                                                target_hw=(vol.shape[0], vol.shape[1]))
             dsc_lo = dice_score(mask_lo, gt3d)
             hd_lo  = hd95(mask_lo, gt3d, spacing)
         If pred_lora is None, set dsc_lo = hd_lo = float("nan").

      g. Append dict(case=case, dsc_zs=dsc_zs, hd95_zs=hd_zs,
                     dsc_lora=dsc_lo, hd95_lora=hd_lo) to results.

    Args:
        cfg       (dict):     Config dict from configs/default.yaml.
        organ_id  (int):      BTCV integer label for the target organ.
        lora_path (str|None): LoRA adapter .pt path, or None for zero-shot only.

    Returns:
        list[dict]: One dict per processed val case with keys:
                    case, dsc_zs, hd95_zs, dsc_lora, hd95_lora.
    """
    pred_zs = build_predictor(cfg)
    pred_lora = None
    if lora_path is not None:
        pred_lora = build_predictor(cfg, ckpt_path=lora_path) 
        
    results = []
    
    val_cases = cfg.get("val_cases", [])
    
    for case in val_cases:
        img_path = Path(cfg["data_dir"]) / "imagesTr" / f"{case}.nii"
        lbl_path = Path(cfg["data_dir"]) / "labelsTr" / f"{case}.nii"
        frames_dir = Path(cfg["output_dir"]) / "frames" / case
        out_dir = Path(cfg["output_dir"]) / "results" / case
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        write_frames_png(img_path, frames_dir)
        
        vol, _ = load_volume(img_path)
        lbl, spacing = load_volume(lbl_path)
        
        vol = apply_hu_window(vol, window_min=-125, window_max=275) 
        
        gt3d = (lbl == organ_id).astype(np.uint8)

        start_z = best_start_slice(lbl, organ_id)
        
        if start_z is None:
            continue
            
        gt_slice = (lbl[:, :, start_z] == organ_id).astype(np.uint8)
        bbox = bbox_from_mask(gt_slice, pad=4)
        
        if bbox is None:
            continue

        state_zs = init_state(pred_zs, str(frames_dir))
        mask_zs  = propagate_bidirectional(
            pred_zs, 
            state_zs, 
            start_z, 
            bbox,
            target_hw=(vol.shape[0], vol.shape[1])
        )
        dsc_zs = dice_score(mask_zs, gt3d)
        hd_zs  = hd95(mask_zs, gt3d, spacing)
        

        if pred_lora is not None:
            state_lo = init_state(pred_lora, str(frames_dir))
            mask_lo  = propagate_bidirectional(
                pred_lora, 
                state_lo, 
                start_z, 
                bbox,
                target_hw=(vol.shape[0], vol.shape[1])
            )
            dsc_lo = dice_score(mask_lo, gt3d)
            hd_lo  = hd95(mask_lo, gt3d, spacing)
            
            # save_gif(mask_lo, out_dir / f"lora_{organ_id}.gif")
        else:
            dsc_lo = float("nan")
            hd_lo  = float("nan")

        # g. Append results
        results.append(dict(
            case=case, 
            dsc_zs=dsc_zs, 
            hd95_zs=hd_zs, 
            dsc_lora=dsc_lo, 
            hd95_lora=hd_lo
        ))
        
    return results


def print_table(rows, organ_id):
    """Print a formatted DSC / HD95 summary table and return aggregated metrics.

    Specification:
    - Compute mean and std for each metric across all rows, ignoring NaN values
      (use np.nanmean and np.nanstd).
    - Print in this exact format:
          ============================================================
            Organ {organ_id}  |  {n} val cases
          ============================================================
            Metric         Zero-shot           LoRA
            --------------------------------------------------
            DSC            {mean:.3f} ± {std:.3f}   {mean:.3f} ± {std:.3f}
            HD95 (mm)      {mean:.1f}  ± {std:.1f}   {mean:.1f}  ± {std:.1f}
          ============================================================
    - Return a dict with keys:
          organ, dsc_zs_mean, dsc_zs_std, dsc_lora_mean, dsc_lora_std,
          hd95_zs_mean, hd95_zs_std, hd95_lora_mean, hd95_lora_std

    Args:
        rows     (list[dict]): Output of evaluate_organ().
        organ_id (int):        Organ label integer (for display).

    Returns:
        dict: Aggregated metric statistics.
    """
    dsc_zs = [r.get("dsc_zs", np.nan) for r in rows]
    hd95_zs = [r.get("hd95_zs", np.nan) for r in rows]
    dsc_lora = [r.get("dsc_lora", np.nan) for r in rows]
    hd95_lora = [r.get("hd95_lora", np.nan) for r in rows]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        dsc_zs_m, dsc_zs_s = np.nanmean(dsc_zs), np.nanstd(dsc_zs)
        hd95_zs_m, hd95_zs_s = np.nanmean(hd95_zs), np.nanstd(hd95_zs)
        
        dsc_lo_m, dsc_lo_s = np.nanmean(dsc_lora), np.nanstd(dsc_lora)
        hd95_lo_m, hd95_lo_s = np.nanmean(hd95_lora), np.nanstd(hd95_lora)
        
    n = len(rows)

    print("=" * 60)
    print(f"  Organ {organ_id}  |  {n} val cases")
    print("=" * 60)
    print("  Metric         Zero-shot           LoRA")
    print("  " + "-" * 50)
    print(f"  DSC            {dsc_zs_m:.3f} ± {dsc_zs_s:.3f}   {dsc_lo_m:.3f} ± {dsc_lo_s:.3f}")
    print(f"  HD95 (mm)      {hd95_zs_m:.1f}  ± {hd95_zs_s:.1f}   {hd95_lo_m:.1f}  ± {hd95_lo_s:.1f}")
    print("=" * 60)

    return {
        "organ": organ_id,
        "dsc_zs_mean": dsc_zs_m,
        "dsc_zs_std": dsc_zs_s,
        "dsc_lora_mean": dsc_lo_m,
        "dsc_lora_std": dsc_lo_s,
        "hd95_zs_mean": hd95_zs_m,
        "hd95_zs_std": hd95_zs_s,
        "hd95_lora_mean": hd95_lo_m,
        "hd95_lora_std": hd95_lo_s,
    }
