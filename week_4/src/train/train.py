# [Week 4] src/train/train.py
# Dependencies: src/train/losses.py (Week 4), src/data/dataset.py (Week 2)
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from src.train.losses import total_loss

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def _normalize(img_t):
    """Normalise (B, 3, H, W) float [0,1] tensor to ImageNet mean/std. (complete — do not change)"""
    mean = _MEAN.to(img_t.device).view(1, 3, 1, 1)
    std  = _STD.to(img_t.device).view(1, 3, 1, 1)
    return (img_t - mean) / std


def forward_prompted(model, img_t_norm, box_batch, image_size):
    """One SAM 2 forward pass through the internal path that keeps LoRA gradients.

    Specification:
    SAM2ImagePredictor.set_image() runs under torch.no_grad() — it kills gradients.
    Use the SAM2Base internal path instead:

    1. Encode backbone:
           backbone_out = model.forward_image(img_t_norm)
           _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    2. Reshape vision_feats from (H*W, B, C) to (B, C, H, W) per scale:
           feats = [f.permute(1,2,0).view(B, -1, *sz)
                    for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])]
           feats = feats[::-1]          # restore finest-first order
           image_embed    = feats[-1]   # coarsest scale — main embedding
           high_res_feats = feats[:-1]  # finer scales

    3. Encode the box prompt:
           box_t = box_batch.to(img_t_norm.device).view(B, 1, 4)
           sparse_emb, dense_emb = model.sam_prompt_encoder(
               points=None, boxes=box_t, masks=None)

    4. Decode to low-resolution logits:
           dec_kwargs = dict(
               image_embeddings=image_embed,
               image_pe=model.sam_prompt_encoder.get_dense_pe(),
               sparse_prompt_embeddings=sparse_emb,
               dense_prompt_embeddings=dense_emb,
               multimask_output=False,
               repeat_image=False,
           )
           if high_res_feats:
               dec_kwargs["high_res_features"] = high_res_feats
           low_res_masks = model.sam_mask_decoder(**dec_kwargs)[0]  # (B, 1, h, w)

    5. Upsample and squeeze channel dim:
           return F.interpolate(low_res_masks, (image_size, image_size),
                                mode="bilinear", align_corners=False).squeeze(1)
           # output shape: (B, H, W)

    Args:
        model        : SAM2Base (predictor.model) with LoRA injected.
        img_t_norm   : (B, 3, H, W) float tensor, ImageNet-normalised.
        box_batch    : (B, 4) float tensor, xyxy pixel coords at image_size scale.
        image_size   : int — spatial size matching cfg["model"]["image_size"].

    Returns:
        Tensor: (B, H, W) logit tensor with gradient attached.
    """
    B = img_t_norm.shape[0]
    
    backbone_out = model.forward_image(img_t_norm)
    
  
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    
    image_embed = vision_feats[-1]
    high_res_feats = vision_feats[:-1]
    

    box_t = box_batch.to(img_t_norm.device).view(B, 1, 4)
    
    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=None, boxes=box_t, masks=None
    )

    dec_kwargs = dict(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
    )
    
    if high_res_feats:
        dec_kwargs["high_res_features"] = high_res_feats
        
    low_res_masks = model.sam_mask_decoder(**dec_kwargs)[0]  # (B, 1, h, w)

    return F.interpolate(
        low_res_masks, 
        (image_size, image_size),
        mode="bilinear", 
        align_corners=False
    ).squeeze(1)


def lora_state_dict(model):
    """Extract only LoRA adapter weights from the model state dict.

    Specification:
    - Iterate model.image_encoder.state_dict().items().
    - Keep only keys that contain ".A", ".B", "lora_A", or "lora_B".
    - Return the filtered dict.
    - This is the file saved to checkpoints/lora_organ<N>.pt.
    - Saving from image_encoder ensures key names match what load_state_dict
      expects when the adapter is loaded back via image_encoder directly.

    Args:
        model: nn.Module with LoRA layers (either peft or custom LoRALinear).

    Returns:
        dict: Filtered state dict containing only LoRA weight tensors.
    """
    return {
        k: v for k, v in model.image_encoder.state_dict().items()
        if ".A" in k or ".B" in k or "lora_A" in k or "lora_B" in k
    }


def run_training(cfg, model, dataset, organ_id, save_path):
    """AMP + gradient-accumulation LoRA fine-tuning loop.

    Specification:
    1. DataLoader:
           loader = DataLoader(dataset, batch_size=cfg["train"]["micro_batch"],
                               shuffle=True, num_workers=0, pin_memory=False)

    2. Optimiser on LoRA params only:
           trainable = [p for p in model.parameters() if p.requires_grad]
           opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"])

    3. AMP GradScaler:
           scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    4. Zero gradients before the loop (set_to_none=True — Fix #7).

    5. For each epoch in range(cfg["train"]["epochs"]):
       model.train()
       For each (step, (img, gt, box)) in enumerate(loader):
         a. Move tensors to GPU; normalise image:
                img_t = _normalize(img.permute(0,3,1,2).float().to(device))
                gt    = gt.to(device, dtype=torch.float32)
                box   = box.to(device)
                # Note: BTCVSliceDataset returns rgb_f already in [0, 1] float.
                # _normalize converts [0,1] → ImageNet-normalised. No /255 here.
         b. bfloat16 autocast forward (Fix #6):
                with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"],
                                             dtype=torch.bfloat16):
                    logits = forward_prompted(model, img_t, box, cfg["model"]["image_size"])
                    loss   = total_loss(logits, gt) / cfg["train"]["accum_steps"]
         c. scaler.scale(loss).backward()
         d. Every accum_steps, step + update + zero_grad (Fix #3 + #7):
                if (step + 1) % cfg["train"]["accum_steps"] == 0:
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)

    6. Flush any leftover gradient after the last epoch.

    7. Save LoRA weights:
           Path(save_path).parent.mkdir(parents=True, exist_ok=True)
           torch.save(lora_state_dict(model), save_path)

    8. Return a list of mean loss values, one per epoch.

    Args:
        cfg       (dict):  Config loaded from configs/default.yaml.
        model     :        SAM2Base with LoRA injected (predictor.model).
        dataset   :        BTCVSliceDataset instance (Week 2).
        organ_id  (int):   BTCV label integer (used for logging only).
        save_path (str):   Path to save the LoRA .pt file.

    Returns:
        list[float]: Mean training loss per epoch.
    """
    device = torch.device(cfg["train"].get("device", "cuda"))
    model.to(device)
    model.train()
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg["train"]["batch_size"], 
        shuffle=True, 
        num_workers=cfg["train"].get("num_workers", 2)
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable_params, 
        lr=cfg["train"]["lr"], 
        weight_decay=cfg["train"].get("weight_decay", 1e-4)
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    
    mean_losses = []
    
    for epoch in range(cfg["train"]["epochs"]):
        epoch_loss = 0.0
        
        opt.zero_grad(set_to_none=True)
        
        for step, (img, gt, box) in enumerate(loader):
            
            img_t = _normalize(img.permute(0, 3, 1, 2).float().to(device))
            gt    = gt.to(device, dtype=torch.float32)
            box   = box.to(device)

            
            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"], dtype=torch.bfloat16):
                logits = forward_prompted(model, img_t, box, cfg["model"]["image_size"])
                loss   = total_loss(logits, gt) / cfg["train"]["accum_steps"]
                
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg["train"]["accum_steps"] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item() * cfg["train"]["accum_steps"]
            
        if len(loader) % cfg["train"]["accum_steps"] != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
        avg_epoch_loss = epoch_loss / len(loader)
        mean_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - Loss: {avg_epoch_loss:.4f}")
        
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), save_path)
    print(f"Saved LoRA weights to {save_path}")
    
    return mean_losses
