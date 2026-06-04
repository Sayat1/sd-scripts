
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Any
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DifferentiableDepthEncoder(nn.Module):
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        input_size: int = 518,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation

        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if grad_checkpoint:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.input_size = input_size
        if device is not None:
            self.to(device)

    def _aspect_preserving_hw(self, H: int, W: int) -> Tuple[int, int]:
        if H >= W:
            new_h = self.input_size
            new_w = max(14, int(round(W * self.input_size / H / 14)) * 14)
        else:
            new_w = self.input_size
            new_h = max(14, int(round(H * self.input_size / W / 14)) * 14)
        return new_h, new_w

    def preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        # Assume pixels in [0, 1]
        pixels = pixels.clamp(0.0, 1.0)
        _, _, H, W = pixels.shape
        new_h, new_w = self._aspect_preserving_hw(H, W)
        x = F.interpolate(
            pixels,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.to(self.mean.dtype)
        x = (x - self.mean) / self.std
        return x.to(next(self.model.parameters()).dtype)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        _, _, H, W = pixels.shape
        x = self.preprocess(pixels)
        out = self.model(pixel_values=x)
        d = out.predicted_depth.float()
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).squeeze(1)
        return d

def ssi_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    p = pred.flatten(1)
    g = target.flatten(1)
    m = mask.flatten(1).float()
    n = m.sum(dim=1).clamp_min(1.0)
    
    mean_p = (p * m).sum(1) / n
    mean_g = (g * m).sum(1) / n
    var_p = (p * p * m).sum(1) / n - mean_p * mean_p
    cov_pg = (p * g * m).sum(1) / n - mean_p * mean_g
    
    s = cov_pg / var_p.clamp_min(1e-6)
    t = mean_g - s * mean_p
    
    aligned = s.view(-1, 1, 1) * pred + t.view(-1, 1, 1)
    diff = (aligned - target).abs() * mask
    loss = diff.sum() / mask.sum().clamp_min(1.0)
    return loss, s.detach(), t.detach()

def multiscale_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: int = 4,
) -> torch.Tensor:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    loss = pred.new_zeros(())
    p, g, m = pred, target, mask.float()
    for k in range(scales):
        if k > 0:
            p = F.avg_pool2d(p.unsqueeze(1), 2).squeeze(1)
            g = F.avg_pool2d(g.unsqueeze(1), 2).squeeze(1)
            m = F.avg_pool2d(m.unsqueeze(1), 2).squeeze(1)
        diff = p - g
        mx = m[:, :, 1:] * m[:, :, :-1]
        my = m[:, 1:, :] * m[:, :-1, :]
        dx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * mx
        dy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * my
        loss = loss + (dx.sum() / mx.sum().clamp_min(1.0)) + (dy.sum() / my.sum().clamp_min(1.0))
    return loss / scales

def compute_depth_consistency_loss(
    encoder: DifferentiableDepthEncoder,
    x0_pixels: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ssi_weight: float = 1.0,
    grad_weight: float = 0.5,
    grad_scales: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_pred = encoder(x0_pixels)
    
    target = gt_depth
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.shape[-2:] != d_pred.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=d_pred.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
    target = target.to(d_pred.device, dtype=d_pred.dtype)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1) # (B, H, W) -> (B, 1, H, W)
        
        if mask.shape[-2:] != d_pred.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=d_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        if mask.shape[1] != 1:
            mask = mask[:, 0:1] # Ensure single channel

        if mask.shape[0] == 1 and d_pred.shape[0] > 1:
            mask = mask.expand(d_pred.shape[0], -1, -1, -1)
            
        mask = mask.to(d_pred.device, dtype=d_pred.dtype)
        # Squeeze back to (B, H, W) if ssi_l1 expects that, but let's check ssi_l1
        mask = mask.squeeze(1)

    ssi, s_align, t_align = ssi_l1(d_pred, target, mask)
    d_pred_aligned = s_align.view(-1, 1, 1) * d_pred + t_align.view(-1, 1, 1)
    grd = multiscale_grad_loss(d_pred_aligned, target, mask, scales=grad_scales)
    loss = ssi_weight * ssi + grad_weight * grd
    return loss, ssi.detach(), grd.detach(), d_pred.detach(), target.detach()

def decode_latents_to_pixels(vae: Any, latents: torch.Tensor, model_type: str, vae_batch_size: Optional[int] = None) -> torch.Tensor:
    if model_type == "anima":
        # anima VAE expects [-1, 1] output and has decode_to_pixels
        pixels = vae.decode_to_pixels(latents) # [-1, 1]
        pixels = (pixels + 1.0) * 0.5
    else:
        # Standard diffusers VAE
        pixels = vae.decode(latents).sample # [-1, 1]
        pixels = (pixels + 1.0) * 0.5
    return pixels.clamp(0.0, 1.0)

class DepthConsistencyManager:
    def __init__(
        self,
        args,
        accelerator,
        weight_dtype,
    ):
        self.args = args
        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.encoder: Optional[DifferentiableDepthEncoder] = None
        
        if args.depth_consistency_weight > 0 or args.depth_consistency_preview_every > 0:
            self.encoder = DifferentiableDepthEncoder(
                model_id=args.depth_consistency_model_id,
                dtype=weight_dtype,
                device=accelerator.device,
                grad_checkpoint=args.gradient_checkpointing
            )

    @staticmethod
    def _to_model_device(module: Any, device: torch.device):
        try:
            if next(module.parameters()).device != device:
                module.to(device)
        except StopIteration:
            pass

    def cache_depths_from_latent_cache_batch(
        self,
        image_infos,
        img_tensor: torch.Tensor,
        latents_tensor: torch.Tensor,
        vae,
        model_type,
        variation_index: Optional[int] = None,
    ):
        if self.encoder is None:
            return

        self.encoder.to(self.accelerator.device)
        self.encoder.eval()

        with torch.no_grad():
            if vae is not None:
                self._to_model_device(vae, self.accelerator.device)
                latents_tensor = latents_tensor.to(self.accelerator.device, dtype=next(vae.parameters()).dtype)
                pixels = decode_latents_to_pixels(
                    vae,
                    latents_tensor,
                    model_type,
                    getattr(self.args, "vae_batch_size", None),
                )
            else:
                pixels = ((img_tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)
                pixels = pixels.to(self.accelerator.device)

            depths = self.encoder(pixels).cpu().half()

        for i, image_info in enumerate(image_infos):
            depth = depths[i]
            if variation_index is None:
                image_info.depth_gt = depth
            else:
                if not isinstance(image_info.depth_gt, list):
                    image_info.depth_gt = []
                while len(image_info.depth_gt) <= variation_index:
                    image_info.depth_gt.append(None)
                image_info.depth_gt[variation_index] = depth

    def cache_depths(self, dataset_group, vae, model_type):
        if self.encoder is None or not hasattr(dataset_group, "image_data"):
            return
        
        logger.info("Caching GT depth maps...")
        self.encoder.to(self.accelerator.device)
        self.encoder.eval()
        
        from library.train_util import load_image, resize_image, trim_and_resize_if_required
        
        for image_key, image_info in tqdm(dataset_group.image_data.items(), desc="Caching depth maps"):
            if hasattr(image_info, "depth_gt") and image_info.depth_gt is not None:
                continue
            
            # Original image
            image = load_image(image_info.absolute_path)
            
            # Check if latents are cached (with variations)
            if hasattr(image_info, "latents_crop_ltrb") and image_info.latents_crop_ltrb is not None:
                # Latents are cached. We should cache depth for EACH variation to match latents exactly.
                crop_ltrbs = image_info.latents_crop_ltrb
                if not isinstance(crop_ltrbs, list):
                    crop_ltrbs = [crop_ltrbs]
                
                depth_list = []
                for crop_ltrb in crop_ltrbs:
                    # Resize to resized_size then crop
                    img_np = np.array(image)
                    img_np = resize_image(img_np, img_np.shape[1], img_np.shape[0], image_info.resized_size[0], image_info.resized_size[1], image_info.resize_interpolation)
                    img_np = img_np[crop_ltrb[1] : crop_ltrb[3], crop_ltrb[0] : crop_ltrb[2]]
                    
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    img_tensor = img_tensor.to(self.accelerator.device,dtype=vae.dtype if vae is not None else torch.float32)
                    
                    # Roundtrip VAE if needed. Use deterministic latents so GT
                    # depth matches the cleanest image the trainer can decode.
                    if vae is not None:
                        self._to_model_device(vae, self.accelerator.device)
                        with torch.no_grad():
                            if model_type == "anima":
                                latents = vae.encode_pixels_to_latents(img_tensor)
                            else:
                                posterior = vae.encode(img_tensor * 2.0 - 1.0)
                                latents = posterior.latent_dist.mode()
                            img_tensor = decode_latents_to_pixels(vae, latents, model_type)
                    
                    with torch.no_grad():
                        depth = self.encoder(img_tensor)[0].cpu().half()
                    depth_list.append(depth)
                
                image_info.depth_gt = depth_list if len(depth_list) > 1 else depth_list[0]
            else:
                # Latents NOT cached (on-the-fly loading). 
                # Cache depth for the "full" resized image (un-cropped).
                # __getitem__ will crop it using crop_ltrb.
                img_np = np.array(image)
                img_np = resize_image(img_np, img_np.shape[1], img_np.shape[0], image_info.resized_size[0], image_info.resized_size[1], image_info.resize_interpolation)
                
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.accelerator.device,dtype=vae.dtype if vae is not None else torch.float32)
                
                # Roundtrip VAE? On-the-fly usually doesn't do this during caching, but let's be consistent.
                if vae is not None:
                    self._to_model_device(vae, self.accelerator.device)
                    with torch.no_grad():
                        if model_type == "anima":
                            latents = vae.encode_pixels_to_latents(img_tensor)
                        else:
                            posterior = vae.encode(img_tensor * 2.0 - 1.0)
                            latents = posterior.latent_dist.mode().to(dtype=vae.dtype)
                        img_tensor = decode_latents_to_pixels(vae, latents, model_type)
                
                with torch.no_grad():
                    depth = self.encoder(img_tensor)[0].cpu().half()
                image_info.depth_gt = depth

    def compute_loss(
        self,
        x0_pixels,
        gt_depth_list,
        mask=None,
        active_mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        preview: bool = False,
    ):
        if self.encoder is None:
            return None, None, None, None, None
        
        total_loss = x0_pixels.new_zeros(())
        weighted_loss = x0_pixels.new_zeros(())
        ssi_sum = 0.0
        grad_sum = 0.0
        first_d_pred = None
        first_d_gt = None
        count = 0
        
        for i in range(len(gt_depth_list)):
            if active_mask is not None and (i >= active_mask.shape[0] or not bool(active_mask[i])):
                continue
            gt_depth = gt_depth_list[i].to(x0_pixels.device, dtype=torch.float32)

            if preview and weights is not None and float(weights[i]) <= 0:
                with torch.no_grad():
                    loss, ssi, grd, d_pred, target = compute_depth_consistency_loss(
                        self.encoder,
                        x0_pixels[i:i+1].detach(),
                        gt_depth,
                        mask[i:i+1] if mask is not None else None,
                        ssi_weight=getattr(self.args, "depth_consistency_ssi_weight", 1.0),
                        grad_weight=getattr(self.args, "depth_consistency_grad_weight", 0.5),
                        grad_scales=getattr(self.args, "depth_consistency_grad_scales", 4),
                    )
            else:
                loss, ssi, grd, d_pred, target = compute_depth_consistency_loss(
                    self.encoder,
                    x0_pixels[i:i+1],
                    gt_depth,
                    mask[i:i+1] if mask is not None else None,
                    ssi_weight=getattr(self.args, "depth_consistency_ssi_weight", 1.0),
                    grad_weight=getattr(self.args, "depth_consistency_grad_weight", 0.5),
                    grad_scales=getattr(self.args, "depth_consistency_grad_scales", 4),
                )

            weight = weights[i].to(loss.device, dtype=loss.dtype) if weights is not None else loss.new_tensor(1.0)
            total_loss = total_loss + loss
            weighted_loss = weighted_loss + loss * weight
            ssi_sum += ssi.item()
            grad_sum += grd.item()
            count += 1
            if first_d_pred is None:
                first_d_pred = d_pred[0]
                first_d_gt = target[0]
            
        if count == 0:
            return None, None, None, None, None

        return weighted_loss / count, ssi_sum / count, grad_sum / count, first_d_pred, first_d_gt

def render_depth_preview(depth: torch.Tensor) -> Image.Image:
    # depth is [H, W]
    d = depth.cpu().float().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d = (d * 255).astype(np.uint8)
    return Image.fromarray(d)

def save_depth_comparison(
    x0_pixels: torch.Tensor,
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    output_path: str
):
    # x0_pixels: [3, H, W], d_pred: [H, W], d_gt: [H, W]
    p = (x0_pixels.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    dp = render_depth_preview(d_pred)
    dg = render_depth_preview(d_gt)
    
    # Resize depth to match pixels
    H, W = p.shape[:2]
    dp = dp.resize((W, H), Image.NEAREST)
    dg = dg.resize((W, H), Image.NEAREST)
    
    # Combine
    combined = Image.new("RGB", (W * 3, H))
    combined.paste(Image.fromarray(p), (0, 0))
    combined.paste(dp.convert("RGB"), (W, 0))
    combined.paste(dg.convert("RGB"), (W * 2, 0))
    combined.save(output_path)

import logging
logger = logging.getLogger(__name__)
