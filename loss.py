# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: loss.py
Description: Reference implementation of the P-JiT training objective, for readers of the
    blog post and code. This file is documentation in executable form: every term is written
    as it was computed during training, with the released model's hyperparameters as
    defaults. It is NOT a training script — no data, optimizer, or distributed machinery is
    shipped — and the frozen perception networks it references (DINOv3 ViT-L/16, LPIPS-VGG)
    are passed in as callables rather than shipped. The two training-only modules are
    implemented here and were verified against the internal ones, tensor for tensor, with
    the checkpoint's weights: the x-prediction readout heads after blocks 10 and 16, and the
    PixelREPA masked transformer adapter after block 5. Their weights are not released; they
    play no part in sampling.

    Notation: x_0 is the clean image in [-1, 1], eps ~ N(0, I), t in (0, 1) with t=0 clean and
    t=1 noise. Shapes are (B, 3, 1, H, W) — one frame — throughout. The training step ran
    entirely under bf16 autocast, with x_0, the noise and t all in bf16.

    Total loss (one trunk; the readouts and the adapter tap it, nothing is stop-gradiented):

        L = L_final + 1.0 * L_block10 + 1.0 * L_block16
              + 0.1 * L_pixel_repa + 0.1 * L_lpips + 0.01 * L_pdino
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyramid_jit.attention import VarlenMeta, build_varlen_metadata
from pyramid_jit.model import (
    OutputHead,
    PyramidJiT,
    SelfAttention,
    SwiGLUFFN,
    build_rope_table,
    build_text_mask,
    compute_rotary_frequencies,
    sinusoidal_embedding_1d,
    unpatchify,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LossConfig:
    """Objective hyperparameters of the released model."""

    noise_scale: float = 2.0          # x_1 = noise_scale * eps  (JiT-style scaled noise)
    t_eps: float = 0.1                # Clamp on t in the 1/t^2 velocity weighting
    t_min: float = 1e-3               # Sampled t is clamped to [t_min, 1 - t_min]
    num_timesteps: int = 1000         # Integer timestep fed to the model = int(t * 1000)
    # Timestep distribution: logit-normal, t = sigmoid(mean + std * z), in two phases by
    # optimizer step (global batch 1,024): (until_step, mean, std).
    #   Phase 1 (steps < 48,828, the first 50M samples): mean 0.8,  std 0.8  (noise-heavy)
    #   Phase 2 (the remaining 88M samples):              mean -0.2, std 1.0  (wider)
    timestep_phases: Tuple[Tuple[Optional[int], float, float], ...] = (
        (48_828, 0.8, 0.8),
        (None, -0.2, 1.0),
    )
    # x-prediction readouts: (1-indexed trunk block, spatial downsample, loss weight).
    # 512x512 -> 128x128 after block 10, 256x256 after block 16.
    readouts: Tuple[Tuple[int, int, float], ...] = ((10, 4, 1.0), (16, 2, 1.0))
    pixel_repa_weight: float = 0.1
    pixel_repa_mask_ratio: float = 0.2
    pixel_repa_block: int = 5         # Trunk block (1-indexed, of 22) whose output is aligned
    lpips_weight: float = 0.1
    pdino_weight: float = 0.01
    perceptual_gate_t: float = 0.7    # Perceptual terms only for samples with t <= 0.7
    text_dropout: float = 0.05        # Caption -> all-zeros embedding with this probability
    ema_half_life_steps: int = 3907   # EMA beta = exp(-ln 2 / half_life) = 0.99982260
    ema_start_step: int = 977


# --------------------------------
# FORWARD PROCESS
# --------------------------------

def sample_timesteps(
        batch_size: int,
        global_step: int,
        cfg: LossConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Logit-normal timesteps from the phase active at `global_step`, clamped away from 0 and 1.

    Args:
        batch_size (int):
            Number of samples.
        global_step (int):
            Optimizer step; a phase is active while global_step < its until_step.
        cfg (LossConfig):
            Objective hyperparameters.
        device (torch.device):
            Device.
        dtype (torch.dtype):
            Dtype of t; training drew it in x_0's dtype, bf16.

    Returns:
        torch.Tensor:
            t of shape (B,).
    """
    for until_step, mean, std in cfg.timestep_phases:
        if until_step is None or global_step < until_step:
            break
    z = torch.empty(batch_size, dtype=dtype, device=device).normal_()
    t = torch.sigmoid(z * std + mean)
    return t.clamp(min=cfg.t_min, max=1.0 - cfg.t_min)


def add_noise(
        x_0: torch.Tensor,
        t: torch.Tensor,
        cfg: LossConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear interpolation x_t = (1 - t) x_0 + t x_1 with scaled Gaussian noise x_1.

    Args:
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W) in [-1, 1] (bf16: uint8 / 127.5 - 1).
        t (torch.Tensor):
            Timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            x_t, and the integer timesteps int(t * num_timesteps) the model consumes.
    """
    x_1 = torch.empty_like(x_0).normal_() * cfg.noise_scale
    t5 = t.view(-1, 1, 1, 1, 1)
    x_t = (1 - t5) * x_0 + t5 * x_1
    return x_t, (t * cfg.num_timesteps).long()


def drop_text(text: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """
    Classifier-free-guidance dropout: whole captions replaced by zeros (lengths unchanged, so
    a dropped caption is `text_lens` zero rows, not an empty sequence).

    Args:
        text (torch.Tensor):
            Caption embeddings (B, L, D).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            Embeddings with a 5 % subset of samples zeroed.
    """
    keep = torch.bernoulli(torch.full((text.shape[0],), 1 - cfg.text_dropout, device=text.device))
    return text * keep.view(-1, 1, 1)


# --------------------------------
# DIFFUSION LOSSES (x-prediction, velocity-space MSE)
# --------------------------------

def velocity_mse(
        x_pred: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        cfg: LossConfig) -> torch.Tensor:
    """
    The model predicts x_0; the loss is the MSE of the implied velocity.

    With x_t = (1 - t) x_0 + t x_1 the true velocity is v = x_1 - x_0 and the predicted one
    is (x_t - x_pred) / t, so ||v_pred - v||^2 = ||x_pred - x_0||^2 / t^2. The 1/t^2 weight is
    clamped at t_eps (max weight 100x). Used unchanged for the final head and both readouts.

    Args:
        x_pred (torch.Tensor):
            Predicted clean image, same shape as x_0.
        x_0 (torch.Tensor):
            Target clean image.
        t (torch.Tensor):
            Continuous timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            Scalar loss (mean over all elements).
    """
    t5 = t.view(-1, 1, 1, 1, 1).clamp_min(cfg.t_eps)
    return (((x_pred - x_0) / t5) ** 2).mean()


def readout_target(x_0: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    A readout's regression target: x_0 antialiased-bilinear resized to the readout's grid.

    Args:
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W).
        size (Tuple[int, int]):
            (h, w) of the readout's prediction (H/4 x W/4 after block 10, H/2 x W/2 after
            block 16).

    Returns:
        torch.Tensor:
            (B, 3, 1, h, w) in x_0's dtype.
    """
    small = F.interpolate(
        x_0[:, :, 0], size=size, mode="bilinear", align_corners=False, antialias=True)
    return small.to(x_0.dtype).unsqueeze(2)


# --------------------------------
# READOUTS (deep supervision of the trunk)
# --------------------------------

def build_readout_heads(model: PyramidJiT, cfg: LossConfig) -> nn.ModuleDict:
    """
    One output head per readout (training-only; 2.85M parameters for the pair). Each is the
    model's own OutputHead (AdaLN-modulated LayerNorm + Linear, fed the same timestep
    embedding `e` as the final head) with a smaller output patch, so a 32x32 input patch
    decodes to (32/d)x(32/d) pixels: 8x8 after block 10, 16x16 after block 16.

    Args:
        model (PyramidJiT):
            The trunk (for width and patch).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        nn.ModuleDict:
            {"block10": OutputHead, "block16": OutputHead}.
    """
    pt, ph, pw = model.patch
    return nn.ModuleDict({
        f"block{block}": OutputHead(
            dim=model.config.dim, out_channels=model.config.out_channels,
            patch=(pt, ph // down, pw // down))
        for block, down, _ in cfg.readouts})


def apply_readout(
        head: OutputHead,
        tokens: torch.Tensor,
        e: torch.Tensor,
        grid: Tuple[int, int, int],
        image_fhw: Tuple[int, int, int],
        model_patch: Tuple[int, int, int],
        downsample: int) -> torch.Tensor:
    """
    Decode an x_0 prediction from one trunk block's image tokens. The tap is read-only: the
    trunk carries on from the same hidden state.

    Args:
        head (OutputHead):
            The readout's head.
        tokens (torch.Tensor):
            That block's output restricted to the image tokens, (B, L_img, dim).
        e (torch.Tensor):
            Timestep embedding (B, dim), the one the final head uses.
        grid (Tuple[int, int, int]):
            Image token grid (f_p, h_p, w_p).
        image_fhw (Tuple[int, int, int]):
            Unpadded input extent (F, H, W).
        model_patch (Tuple[int, int, int]):
            The model's input patch.
        downsample (int):
            Spatial downsample of this readout.

    Returns:
        torch.Tensor:
            (B, 3, F, H / downsample, W / downsample), cropped like the final head.
    """
    pt, ph, pw = model_patch
    out_patch = (pt, ph // downsample, pw // downsample)
    x = unpatchify(x_head=head(x=tokens, e=e), f_p=grid[0], h_p=grid[1], w_p=grid[2],
                   patch=out_patch, out_channels=3)
    f, h, w = image_fhw
    return x[:, :, :f, :h * out_patch[1] // ph, :w * out_patch[2] // pw]


# --------------------------------
# PIXEL-REPA (representation alignment on the trunk)
# --------------------------------

class AdapterBlock(nn.Module):
    """
    The adapter's DiT block: standard 6-chunk AdaLN (shift / scale / gate for attention and
    FFN, a per-block learned offset added to the shared projection), LayerNorm pre-norms,
    no sandwich norm. Attention and FFN are the model's own modules.
    """

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float):
        """
        Build the block.

        Args:
            dim (int):
                Width.
            ffn_dim (int):
                SwiGLU target width.
            num_heads (int):
                Attention heads.
            eps (float):
                q/k RMSNorm epsilon (the LayerNorms keep PyTorch's default 1e-5).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, eps=eps)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim=dim, hidden_dim=ffn_dim)
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(
            self,
            x: torch.Tensor,
            e: torch.Tensor,
            mask: torch.Tensor,
            rope_table: torch.Tensor,
            varlen_meta: VarlenMeta) -> torch.Tensor:
        """
        Apply the block.

        Args:
            x (torch.Tensor):
                Tokens (B, L, dim).
            e (torch.Tensor):
                AdaLN conditioning (B, 6, dim).
            mask (torch.Tensor):
                Validity mask (B, L).
            rope_table (torch.Tensor):
                Complex rotation table (L, head_dim/2).
            varlen_meta (VarlenMeta):
                Packing metadata for `mask`.

        Returns:
            torch.Tensor:
                Updated tokens (B, L, dim).
        """
        # The fp32 master offset is used at e's dtype (bf16 under autocast).
        e = (self.modulation.to(e.dtype) + e).chunk(6, dim=1)               # 6 x [B, 1, dim]
        y = self.attn(
            x=self.norm1(x) * (1 + e[1]) + e[0], mask=mask, rope_table=rope_table,
            varlen_meta=varlen_meta)
        x = x + y * e[2]
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        y = y * mask.unsqueeze(-1)
        return x + y * e[5]


class PixelRepaAdapter(nn.Module):
    """
    PixelREPA masked transformer adapter (training-only; 145M parameters). Reads the trunk's
    hidden state after block 5 — the full in-context sequence [image tokens | caption tokens]
    — projects it 2944 -> 2048, replaces a random 20 % of the image tokens with a learned mask
    token, runs two AdaLN DiT blocks (16 heads x 128, FFN 8192, its own timestep pipeline,
    3-D RoPE with captions at identity), and projects the image tokens to DINOv3-L's 1024
    dims through Linear -> LayerNorm -> SiLU -> Linear.
    """

    def __init__(
            self,
            in_dim: int = 2944,
            dim: int = 2048,
            num_heads: int = 16,
            ffn_dim: int = 8192,
            num_blocks: int = 2,
            freq_dim: int = 256,
            eps: float = 1e-6,
            out_dim: int = 1024,
            mask_ratio: float = 0.2,
            rope_max_positions: int = 1024):
        """
        Build the adapter (defaults are the released model's).

        Args:
            in_dim (int):
                Trunk width.
            dim (int):
                Adapter width (head dim 128).
            num_heads (int):
                Attention heads.
            ffn_dim (int):
                SwiGLU target width (hidden 5632).
            num_blocks (int):
                Number of DiT blocks.
            freq_dim (int):
                Sinusoidal timestep width.
            eps (float):
                q/k RMSNorm epsilon.
            out_dim (int):
                Target feature width (1024 for DINOv3-L).
            mask_ratio (float):
                Fraction of image tokens replaced by the mask token.
            rope_max_positions (int):
                RoPE table length.
        """
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.mask_ratio = mask_ratio
        self.input_proj = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embed = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            AdapterBlock(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, out_dim))
        head_dim = dim // num_heads
        with torch.device("cpu"):
            self.freqs = torch.cat([
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=head_dim - 4 * (head_dim // 6)),
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=2 * (head_dim // 6)),
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=2 * (head_dim // 6)),
            ], dim=1)

    def forward(
            self,
            tokens: torch.Tensor,
            mask: torch.Tensor,
            t: torch.Tensor,
            grid: Tuple[int, int, int]) -> torch.Tensor:
        """
        Project masked trunk tokens to DINO space.

        Args:
            tokens (torch.Tensor):
                Trunk hidden state after the tapped block, (B, L_img + L_text, in_dim).
            mask (torch.Tensor):
                The trunk's validity mask for that sequence, (B, L_img + L_text).
            t (torch.Tensor):
                Integer timesteps (B,).
            grid (Tuple[int, int, int]):
                Image token grid (f, h, w); L_img = f * h * w.

        Returns:
            torch.Tensor:
                (B, L_img, out_dim).
        """
        if self.freqs.device != tokens.device:
            self.freqs = self.freqs.to(tokens.device)
        f_p, h_p, w_p = grid
        l_img = f_p * h_p * w_p
        x = self.input_proj(tokens)
        # Per-token Bernoulli keep mask from the global RNG; captions are never masked.
        keep = torch.rand(x.shape[0], l_img, device=x.device) >= self.mask_ratio
        x_img = torch.where(keep.unsqueeze(-1), x[:, :l_img], self.mask_token.to(x.dtype))
        x = torch.cat([x_img, x[:, l_img:]], dim=1)

        e = self.time_embed(sinusoidal_embedding_1d(dim=self.freq_dim, position=t, dtype=x.dtype))
        e0 = self.time_proj(e).unflatten(1, (6, self.dim))
        meta = build_varlen_metadata(mask=mask)
        rope = build_rope_table(
            freqs=self.freqs, f_s=f_p, h_s=h_p, w_s=w_p, text_len=tokens.shape[1] - l_img)
        for block in self.blocks:
            x = block(x=x, e=e0, mask=mask, rope_table=rope, varlen_meta=meta)
        return self.head(x[:, :l_img])


def dino_alignment_target(
        x_0: torch.Tensor,
        dino_features: Callable[[torch.Tensor], torch.Tensor],
        model_patch: int = 32,
        dino_patch: int = 16) -> torch.Tensor:
    """
    DINOv3-L patch features of the CLEAN image on the trunk's token grid, so DINO token
    (i, j) covers the same pixels as trunk token (i, j). The image is bicubic-resized by
    dino_patch / model_patch (512x512 -> 256x256, aspect preserved), clamped to [-1, 1]
    (bicubic overshoots), zero-padded (mid-grey) out to grid * dino_patch exactly as the
    model pads its input, mapped to [0, 1] and ImageNet-normalized; then read at DINOv3's
    final block with its final norm, CLS and register tokens dropped: 256 tokens at 512x512.

    Args:
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W) in [-1, 1].
        dino_features (Callable):
            Frozen DINOv3 ViT-L/16 (timm `vit_large_patch16_dinov3.lvd1689m`, dynamic image
            size): normalized image -> final-block, final-norm patch tokens (B, N, 1024).
        model_patch (int):
            The model's spatial patch.
        dino_patch (int):
            DINOv3's patch.

    Returns:
        torch.Tensor:
            (B, grid_h * grid_w, 1024).
    """
    h_raw, w_raw = x_0.shape[-2:]
    grid_h, grid_w = math.ceil(h_raw / model_patch), math.ceil(w_raw / model_patch)
    canvas_h, canvas_w = grid_h * dino_patch, grid_w * dino_patch
    scale = dino_patch / model_patch
    content_h = min(max(1, round(h_raw * scale)), canvas_h)
    content_w = min(max(1, round(w_raw * scale)), canvas_w)
    with torch.no_grad():
        # torchvision's tensor resize: bicubic, antialiased, align_corners=False, in fp32.
        x = F.interpolate(
            x_0[:, :, 0].float(), size=(content_h, content_w), mode="bicubic",
            align_corners=False, antialias=True).clamp_(-1.0, 1.0)
        x = F.pad(x, (0, canvas_w - content_w, 0, canvas_h - content_h), value=0.0)
        return dino_features(_imagenet_normalize(x=(x + 1.0) * 0.5))


def pixel_repa_loss(z_adapter: torch.Tensor, z_dino: torch.Tensor) -> torch.Tensor:
    """
    Negative cosine similarity between the adapter's projection and the DINO target,
    averaged over ALL image tokens, masked and kept alike.

    Args:
        z_adapter (torch.Tensor):
            PixelRepaAdapter output (B, L_img, 1024).
        z_dino (torch.Tensor):
            dino_alignment_target output (B, L_img, 1024).

    Returns:
        torch.Tensor:
            Scalar in [-1, 1]; lower is better.
    """
    return -(F.normalize(z_dino, dim=-1) * F.normalize(z_adapter, dim=-1)).sum(-1).mean()


# --------------------------------
# PERCEPTUAL LOSSES ON THE FINAL PREDICTION
# --------------------------------

def perceptual_gate(t: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """
    Per-sample gate: perceptual terms apply only when t <= 0.7 (the prediction is a
    plausible image). Terms are averaged over the gated-in samples, not the whole batch.

    Args:
        t (torch.Tensor):
            Timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            Float gate (B,).
    """
    return (t <= cfg.perceptual_gate_t).float()


def gated_mean(per_sample: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Mean over the samples the gate lets through.

    Args:
        per_sample (torch.Tensor):
            Per-sample losses (B,).
        gate (torch.Tensor):
            Float gate (B,).

    Returns:
        torch.Tensor:
            Scalar.
    """
    return (per_sample * gate).sum() / gate.sum().clamp_min(1.0)


def lpips_loss(
        x_pred: torch.Tensor,
        x_0: torch.Tensor,
        lpips_vgg: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        gate: torch.Tensor,
        crop: int = 224) -> torch.Tensor:
    """
    LPIPS (VGG) between the final x_0 prediction and the clean image on one shared random
    224x224 crop per batch. Inputs stay in [-1, 1].

    Args:
        x_pred (torch.Tensor):
            Final prediction (B, 3, 1, H, W).
        x_0 (torch.Tensor):
            Clean image (B, 3, 1, H, W).
        lpips_vgg (Callable):
            Frozen LPIPS network (`lpips.LPIPS(net="vgg")`): (pred, target) 4-D in [-1, 1]
            -> (B, 1, 1, 1).
        gate (torch.Tensor):
            Output of perceptual_gate.
        crop (int):
            Crop size. Default 224.

    Returns:
        torch.Tensor:
            Scalar.
    """
    pred, target = x_pred[:, :, 0], x_0[:, :, 0]
    h, w = pred.shape[-2:]
    if h < crop or w < crop:   # bicubic upsize of the short axis only (never at 512px)
        size = (max(h, crop), max(w, crop))
        pred = F.interpolate(pred, size=size, mode="bicubic", align_corners=False)
        target = F.interpolate(target, size=size, mode="bicubic", align_corners=False)
    top = int(torch.randint(0, pred.shape[-2] - crop + 1, (1,)))
    left = int(torch.randint(0, pred.shape[-1] - crop + 1, (1,)))
    pred = pred[..., top:top + crop, left:left + crop]
    target = target[..., top:top + crop, left:left + crop]
    return gated_mean(per_sample=lpips_vgg(pred, target).view(-1), gate=gate)


def pdino_loss(
        x_pred: torch.Tensor,
        x_0: torch.Tensor,
        dino_features: Callable[[torch.Tensor], torch.Tensor],
        gate: torch.Tensor,
        max_side: int = 224,
        dino_patch: int = 16) -> torch.Tensor:
    """
    Perceptual DINO loss: 1 - cosine similarity between DINOv3-L patch features of the
    prediction (with gradient) and of the clean image (no gradient), averaged over tokens.
    Images are ImageNet-normalized, then bicubic-resized so the longer side is <= 224 with
    each side rounded to DINOv3's 16-pixel patch (512x512 -> 224x224). Same frozen DINOv3-L
    as the alignment target, final block, final norm.

    Args:
        x_pred (torch.Tensor):
            Final prediction (B, 3, 1, H, W).
        x_0 (torch.Tensor):
            Clean image (B, 3, 1, H, W).
        dino_features (Callable):
            See dino_alignment_target.
        gate (torch.Tensor):
            Output of perceptual_gate.
        max_side (int):
            Longest side after resizing.
        dino_patch (int):
            DINOv3's patch.

    Returns:
        torch.Tensor:
            Scalar.
    """
    h, w = x_pred.shape[-2:]
    scale = min(1.0, max_side / max(h, w))
    size = (max(dino_patch, round(h * scale / dino_patch) * dino_patch),
            max(dino_patch, round(w * scale / dino_patch) * dino_patch))

    def prepare(x: torch.Tensor) -> torch.Tensor:
        x = _imagenet_normalize(x=(x[:, :, 0].float() + 1.0) * 0.5)
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)

    with torch.no_grad():
        feat_target = dino_features(prepare(x_0))
    feat_pred = dino_features(prepare(x_pred))
    per_sample = (1 - F.cosine_similarity(feat_pred, feat_target, dim=-1)).mean(dim=-1)
    return gated_mean(per_sample=per_sample, gate=gate)


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    ImageNet mean/std normalization of a [0, 1] image batch.

    Args:
        x (torch.Tensor):
            (B, 3, H, W) in [0, 1].

    Returns:
        torch.Tensor:
            Normalized batch.
    """
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------
# TAPPED FORWARD
# --------------------------------

def forward_with_taps(
        model: PyramidJiT,
        readout_heads: nn.ModuleDict,
        adapter: Optional[PixelRepaAdapter],
        x_t: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        t_int: torch.Tensor,
        cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """
    One training forward: the final prediction plus the readout predictions and the adapter
    projection, read off the trunk with forward hooks so the trunk itself runs exactly as at
    sampling time.

    Args:
        model (PyramidJiT):
            The trunk.
        readout_heads (nn.ModuleDict):
            From build_readout_heads.
        adapter (Optional[PixelRepaAdapter]):
            PixelREPA adapter, or None to skip the alignment tap.
        x_t (torch.Tensor):
            Noisy images (B, 3, 1, H, W).
        text (torch.Tensor):
            Caption embeddings (B, 256, 7680), after drop_text.
        text_lens (torch.Tensor):
            Caption lengths (B,).
        t_int (torch.Tensor):
            Integer timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        Dict[str, torch.Tensor]:
            "x_pred", "block{k}" for each readout, and "z_adapter" when an adapter is given.
    """
    taps: Dict[object, torch.Tensor] = {}
    wanted = {block for block, _, _ in cfg.readouts}
    if adapter is not None:
        wanted.add(cfg.pixel_repa_block)
    hooks = [
        model.blocks[block - 1].register_forward_hook(
            lambda m, i, o, k=block: taps.__setitem__(k, o))
        for block in sorted(wanted)]
    # The final head's timestep embedding, which the readouts share.
    hooks.append(model.head.register_forward_hook(
        lambda m, args, kwargs, o: taps.__setitem__("e", kwargs["e"]), with_kwargs=True))
    try:
        x_pred = model(x_t=x_t, text=text, text_lens=text_lens, t=t_int)
    finally:
        for hook in hooks:
            hook.remove()

    f_in, h_in, w_in = x_t.shape[2:]
    pt, ph, pw = model.patch
    grid = (-(-f_in // pt), -(-h_in // ph), -(-w_in // pw))
    l_img = math.prod(grid)
    out = {"x_pred": x_pred}
    for block, down, _ in cfg.readouts:
        out[f"block{block}"] = apply_readout(
            head=readout_heads[f"block{block}"], tokens=taps[block][:, :l_img], e=taps["e"],
            grid=grid, image_fhw=(f_in, h_in, w_in), model_patch=model.patch,
            downsample=down)
    if adapter is not None:
        img_mask = torch.ones(x_t.shape[0], l_img, dtype=torch.bool, device=x_t.device)
        mask = torch.cat([img_mask, build_text_mask(seq_len=model.text_len, lengths=text_lens)],
                         dim=1)
        out["z_adapter"] = adapter(
            tokens=taps[cfg.pixel_repa_block], mask=mask, t=t_int, grid=grid)
    return out


# --------------------------------
# TOTAL
# --------------------------------

def pjit_loss(
        model: PyramidJiT,
        readout_heads: nn.ModuleDict,
        x_0: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        global_step: int,
        cfg: LossConfig,
        adapter: Optional[PixelRepaAdapter] = None,
        dino_features: Optional[Callable] = None,
        lpips_vgg: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
    """
    One training step's losses, as computed for the released model. Call under
    torch.autocast("cuda", dtype=torch.bfloat16), as training did.

    Args:
        model (PyramidJiT):
            The trunk.
        readout_heads (nn.ModuleDict):
            From build_readout_heads.
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W) in [-1, 1], bf16.
        text (torch.Tensor):
            Caption embeddings (B, 256, 7680).
        text_lens (torch.Tensor):
            Caption lengths (B,).
        global_step (int):
            Optimizer step (selects the timestep phase).
        cfg (LossConfig):
            Objective hyperparameters.
        adapter (Optional[PixelRepaAdapter]):
            PixelREPA adapter (training-only module).
        dino_features (Optional[Callable]):
            Frozen DINOv3-L feature extractor (PixelREPA target and P-DINO).
        lpips_vgg (Optional[Callable]):
            Frozen LPIPS-VGG.

    Returns:
        Dict[str, torch.Tensor]:
            Every weighted term plus "total".
    """
    t = sample_timesteps(
        batch_size=x_0.shape[0], global_step=global_step, cfg=cfg, device=x_0.device,
        dtype=x_0.dtype)
    x_t, t_int = add_noise(x_0=x_0, t=t, cfg=cfg)
    use_repa = adapter is not None and dino_features is not None
    z_dino = dino_alignment_target(x_0=x_0, dino_features=dino_features) if use_repa else None
    out = forward_with_taps(
        model=model, readout_heads=readout_heads, adapter=adapter if use_repa else None,
        x_t=x_t, text=drop_text(text=text, cfg=cfg), text_lens=text_lens, t_int=t_int, cfg=cfg)

    losses = {"final": velocity_mse(x_pred=out["x_pred"], x_0=x_0, t=t, cfg=cfg)}
    for block, _, weight in cfg.readouts:
        pred = out[f"block{block}"]
        losses[f"block{block}"] = weight * velocity_mse(
            x_pred=pred, x_0=readout_target(x_0=x_0, size=tuple(pred.shape[-2:])), t=t, cfg=cfg)
    if use_repa:
        losses["pixel_repa"] = cfg.pixel_repa_weight * pixel_repa_loss(
            z_adapter=out["z_adapter"], z_dino=z_dino)
    gate = perceptual_gate(t=t, cfg=cfg)
    if lpips_vgg is not None:
        losses["lpips"] = cfg.lpips_weight * lpips_loss(
            x_pred=out["x_pred"], x_0=x_0, lpips_vgg=lpips_vgg, gate=gate)
    if dino_features is not None:
        losses["pdino"] = cfg.pdino_weight * pdino_loss(
            x_pred=out["x_pred"], x_0=x_0, dino_features=dino_features, gate=gate)
    losses["total"] = sum(losses.values())
    return losses


def ema_beta(cfg: LossConfig) -> float:
    """
    EMA decay used to produce the released weights: shadow = beta * shadow + (1 - beta) * w,
    every optimizer step from `ema_start_step` on.

    Args:
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        float:
            beta (0.99982260 for a 3907-step half-life).
    """
    return math.exp(-math.log(2.0) / cfg.ema_half_life_steps)
