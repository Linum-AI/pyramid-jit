# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: sampler.py
Description: Text-to-image sampling for P-JiT: a plain Euler ODE solver over uniform
    timesteps from t=1 (noise) to t=0 (clean), with adaptive projected guidance (APG) applied
    in velocity space.

    The model predicts the clean image x_0 directly. With x_t = (1 - t) x_0 + t x_1, the ODE
    velocity is v = dx_t/dt = x_1 - x_0 = (x_t - x_0) / t, which is what each Euler step uses.
"""

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from tqdm import tqdm

from pyramid_jit.config import SamplerConfig
from pyramid_jit.noise import StackedRandomGenerator


# --------------------------------
# GENERATION
# --------------------------------

@torch.no_grad()
def generate(
        model: torch.nn.Module,
        text_encoder,
        prompt: str,
        seeds: List[int],
        sampler: SamplerConfig,
        negative_prompt: Optional[str] = None,
        device: str = "cuda",
        quiet: bool = False) -> List[torch.Tensor]:
    """
    Sample one image per seed for a prompt.

    Args:
        model (torch.nn.Module):
            A PyramidJiT in eval mode on `device`.
        text_encoder:
            A QwenTextEncoder; called as text_encoder([prompt]) -> [(L, text_dim)].
        prompt (str):
            Caption.
        seeds (List[int]):
            One image per seed; the batch size is len(seeds). Sample k depends only on
            seeds[k] in the sense that its noise does, but GEMM/attention tiling depends on
            the batch size, so bit-exact reproduction also needs the same batch.
        sampler (SamplerConfig):
            Sampling settings.
        negative_prompt (Optional[str]):
            Caption for the unconditional branch; default sampler.negative_prompt.
        device (str):
            CUDA device string.
        quiet (bool):
            Suppress the progress bar.

    Returns:
        List[torch.Tensor]:
            One (3, 1, H, W) uint8 tensor per seed.
    """
    if negative_prompt is None:
        negative_prompt = sampler.negative_prompt
    batch_size = len(seeds)
    text_len = model.text_len

    text_cond = text_encoder([prompt])[0]                                     # (L_c, D)
    text_uncond = text_encoder([negative_prompt])[0]                          # (L_u, D)
    cond = _text_kwargs(text=text_cond, batch_size=batch_size, text_len=text_len)
    uncond = _text_kwargs(text=text_uncond, batch_size=batch_size, text_len=text_len)

    # Initial noise, drawn per seed in bf16 on the device so sample k's noise depends on
    # seeds[k] alone, and by default exactly as an H100 SXM draws it (see noise.py).
    generator = StackedRandomGenerator(
        device=device, seeds=seeds, sm_count=sampler.noise_sm_count)
    x_t = sampler.noise_scale * generator.randn(
        (batch_size, 3, 1, sampler.height, sampler.width),
        dtype=torch.bfloat16, device=device)

    timesteps = torch.linspace(1.0, 0.0, sampler.sampling_steps + 1, device=device)

    with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        momentum_buffer = MomentumBuffer(momentum=sampler.apg_momentum)
        for i in tqdm(range(sampler.sampling_steps), disable=quiet):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_int = (t * sampler.num_timesteps).long()
            t_batch = t_int.repeat(batch_size).to(device)

            x_pred_cond = model(x_t=x_t, t=t_batch, **cond)
            x_pred_uncond = model(x_t=x_t, t=t_batch, **uncond)
            v_cond = (x_t - x_pred_cond) / t
            v_uncond = (x_t - x_pred_uncond) / t

            v_guided = adaptive_projected_guidance(
                pred_cond=v_cond,
                pred_other=v_uncond,
                guidance_scale=sampler.guidance_scale,
                momentum_buffer=momentum_buffer,
                eta=sampler.apg_eta,
                rescale=sampler.apg_rescale)

            x_t = x_t + (t_next - t) * v_guided

        samples = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return [s for s in samples]


def _text_kwargs(text: torch.Tensor, batch_size: int, text_len: int) -> dict:
    """
    Pad one caption embedding to the model's text length and replicate it over the batch.

    Args:
        text (torch.Tensor):
            Caption embedding (L, D), L <= text_len.
        batch_size (int):
            Batch size.
        text_len (int):
            Padded length.

    Returns:
        dict:
            {"text": (B, text_len, D), "text_lens": (B,) int32}.
    """
    if text.shape[0] > text_len:
        raise ValueError(f"Caption has {text.shape[0]} tokens > text_len={text_len}.")
    padded = torch.cat([text, text.new_zeros(text_len - text.shape[0], text.shape[1])], dim=0)
    stacked = torch.stack([padded] * batch_size, dim=0).contiguous()
    lens = torch.full((batch_size,), text.shape[0], dtype=torch.int32, device=text.device)
    return {"text": stacked, "text_lens": lens}


# --------------------------------
# ADAPTIVE PROJECTED GUIDANCE
# --------------------------------
# Sadat et al., "Eliminating Oversaturation and Artifacts of High Guidance Scales in
# Diffusion Models", ICLR 2025. https://arxiv.org/abs/2410.02416

class MomentumBuffer:
    """Running average of the guidance difference across sampling steps."""

    def __init__(self, momentum: float):
        """
        Create the buffer.

        Args:
            momentum (float):
                Momentum coefficient (negative values damp oscillation).
        """
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor) -> None:
        """
        Fold a new difference into the running average.

        Args:
            update_value (torch.Tensor):
                The new guidance difference.
        """
        self.running_average = update_value + self.momentum * self.running_average


def project(v0: torch.Tensor, v1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split v0 into its components parallel and orthogonal to v1 (per sample, in float64).

    Args:
        v0 (torch.Tensor):
            Vector to split (B, ...).
        v1 (torch.Tensor):
            Reference direction (B, ...).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (parallel, orthogonal) in v0's dtype.
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    dims = list(range(1, v1.ndim))
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    return v0_parallel.to(dtype), (v0 - v0_parallel).to(dtype)


def adaptive_projected_guidance(
        pred_cond: torch.Tensor,
        pred_other: torch.Tensor,
        guidance_scale: float,
        momentum_buffer: MomentumBuffer,
        eta: float,
        rescale: float) -> torch.Tensor:
    """
    APG: momentum on the cond - uncond difference, clip its per-frame L2 norm to `rescale`,
    drop (scale by `eta`) the component parallel to the conditional prediction, then apply
    the guidance scale to what remains.

    Args:
        pred_cond (torch.Tensor):
            Conditional velocity (B, C, T, H, W).
        pred_other (torch.Tensor):
            Unconditional velocity (B, C, T, H, W).
        guidance_scale (float):
            Guidance scale.
        momentum_buffer (MomentumBuffer):
            Momentum state carried across steps.
        eta (float):
            Weight of the parallel component.
        rescale (float):
            Maximum per-frame L2 norm of the difference (<= 0 disables).

    Returns:
        torch.Tensor:
            Guided velocity.
    """
    diff = pred_cond - pred_other
    momentum_buffer.update(update_value=diff)
    diff = momentum_buffer.running_average

    if rescale > 0:
        diff_norm = diff.norm(p=2, dim=[1, 3, 4], keepdim=True)                # (B, 1, T, 1, 1)
        diff = diff * torch.minimum(torch.ones_like(diff), rescale / diff_norm)

    diff_parallel, diff_orthogonal = project(v0=diff, v1=pred_cond)
    return pred_cond + guidance_scale * (diff_orthogonal + eta * diff_parallel)
