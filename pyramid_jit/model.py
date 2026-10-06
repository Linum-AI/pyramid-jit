# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: model.py
Description: The P-JiT model — a pixel-space text-to-image diffusion transformer. One
    single-stream ("in-context") DiT reads the noisy image as 32x32-pixel patches (256 tokens
    at 512x512) together with the caption and predicts the clean image directly
    (x-prediction).

    Each stream has a Z-Image-style pre-conditioner before the concat: a 2-block image
    refiner and a 2-block text refiner, both timestep-modulated. Blocks follow the Z-Image
    recipe: RMSNorm before and after each sub-block (sandwich norm), low-rank AdaLN (a shared
    timestep -> 256 down-projection, a per-block up-projection to scale-in / tanh gate-out,
    no shift), RMSNorm on q/k, a per-head sigmoid attention gate, SwiGLU FFN, and 3-D rotary
    position embeddings on the image tokens.

    During training, two extra heads read the trunk out after block 10 (a 128x128 image) and
    block 16 (256x256), and a PixelREPA adapter after block 5 aligned the trunk to DINOv3.
    None of them take part in sampling, so they are not in this module or the released
    weights; loss.py documents them.

    Every module here mirrors the training implementation op-for-op, including the mixed
    dtype flow (fp32 master weights under bf16 autocast, fp32 norms, fp32 RoPE), so the
    released weights reproduce the training-time samples bit-for-bit on the same hardware.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from pyramid_jit.attention import VarlenMeta, build_varlen_metadata, varlen_attention
from pyramid_jit.config import PyramidJiTConfig

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"


# --------------------------------
# P-JIT
# --------------------------------

class PyramidJiT(nn.Module):
    """
    One in-context diffusion transformer with text + image refiner pre-streams.
    """

    def __init__(self, config: PyramidJiTConfig):
        """
        Build the DiT from the architecture config.

        Args:
            config (PyramidJiTConfig):
                Architecture hyperparameters.
        """
        super().__init__()
        dim = config.dim
        self.config = config
        self.patch = tuple(config.patch)

        self.patch_embed = BottleneckPatchEmbed(
            in_channels=config.in_channels, dim=dim, patch=self.patch,
            bottleneck_dim=config.bottleneck_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # Low-rank AdaLN roots: timestep embedding -> 256-wide conditioning vector, one for
        # the image path (image refiner + trunk) and one for the text refiner.
        self.image_adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, config.adaln_rank))
        self.text_adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, config.adaln_rank))
        # bias=False so zero-padded caption rows stay exactly zero.
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim, bias=False))

        block_kwargs = dict(
            dim=dim, num_heads=config.num_heads, eps=config.eps, adaln_rank=config.adaln_rank)
        self.image_refiner = nn.ModuleList([
            Block(ffn_dim=config.refiner_ffn_dim, **block_kwargs)
            for _ in range(config.refiner_blocks)])
        self.text_refiner = nn.ModuleList([
            Block(ffn_dim=config.refiner_ffn_dim, **block_kwargs)
            for _ in range(config.refiner_blocks)])
        self.blocks = nn.ModuleList([
            Block(ffn_dim=config.ffn_dim, **block_kwargs)
            for _ in range(config.num_layers)])
        self.head = OutputHead(dim=dim, out_channels=config.out_channels, patch=self.patch)

        # 3-D RoPE table, complex128, split (T: 22, H: 21, W: 21) pairs of the 128-wide
        # head. A plain attribute (not a buffer) so it is never cast by .to(dtype); computed
        # on CPU in fp64 and moved to the model's device on first use.
        head_dim = config.head_dim
        assert head_dim % 2 == 0
        with torch.device("cpu"):
            self.freqs = torch.cat([
                compute_rotary_frequencies(
                    max_seq_len=config.rope_max_positions,
                    dim=head_dim - 4 * (head_dim // 6)),
                compute_rotary_frequencies(
                    max_seq_len=config.rope_max_positions, dim=2 * (head_dim // 6)),
                compute_rotary_frequencies(
                    max_seq_len=config.rope_max_positions, dim=2 * (head_dim // 6)),
            ], dim=1)

    @property
    def text_len(self) -> int:
        """Padded caption length the model expects."""
        return self.config.text_len

    def forward(
            self,
            x_t: torch.Tensor,
            text: torch.Tensor,
            text_lens: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:
        """
        Predict the clean image for one denoising step.

        Sequence layout in the trunk: [image tokens | caption tokens (padded)]. Inputs whose
        height or width is not a multiple of 32 are zero-padded up to one (as in training,
        e.g. 640x360 -> 640x384) and the prediction is cropped back.

        Args:
            x_t (torch.Tensor):
                Noisy image (B, 3, 1, H, W).
            text (torch.Tensor):
                Caption embeddings zero-padded to (B, text_len, text_dim).
            text_lens (torch.Tensor):
                True caption lengths (B,), int32.
            t (torch.Tensor):
                Integer timesteps (B,), int64, in [0, 1000].

        Returns:
            torch.Tensor:
                The clean-image prediction (B, 3, 1, H, W).
        """
        if self.freqs.device != x_t.device:
            self.freqs = self.freqs.to(x_t.device)
        f_in, h_in, w_in = x_t.shape[2:]
        pads = [(p - size % p) % p for size, p in zip((f_in, h_in, w_in), self.patch)]
        if any(pads):
            # F.pad order: (W_left, W_right, H_top, H_bottom, F_front, F_back)
            x_t = F.pad(x_t, (0, pads[2], 0, pads[1], 0, pads[0]), mode="constant", value=0)

        # Patchify: (B, dim, f_p, h_p, w_p) -> (B, L_img, dim).
        x_emb = self.patch_embed(x_t)
        f_p, h_p, w_p = x_emb.shape[2:]
        x_img = x_emb.flatten(2).transpose(1, 2)
        b, l_img, _ = x_img.shape
        img_mask = torch.ones(b, l_img, dtype=torch.bool, device=x_t.device)
        # A no-op on values (every image token is valid), but it materializes the transposed
        # view contiguously, as training did; the GEMMs below round differently on the
        # strided layout.
        x_img = x_img.masked_fill(~img_mask.unsqueeze(-1), 0.0)

        # Timestep conditioning: e feeds the output head, c_image / c_text the blocks' AdaLN.
        e = self.time_embed(
            sinusoidal_embedding_1d(dim=self.config.freq_dim, position=t, dtype=x_img.dtype))
        c_image = self.image_adaln(e)                                         # [B, rank]
        c_text = self.text_adaln(e)                                           # [B, rank]

        text_emb = self.text_proj(text)                                       # [B, L_t, dim]

        # Image refiner: image tokens only, usual (t, h, w) RoPE.
        img_meta = build_varlen_metadata(mask=img_mask)
        img_rope = build_rope_table(
            freqs=self.freqs, f_s=f_p, h_s=h_p, w_s=w_p, text_len=0)
        for block in self.image_refiner:
            x_img = block(
                x=x_img, c=c_image, mask=img_mask, rope_table=img_rope, varlen_meta=img_meta)

        # Text refiner: caption tokens only, all at the identity rotation.
        text_mask = build_text_mask(seq_len=self.text_len, lengths=text_lens)
        text_meta = build_varlen_metadata(mask=text_mask)
        text_rope = build_rope_table(
            freqs=self.freqs, f_s=0, h_s=0, w_s=0, text_len=self.text_len)
        for block in self.text_refiner:
            text_emb = block(
                x=text_emb, c=c_text, mask=text_mask, rope_table=text_rope,
                varlen_meta=text_meta)

        # Trunk sequence: [image | text], padded rows zeroed.
        seq = torch.cat([x_img, text_emb], dim=1)
        mask = torch.cat([img_mask, text_mask], dim=1)
        seq = seq.masked_fill(~mask.unsqueeze(-1), 0.0)
        meta = build_varlen_metadata(mask=mask)
        rope = build_rope_table(
            freqs=self.freqs, f_s=f_p, h_s=h_p, w_s=w_p, text_len=self.text_len)
        for block in self.blocks:
            seq = block(x=seq, c=c_image, mask=mask, rope_table=rope, varlen_meta=meta)

        out = self.head(x=seq[:, :l_img, :], e=e)
        out = unpatchify(
            x_head=out, f_p=f_p, h_p=h_p, w_p=w_p, patch=self.patch,
            out_channels=self.config.out_channels)
        return out[:, :, :f_in, :h_in, :w_in]

    @classmethod
    def from_pretrained(
            cls,
            weights_dir: str,
            device: str = "cuda") -> "PyramidJiT":
        """
        Load the released weights (`config.json` + `model.safetensors`).

        Args:
            weights_dir (str):
                A local directory holding the two files, or a Hugging Face Hub model id
                (downloaded with `huggingface_hub.snapshot_download`; set `HF_TOKEN` for a
                private repo).
            device (str):
                Device to load onto. Default "cuda".

        Returns:
            PyramidJiT:
                The model on `device`, in eval mode, fp32 parameters.
        """
        from safetensors.torch import load_file

        if not os.path.isdir(weights_dir):
            from huggingface_hub import snapshot_download
            weights_dir = snapshot_download(
                repo_id=weights_dir, allow_patterns=[CONFIG_FILENAME, WEIGHTS_FILENAME])
        config = PyramidJiTConfig.from_json(path=os.path.join(weights_dir, CONFIG_FILENAME))
        with torch.device("meta"):
            model = cls(config=config)
        state = load_file(os.path.join(weights_dir, WEIGHTS_FILENAME), device=device)
        model.load_state_dict(state, strict=True, assign=True)
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                raise ValueError(
                    f"{name} is {param.dtype}; the released weights must stay fp32 (they "
                    "are used as fp32 masters under bf16 autocast).")
        return model.to(device).eval()


# --------------------------------
# BLOCKS
# --------------------------------

class Block(nn.Module):
    """
    DiT block with low-rank AdaLN and sandwich norm:

        x = x + tanh(g1) * PostNorm1(Attn(Norm1(x) * (1 + s1)))
        x = x + tanh(g2) * PostNorm2(FFN(Norm2(x) * (1 + s2)))

    where (s1, g1, s2, g2) = adaln_up(c) and c is the shared low-rank timestep vector.
    """

    def __init__(
            self,
            dim: int,
            ffn_dim: int,
            num_heads: int,
            eps: float,
            adaln_rank: int):
        """
        Build the block.

        Args:
            dim (int):
                Hidden width.
            ffn_dim (int):
                SwiGLU target width (see SwiGLUFFN).
            num_heads (int):
                Attention heads.
            eps (float):
                RMSNorm epsilon.
            adaln_rank (int):
                Width of the shared low-rank conditioning vector.
        """
        super().__init__()
        self.norm1 = RMSNorm(dim=dim, eps=eps)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, eps=eps)
        self.post_norm1 = RMSNorm(dim=dim, eps=eps)
        self.norm2 = RMSNorm(dim=dim, eps=eps)
        self.ffn = SwiGLUFFN(dim=dim, hidden_dim=ffn_dim)
        self.post_norm2 = RMSNorm(dim=dim, eps=eps)
        self.adaln_up = nn.Linear(adaln_rank, 4 * dim)

    def forward(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            mask: torch.Tensor,
            rope_table: torch.Tensor,
            varlen_meta: VarlenMeta) -> torch.Tensor:
        """
        Apply the block.

        Args:
            x (torch.Tensor):
                Tokens (B, L, dim).
            c (torch.Tensor):
                Low-rank timestep conditioning (B, rank).
            mask (torch.Tensor):
                Validity mask (B, L).
            rope_table (torch.Tensor):
                Assembled complex rotation table (L, head_dim/2).
            varlen_meta (VarlenMeta):
                Packing metadata for `mask`.

        Returns:
            torch.Tensor:
                Updated tokens (B, L, dim).
        """
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_up(c).unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        y = self.attn(
            x=self.norm1(x) * scale_msa, mask=mask, rope_table=rope_table,
            varlen_meta=varlen_meta)
        y = self.post_norm1(y) * mask.unsqueeze(-1)
        x = x + gate_msa * y

        y = self.ffn(self.norm2(x) * scale_mlp)
        y = self.post_norm2(y)
        y = y * mask.unsqueeze(-1)
        x = x + gate_mlp * y
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with RMSNorm on q/k, 3-D RoPE, and a per-head sigmoid output gate.
    """

    def __init__(self, dim: int, num_heads: int, eps: float):
        """
        Build the attention module.

        Args:
            dim (int):
                Hidden width.
            num_heads (int):
                Attention heads.
            eps (float):
                Epsilon for the q/k RMSNorm.
        """
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim=dim, eps=eps)
        self.norm_k = RMSNorm(dim=dim, eps=eps)
        self.gate_proj = nn.Linear(dim, num_heads)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            rope_table: torch.Tensor,
            varlen_meta: VarlenMeta) -> torch.Tensor:
        """
        Apply attention.

        Args:
            x (torch.Tensor):
                Pre-normalized (and scaled) tokens (B, L, dim).
            mask (torch.Tensor):
                Validity mask (B, L).
            rope_table (torch.Tensor):
                Complex rotation table (L, head_dim/2).
            varlen_meta (VarlenMeta):
                Packing metadata for `mask`.

        Returns:
            torch.Tensor:
                Attention output (B, L, dim), zeros at padded positions.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        not_valid = ~mask.unsqueeze(-1)

        # Mask after the projections so the biases do not leak into padded rows.
        q = self.norm_q(self.q(x).masked_fill(not_valid, 0.0)).view(b, s, n, d)
        k = self.norm_k(self.k(x).masked_fill(not_valid, 0.0)).view(b, s, n, d)
        v = self.v(x).masked_fill(not_valid, 0.0).view(b, s, n, d)
        q = apply_rope(x=q, rope_table=rope_table)
        k = apply_rope(x=k, rope_table=rope_table)

        attn = varlen_attention(q=q, k=k, v=v, mask=mask, varlen_meta=varlen_meta)

        gate_logits = self.gate_proj(x).masked_fill(not_valid, -1e4)          # [B, L, n]
        attn = attn * torch.sigmoid(gate_logits).unsqueeze(-1)
        attn = self.o(attn.flatten(2))
        return attn.masked_fill(not_valid, 0.0)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward: w2(silu(w1 x) * w3 x), with w1/w3 packed into one Linear (w13).
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        """
        Build the FFN.

        Args:
            dim (int):
                Input/output width.
            hidden_dim (int):
                Target width; the actual hidden width is 2/3 of it rounded up to `multiple_of`
                (7936 -> 5376, 4096 -> 2816).
            multiple_of (int):
                Rounding granularity. Default 256.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the FFN.

        Args:
            x (torch.Tensor):
                Input (..., dim).

        Returns:
            torch.Tensor:
                Output (..., dim).
        """
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x3)


class RMSNorm(nn.Module):
    """
    RMS normalization. Normalizes in fp32, rounds to the input dtype, then applies the fp32
    gain — so under bf16 autocast the output is promoted to fp32 (the training behaviour).
    """

    def __init__(self, dim: int, eps: float):
        """
        Build the norm.

        Args:
            dim (int):
                Normalized width.
            eps (float):
                Epsilon inside the rsqrt.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the last dimension.

        Args:
            x (torch.Tensor):
                Input (..., dim).

        Returns:
            torch.Tensor:
                Normalized, gained output in the promoted dtype.
        """
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


# --------------------------------
# INPUT / OUTPUT LAYERS
# --------------------------------

class BottleneckPatchEmbed(nn.Module):
    """
    Two-stage patch embedding: a patchifying Conv3d into `bottleneck_dim` channels, then a 1x1
    Conv3d up to the model width (after JiT's BottleneckPatchEmbed, arXiv:2511.13720).
    """

    def __init__(
            self,
            in_channels: int,
            dim: int,
            patch: Tuple[int, int, int],
            bottleneck_dim: int):
        """
        Build the embedding.

        Args:
            in_channels (int):
                Image channels.
            dim (int):
                Model width.
            patch (Tuple[int, int, int]):
                Patch size (t, h, w).
            bottleneck_dim (int):
                Intermediate channels.
        """
        super().__init__()
        self.proj1 = nn.Conv3d(
            in_channels, bottleneck_dim, kernel_size=patch, stride=patch, bias=False)
        self.proj2 = nn.Conv3d(bottleneck_dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patchify.

        Args:
            x (torch.Tensor):
                Image (B, C, F, H, W).

        Returns:
            torch.Tensor:
                Patch tokens (B, dim, F', H', W').
        """
        return self.proj2(self.proj1(x))


class OutputHead(nn.Module):
    """
    Final AdaLN-modulated LayerNorm + linear projection to prod(patch) * out_channels per token.
    """

    def __init__(
            self,
            dim: int,
            out_channels: int,
            patch: Tuple[int, int, int]):
        """
        Build the head.

        Args:
            dim (int):
                Model width.
            out_channels (int):
                Output image channels.
            patch (Tuple[int, int, int]):
                Output patch size.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, math.prod(patch) * out_channels)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Project tokens to pixels.

        Args:
            x (torch.Tensor):
                Final-block image tokens (B, L, dim).
            e (torch.Tensor):
                Timestep embedding (B, dim).

        Returns:
            torch.Tensor:
                (B, L, prod(patch) * out_channels).
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)               # 2 x [B, 1, dim]
        return self.proj(self.norm(x) * (1 + e[1]) + e[0])


def unpatchify(
        x_head: torch.Tensor,
        f_p: int,
        h_p: int,
        w_p: int,
        patch: Tuple[int, int, int],
        out_channels: int) -> torch.Tensor:
    """
    Reassemble per-token pixel patches into an image.

    Args:
        x_head (torch.Tensor):
            Head output (B, f_p * h_p * w_p, prod(patch) * out_channels).
        f_p (int):
            Patch grid depth.
        h_p (int):
            Patch grid height.
        w_p (int):
            Patch grid width.
        patch (Tuple[int, int, int]):
            Patch size (t, h, w).
        out_channels (int):
            Image channels.

    Returns:
        torch.Tensor:
            Image (B, out_channels, f_p * t, h_p * h, w_p * w).
    """
    b = x_head.shape[0]
    pt, ph, pw = patch
    assert x_head.shape[1] == f_p * h_p * w_p
    u = x_head.view(b, f_p, h_p, w_p, pt, ph, pw, out_channels)
    u = torch.einsum("bfhwpqrc->bcfphqwr", u)
    return u.reshape(b, out_channels, f_p * pt, h_p * ph, w_p * pw)


# --------------------------------
# EMBEDDINGS, MASKS, ROPE
# --------------------------------

def build_text_mask(seq_len: int, lengths: torch.Tensor) -> torch.Tensor:
    """
    Prefix validity mask from per-sample lengths.

    Args:
        seq_len (int):
            Padded length.
        lengths (torch.Tensor):
            True lengths (B,).

    Returns:
        torch.Tensor:
            Boolean mask (B, seq_len).
    """
    positions = torch.arange(seq_len, device=lengths.device)
    return positions[None, :] < lengths[:, None]


def sinusoidal_embedding_1d(
        dim: int,
        position: torch.Tensor,
        dtype: torch.dtype) -> torch.Tensor:
    """
    Sinusoidal timestep embedding, computed in fp64.

    Args:
        dim (int):
            Embedding width (even).
        position (torch.Tensor):
            Timesteps (B,).
        dtype (torch.dtype):
            Output dtype.

    Returns:
        torch.Tensor:
            [cos | sin] embedding (B, dim).
    """
    assert dim % 2 == 0
    half = dim // 2
    positions_float = position.to(torch.float64)
    inv_freq = torch.pow(10000.0, -torch.arange(half).to(positions_float) / half)
    angle_rads = torch.outer(positions_float, inv_freq)
    pos_encoding = torch.cat([torch.cos(angle_rads), torch.sin(angle_rads)], dim=1)
    return pos_encoding.to(dtype)


@amp.autocast(enabled=False, device_type="cuda")
def compute_rotary_frequencies(
        max_seq_len: int,
        dim: int,
        theta: float = 10000) -> torch.Tensor:
    """
    Complex rotary frequencies exp(i * p * theta^(-2k/dim)) for positions p < max_seq_len.

    Args:
        max_seq_len (int):
            Number of positions.
        dim (int):
            Real width of this axis (even); dim/2 complex pairs.
        theta (float):
            Frequency base. Default 10000.

    Returns:
        torch.Tensor:
            complex128 (max_seq_len, dim/2).
    """
    assert dim % 2 == 0
    seq_positions = torch.arange(max_seq_len, dtype=torch.float64)
    dim_indices = torch.arange(0, dim, 2, dtype=torch.float64)
    inv_freq_base = 1.0 / torch.pow(theta, dim_indices / dim)
    rotation_angles = torch.outer(seq_positions, inv_freq_base)
    return torch.polar(torch.ones_like(rotation_angles), rotation_angles)


def build_rope_table(
        freqs: torch.Tensor,
        f_s: int,
        h_s: int,
        w_s: int,
        text_len: int) -> torch.Tensor:
    """
    Assemble the per-token complex rotation table for one sequence layout
    [image f_s*h_s*w_s | text text_len]. Image tokens rotate by their (t, h, w) grid
    position; caption tokens sit at identity (position 0 on every axis).

    Args:
        freqs (torch.Tensor):
            Per-axis tables concatenated along dim 1, (max_positions, head_dim/2) complex.
        f_s (int):
            Image grid depth.
        h_s (int):
            Image grid height.
        w_s (int):
            Image grid width.
        text_len (int):
            Number of caption positions (all at identity).

    Returns:
        torch.Tensor:
            (L, head_dim/2) complex table in sequence order.
    """
    head_dim_half = freqs.shape[1]
    temporal_dim = head_dim_half - 2 * (head_dim_half // 3)
    spatial_dim = head_dim_half // 3
    freq_t, freq_h, freq_w = freqs.split([temporal_dim, spatial_dim, spatial_dim], dim=1)
    if max(f_s, h_s, w_s) > freqs.shape[0]:
        raise ValueError(
            f"RoPE grid ({f_s}, {h_s}, {w_s}) exceeds the {freqs.shape[0]}-position table.")

    image_seq_len = f_s * h_s * w_s
    if image_seq_len > 0:
        grid_t = freq_t.narrow(0, 0, f_s).view(f_s, 1, 1, -1).expand(f_s, h_s, w_s, -1)
        grid_h = freq_h.narrow(0, 0, h_s).view(1, h_s, 1, -1).expand(f_s, h_s, w_s, -1)
        grid_w = freq_w.narrow(0, 0, w_s).view(1, 1, w_s, -1).expand(f_s, h_s, w_s, -1)
        freqs_image = torch.cat([grid_t, grid_h, grid_w], dim=-1).reshape(
            image_seq_len, head_dim_half)
    else:
        freqs_image = freqs.new_empty((0, head_dim_half))

    frags = [freqs_image]
    if text_len > 0:
        freq_zero = torch.cat([freq_t[0:1], freq_h[0:1], freq_w[0:1]], dim=-1)   # [1, D]
        frags.append(freq_zero.expand(text_len, head_dim_half))
    return torch.cat(frags, dim=0)


@amp.autocast(enabled=False, device_type="cuda")
def apply_rope(x: torch.Tensor, rope_table: torch.Tensor) -> torch.Tensor:
    """
    Rotate q/k by the assembled table: the complex multiply (a + bi)(cos + i sin) written out
    in fp32 over the interleaved channel pairs.

    Args:
        x (torch.Tensor):
            (B, L, num_heads, head_dim).
        rope_table (torch.Tensor):
            (L, head_dim/2) complex table.

    Returns:
        torch.Tensor:
            Rotated tensor in x's dtype.
    """
    b, seq_len, n, head_dim = x.shape
    half = head_dim // 2
    cos = rope_table.real.to(torch.float32).view(1, seq_len, 1, half)
    sin = rope_table.imag.to(torch.float32).view(1, seq_len, 1, half)
    pairs = x.to(torch.float32).reshape(b, seq_len, n, half, 2)
    x_even, x_odd = pairs[..., 0], pairs[..., 1]
    out = torch.stack(
        [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1).flatten(3)
    return out.to(x.dtype)
