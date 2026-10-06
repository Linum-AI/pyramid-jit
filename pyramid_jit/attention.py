# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: attention.py
Description: Variable-length self-attention for P-JiT with two backends:

      * "flash3" — the FlashAttention-3 varlen kernel (Hopper GPUs; `flash_attn_interface`
        built from github.com/Dao-AILab/flash-attention). Padded positions (caption rows past
        the caption length) are packed out with integer indexing, attended, and scattered
        back as zeros. This is what the model was trained and validated with.
      * "sdpa" — PyTorch's scaled_dot_product_attention with an explicit padding mask. Works
        on any GPU with no extra install; images differ from the flash3 ones only by kernel
        rounding.

    The default "auto" uses flash3 when importable, else sdpa.
"""

import warnings
from typing import Tuple

import torch
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
    FLASH_ATTN3_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the machine
    _flash_attn_varlen_func = None
    FLASH_ATTN3_AVAILABLE = False

VarlenMeta = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BACKENDS = ("auto", "flash3", "sdpa")
_BACKEND = "auto"
_WARNED_SDPA = False


# --------------------------------
# BACKEND SELECTION
# --------------------------------

def set_attention_backend(backend: str) -> None:
    """
    Choose the attention backend for every subsequent forward.

    Args:
        backend (str):
            "auto" (flash3 if importable, else sdpa), "flash3" (error if not installed), or
            "sdpa".
    """
    global _BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"attention backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "flash3" and not FLASH_ATTN3_AVAILABLE:
        raise RuntimeError(
            "attention backend 'flash3' requested but flash_attn_interface is not installed "
            "(build it from github.com/Dao-AILab/flash-attention, hopper/).")
    _BACKEND = backend


def active_attention_backend() -> str:
    """
    The backend that will actually run.

    Returns:
        str:
            "flash3" or "sdpa".
    """
    if _BACKEND == "auto":
        return "flash3" if FLASH_ATTN3_AVAILABLE else "sdpa"
    return _BACKEND


# --------------------------------
# PACKING METADATA
# --------------------------------

def build_varlen_metadata(mask: torch.Tensor) -> VarlenMeta:
    """
    Precompute the varlen packing metadata for a boolean [B, L] validity mask.

    Computed once per distinct mask per model forward and shared by every attention layer
    that uses that mask.

    Args:
        mask (torch.Tensor):
            Boolean validity mask of shape [B, L], True = valid. Must be a prefix mask per row.

    Returns:
        VarlenMeta:
            (flat_idx, cu_seqlens, max_seqlen_cpu): int64 [total_valid] positions in the
            flattened [B*L] sequence; int32 [B+1] cumulative lengths on device; 0-dim CPU
            int32 tensor holding the longest per-sample length.
    """
    seq_lens = mask.sum(dim=1).to(torch.int32)
    cu_seqlens = torch.cat(
        [seq_lens.new_zeros([1]), seq_lens]).cumsum(0, dtype=torch.int32)
    flat_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    max_seqlen_cpu = seq_lens.max().cpu()
    return (flat_idx, cu_seqlens, max_seqlen_cpu)


# --------------------------------
# ATTENTION
# --------------------------------

def varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        varlen_meta: VarlenMeta) -> torch.Tensor:
    """
    Self-attention over the valid positions of each sample.

    Args:
        q (torch.Tensor):
            Queries [B, L, num_heads, head_dim] (any float dtype; attended in bf16).
        k (torch.Tensor):
            Keys [B, L, num_heads, head_dim].
        v (torch.Tensor):
            Values [B, L, num_heads, head_dim].
        mask (torch.Tensor):
            Boolean validity mask [B, L] shared by queries and keys.
        varlen_meta (VarlenMeta):
            Output of build_varlen_metadata(mask).

    Returns:
        torch.Tensor:
            Attention output [B, L, num_heads, head_dim] in q's dtype, zeros at padding.
    """
    if active_attention_backend() == "sdpa" or not q.is_cuda:
        # FlashAttention-3 is CUDA-only; CPU/meta tensors always take the SDPA path.
        return _sdpa_attention(q=q, k=k, v=v, mask=mask)

    out_dtype = q.dtype
    flat_idx, cu_seqlens, max_seqlen_cpu = varlen_meta
    b, seq_len, num_heads, head_dim = q.shape

    def pack(t: torch.Tensor) -> torch.Tensor:
        packed = t.reshape(b * seq_len, num_heads, head_dim).index_select(0, flat_idx)
        return packed if packed.dtype == torch.bfloat16 else packed.to(torch.bfloat16)

    x_packed = _flash_attn_varlen_func(
        q=pack(q),
        k=pack(k),
        v=pack(v),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(max_seqlen_cpu),
        max_seqlen_k=int(max_seqlen_cpu))
    if isinstance(x_packed, tuple):   # some FA3 builds return (out, lse)
        x_packed = x_packed[0]

    flat = torch.zeros(
        (b * seq_len, num_heads, head_dim), dtype=x_packed.dtype, device=x_packed.device)
    flat = flat.index_copy(0, flat_idx, x_packed)
    return flat.view(b, seq_len, num_heads, head_dim).type(out_dtype)


def _sdpa_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    """
    PyTorch SDPA with an explicit padding mask. Not bit-identical to FlashAttention-3.

    Args:
        q (torch.Tensor):
            Queries [B, L, num_heads, head_dim].
        k (torch.Tensor):
            Keys [B, L, num_heads, head_dim].
        v (torch.Tensor):
            Values [B, L, num_heads, head_dim].
        mask (torch.Tensor):
            Boolean validity mask [B, L].

    Returns:
        torch.Tensor:
            Attention output [B, L, num_heads, head_dim] in q's dtype, zeros at padding.
    """
    global _WARNED_SDPA
    if not _WARNED_SDPA and _BACKEND == "auto":
        warnings.warn(
            "FlashAttention-3 (flash_attn_interface) is not installed; using PyTorch SDPA. "
            "Images will differ from FlashAttention-3 samples by kernel rounding only.")
        _WARNED_SDPA = True

    out_dtype = q.dtype
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    attn_mask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(1)          # [B, 1, L, L]
    attn_mask = torch.where(attn_mask, 0.0, float("-inf")).to(q.dtype)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = x.transpose(1, 2) * mask[:, :, None, None]
    return x.type(out_dtype)
