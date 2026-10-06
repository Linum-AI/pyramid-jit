# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: noise.py
Description: Seeded initial noise that is the same on every NVIDIA GPU.

    `torch.randn(..., generator=g)` on CUDA does not give the same tensor on every GPU model.
    PyTorch's normal kernel (`distribution_elementwise_grid_stride_kernel` in ATen's
    DistributionTemplates.h) launches `min(ceil(numel / 256), 8 * SM_count)` blocks of 256
    threads. Thread i calls `curand_init(seed, subsequence=i, offset)` and writes the 4
    normals of each `curand_normal4` call to elements `i + T*k` (T = total threads). Once the
    tensor is large enough for the SM cap to bind, which it is for one 512x512 image, T
    depends on the SM count, so the same seed lands its normals on different pixels: an
    H100 PCIe (114 SMs), an H100 SXM (132) and a B200 (148) draw three unrelated noise fields
    and the sampler turns them into three different images.

    P-JiT was trained and validated on H100 SXM. `randn_as_on_sm_count` reproduces, bit for
    bit, what torch.randn draws on a GPU with a given SM count, on any NVIDIA GPU. It is a
    Triton port of that kernel: Triton's Philox4x32-10 (same round constants and key schedule
    as curand's) at counter (offset/4 + iteration, 0, i_lo, i_hi), then curand's Box-Muller
    exactly as nvcc compiles it for the device (fma-contracted uniform transform, accurate
    logf, sqrt.rn, and the sin.approx / cos.approx intrinsics behind __sincosf). Triton ships
    with PyTorch, so no CUDA toolkit is needed.

    Checked against native torch.randn on a 114-SM H100 PCIe (fp32 / bf16 / fp16, single- and
    multi-iteration shapes, 64-bit seeds, consecutive draws from one generator: all
    torch.equal, generator offsets equal) and against SHA256s of noise drawn on a real H100
    SXM.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

_BLOCK = 256          # threads per block in torch's RNG launch
_UNROLL = 4           # normals per curand_normal4 call
# curand_globals.h constants, folded in fp32 as nvcc folds them.
_INV_2POW32 = float(np.float32(2.3283064e-10))
_INV_2POW32_2PI = float(np.float32(np.float32(2.3283064e-10) * np.float32(6.2831855)))
_INV_2POW32_HALF = float(np.float32(_INV_2POW32) / np.float32(2))
_INV_2POW32_2PI_HALF = float(np.float32(_INV_2POW32_2PI) / np.float32(2))

_KERNEL = None


def _build_kernel():
    """Compile the Triton kernel on first use (keeps `import pyramid_jit` Triton-free)."""
    import triton
    import triton.language as tl

    @triton.jit
    def box_muller(x, y, inv, inv_half, inv_2pi, inv_2pi_half):
        # curand_normal.h _curand_box_muller, device branch.
        u = tl.fma(x.to(tl.float32), inv, inv_half)
        v = tl.fma(y.to(tl.float32), inv_2pi, inv_2pi_half)
        s = tl.sqrt_rn(-2.0 * tl.extra.cuda.libdevice.log(u))
        sin_v = tl.inline_asm_elementwise(
            "sin.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        cos_v = tl.inline_asm_elementwise(
            "cos.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        return sin_v * s, cos_v * s

    @triton.jit
    def randn_kernel(out_ptr, seed, counter0, threads, numel, iterations,
                     inv, inv_half, inv_2pi, inv_2pi_half,
                     BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = idx < threads
        zero = tl.zeros([BLOCK], dtype=tl.uint32)
        subsequence_lo = (idx & 0xFFFFFFFF).to(tl.uint32)
        subsequence_hi = (idx >> 32).to(tl.uint32)
        for it in range(iterations):
            c0 = zero + (counter0 + it).to(tl.uint32)
            r0, r1, r2, r3 = tl.philox(seed, c0, zero, subsequence_lo, subsequence_hi)
            n0, n1 = box_muller(r0, r1, inv, inv_half, inv_2pi, inv_2pi_half)
            n2, n3 = box_muller(r2, r3, inv, inv_half, inv_2pi, inv_2pi_half)
            base = idx + threads * UNROLL * it
            tl.store(out_ptr + base, n0, mask=live & (base < numel))
            tl.store(out_ptr + base + threads, n1, mask=live & (base + threads < numel))
            tl.store(out_ptr + base + 2 * threads, n2, mask=live & (base + 2 * threads < numel))
            tl.store(out_ptr + base + 3 * threads, n3, mask=live & (base + 3 * threads < numel))

    return randn_kernel


def randn_as_on_sm_count(
        size: Sequence[int],
        generator: torch.Generator,
        sm_count: int,
        max_threads_per_sm: int = 2048,
        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    `torch.randn(size, generator=generator, dtype=dtype)` as a CUDA GPU with `sm_count` SMs
    would draw it, on whatever CUDA GPU `generator` lives on.

    Args:
        size (Sequence[int]):
            Output shape.
        generator (torch.Generator):
            A CUDA generator; advanced exactly as the native draw would advance it.
        sm_count (int):
            SM count of the GPU to reproduce (H100 SXM: 132, H100 PCIe: 114, B200: 148).
        max_threads_per_sm (int):
            Resident threads per SM on that GPU (2048 on Hopper and Blackwell).
        dtype (Optional[torch.dtype]):
            float32 (default), bfloat16 or float16. Drawn in fp32 and rounded, as torch does.

    Returns:
        torch.Tensor:
            Standard normal noise of shape `size` on the generator's device.
    """
    global _KERNEL
    if generator.device.type != "cuda":
        raise ValueError("randn_as_on_sm_count reproduces CUDA draws; pass a CUDA generator.")
    if dtype not in (None, torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"Unsupported dtype {dtype}; torch draws {dtype} through another path.")
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    numel = math.prod(size)
    grid = min(math.ceil(numel / _BLOCK), max_threads_per_sm // _BLOCK * sm_count)
    threads = _BLOCK * grid
    iterations = (numel - 1) // (threads * _UNROLL) + 1
    offset = generator.get_offset()
    if offset % 4 != 0:
        raise ValueError(f"Generator offset {offset} is not a multiple of 4.")

    out = torch.empty(numel, dtype=torch.float32, device=generator.device)
    with torch.cuda.device(generator.device):
        _KERNEL[(grid,)](
            out, generator.initial_seed(), offset // 4, threads, numel, iterations,
            _INV_2POW32, _INV_2POW32_HALF, _INV_2POW32_2PI, _INV_2POW32_2PI_HALF,
            BLOCK=_BLOCK, UNROLL=_UNROLL)
    generator.set_offset(offset + iterations * 4)       # torch's counter_offset
    return out.view(tuple(size)).to(dtype or torch.float32)


class StackedRandomGenerator:
    """
    One torch.Generator per seed, so each sample in a batch has its own reproducible stream.
    """

    def __init__(self, device: str, seeds: List[int], sm_count: Optional[int] = 132):
        """
        Create the generators.

        Args:
            device (str):
                CUDA device the noise is drawn on.
            seeds (List[int]):
                One seed per sample.
            sm_count (Optional[int]):
                Reproduce the draw of a GPU with this many SMs (132 = H100 SXM, what the
                model was validated on). None = the local GPU's native draw.
        """
        self.sm_count = sm_count
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        """
        Draw standard normal noise, one generator per leading index.

        Args:
            size (Tuple[int, ...]):
                Output shape; size[0] must equal the number of seeds.
            dtype (torch.dtype):
                Output dtype.
            device (str):
                Output device (the generators' device).

        Returns:
            torch.Tensor:
                Noise of shape `size`.
        """
        assert size[0] == len(self.generators)
        if self.sm_count is None:
            return torch.stack([
                torch.randn(size[1:], generator=gen, dtype=dtype, device=device)
                for gen in self.generators])
        return torch.stack([
            randn_as_on_sm_count(
                size=size[1:], generator=gen, sm_count=self.sm_count, dtype=dtype)
            for gen in self.generators])
