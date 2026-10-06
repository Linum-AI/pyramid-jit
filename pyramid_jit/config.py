# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: config.py
Description: Architecture and sampler configuration for P-JiT. The defaults are exactly the
    released model; the architecture dataclass exists so the shapes are documented in one
    place and can be round-tripped through the `config.json` shipped with the weights, not
    because other shapes are supported.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class PyramidJiTConfig:
    """
    P-JiT architecture: one single-stream DiT that patchifies 32x32 pixels into a token and
    predicts the clean image at full resolution. During training two extra readout heads
    predicted the image at 128x128 (after block 10) and 256x256 (after block 16); they are not
    used for sampling and are not part of the released weights.
    """

    dim: int = 2944
    num_heads: int = 23
    num_layers: int = 22
    ffn_dim: int = 7936               # SwiGLU target width; hidden = round256(2/3 * ffn_dim)
    refiner_blocks: int = 2           # Per-stream pre-conditioner depth (text + image)
    refiner_ffn_dim: int = 4096
    adaln_rank: int = 256             # Low-rank AdaLN: shared down-projection width
    text_dim: int = 7680              # 3 x 2560: Qwen3.5-4B hidden states, layers (7, 15, 27)
    text_len: int = 256
    in_channels: int = 3
    out_channels: int = 3
    freq_dim: int = 256               # Sinusoidal timestep embedding width
    eps: float = 1e-6
    bottleneck_dim: int = 1024        # BottleneckPatchEmbed intermediate channels (3:1)
    patch: Tuple[int, int, int] = (1, 32, 32)
    rope_max_positions: int = 1024
    text_layers: Tuple[int, int, int] = (7, 15, 27)

    @property
    def head_dim(self) -> int:
        """Per-head width (128 for the released model)."""
        return self.dim // self.num_heads

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PyramidJiTConfig":
        """
        Build a config from a plain dict (e.g. the `architecture` block of `config.json`).

        Args:
            data (Dict[str, Any]):
                Field values; tuple fields may be given as lists.

        Returns:
            PyramidJiTConfig:
                The config.
        """
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise KeyError(f"Unknown PyramidJiTConfig fields: {unknown}")
        clean = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()}
        return cls(**clean)

    @classmethod
    def from_json(cls, path: str) -> "PyramidJiTConfig":
        """
        Read the `architecture` block of a `config.json`.

        Args:
            path (str):
                Path to `config.json`.

        Returns:
            PyramidJiTConfig:
                The config.
        """
        with open(path) as fh:
            data = json.load(fh)
        return cls.from_dict(data["architecture"])

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Returns:
            Dict[str, Any]:
                Field values with tuples as lists.
        """
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


@dataclass
class SamplerConfig:
    """
    Sampling settings. The defaults are the ones the released model was validated with during
    training: 50 uniform Euler steps from t=1 (noise) to t=0, adaptive projected guidance
    (APG) at scale 15, initial noise scaled by 2 and drawn as on an H100 SXM (see noise.py).
    """

    height: int = 512
    width: int = 512
    sampling_steps: int = 50
    guidance_scale: float = 15.0
    apg_momentum: float = -0.75
    apg_eta: float = 0.0
    apg_rescale: float = 10.0
    noise_scale: float = 2.0
    negative_prompt: str = "watermark, signature, logo, copyright, url"
    # SM count of the GPU whose seeded noise to reproduce. 132 = H100 SXM, the GPU the model
    # was trained and validated on, so a seed gives the same image on every NVIDIA GPU.
    # None draws with the local GPU's native torch.randn (seed results then vary by GPU model).
    noise_sm_count: Optional[int] = 132
    num_timesteps: int = field(default=1000, repr=False)   # Timestep-embedding scale (fixed)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Returns:
            Dict[str, Any]:
                Field values.
        """
        return asdict(self)
