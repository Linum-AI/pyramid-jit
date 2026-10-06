# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: __init__.py
Description: P-JiT (Pyramid-JiT) — a pixel-space text-to-image diffusion transformer.
"""

from pyramid_jit.attention import active_attention_backend, set_attention_backend
from pyramid_jit.config import PyramidJiTConfig, SamplerConfig
from pyramid_jit.model import PyramidJiT
from pyramid_jit.noise import randn_as_on_sm_count
from pyramid_jit.sampler import generate
from pyramid_jit.text_encoder import QwenTextEncoder

__all__ = [
    "PyramidJiT",
    "PyramidJiTConfig",
    "QwenTextEncoder",
    "SamplerConfig",
    "active_attention_backend",
    "generate",
    "randn_as_on_sm_count",
    "set_attention_backend",
]
