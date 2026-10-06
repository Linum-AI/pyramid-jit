# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: text_encoder.py
Description: Caption encoder for P-JiT. Runs Qwen3.5-4B on the chat-templated caption and
    concatenates the hidden states of three of its full-attention layers (7, 15, 27) along
    the feature axis: 3 x 2560 = 7680 features per token (the FLUX.2 recipe). P-JiT was
    trained on exactly these embeddings.
"""

from typing import List, Optional, Sequence

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as transformers_logging,
)


class QwenTextEncoder:
    """
    Multi-layer Qwen3.5-4B hidden-state encoder.
    """

    def __init__(
            self,
            model_path: str,
            extraction_layers: Sequence[int] = (7, 15, 27),
            max_length: int = 256,
            device: Optional[torch.device] = None):
        """
        Load the model and tokenizer.

        Args:
            model_path (str):
                Local directory or Hugging Face id of Qwen3.5-4B (tokenizer + weights).
            extraction_layers (Sequence[int]):
                0-indexed transformer blocks whose outputs are concatenated. They must be
                full-attention layers of the hybrid Qwen3.5 stack (validated at load).
            max_length (int):
                Maximum caption tokens after templating (longer captions are truncated).
            device (Optional[torch.device]):
                Device to load onto. Default: current CUDA device.
        """
        self.device = torch.cuda.current_device() if device is None else device
        self.max_length = max_length
        self.extraction_layers = list(extraction_layers)

        config = AutoConfig.from_pretrained(model_path)
        layer_types = config.text_config.layer_types
        for idx in self.extraction_layers:
            if idx >= len(layer_types) or layer_types[idx] != "full_attention":
                full = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
                raise ValueError(
                    f"Layer {idx} is not a full-attention layer; Qwen3.5's full-attention "
                    f"layers are {full}.")

        verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        transformers_logging.set_verbosity(verbosity)

    def __call__(self, text: List[str]) -> List[torch.Tensor]:
        """
        Encode captions.

        Args:
            text (List[str]):
                Captions.

        Returns:
            List[torch.Tensor]:
                One bf16 tensor per caption of shape (num_tokens, 7680), padding removed.
        """
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False)
            for caption in text]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states[0] is the embedding output, so block i is hidden_states[i + 1].
        hidden_states = outputs.hidden_states
        attention_mask = inputs["attention_mask"]
        results = []
        for i in range(len(text)):
            seq_len = attention_mask[i].sum().item()
            layers = [
                hidden_states[layer_idx + 1][i, :seq_len, :].to(dtype=torch.bfloat16)
                for layer_idx in self.extraction_layers]
            results.append(torch.cat(layers, dim=-1))
        return results
