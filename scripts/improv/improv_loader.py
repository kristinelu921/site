#!/usr/bin/env python3
"""Strict local loader for the pinned official IMProv checkpoint."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


PROMPT = "Left - input image, right - output image."


def _public_config(path: Path, cls) -> dict:
    raw = json.loads(path.read_text())
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in raw.items() if key in accepted and key != "self"}


def _load_safetensors(module, path: Path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"strict checkpoint mismatch for {path}: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def load_components(source_dir: Path, model_dir: Path, vq_dir: Path, *, load_text: bool):
    source_dir = source_dir.resolve()
    model_dir = model_dir.resolve()
    vq_dir = vq_dir.resolve()
    sys.path.insert(0, str(source_dir))

    from improv.modeling.meta_arch.modules import PatchEmbedEncoder, TransformerMAE, VQDecoder
    from improv.pipelines.pipeline_improv import MAEScheduler

    image_encoder = PatchEmbedEncoder(
        **_public_config(model_dir / "image_encoder" / "config.json", PatchEmbedEncoder)
    )
    _load_safetensors(
        image_encoder, model_dir / "image_encoder" / "diffusion_pytorch_model.safetensors"
    )

    # The official decoder config names its separately published base VQ model.
    # Instantiate from the pinned local copy, then strictly apply the official
    # IMProv decoder state without rewriting either checkpoint.
    decoder_config = json.loads((model_dir / "image_decoder" / "config.json").read_text())
    image_decoder = VQDecoder(
        pretrained_model_name_or_path=str(vq_dir),
        pixel_mean=tuple(decoder_config["pixel_mean"]),
        pixel_std=tuple(decoder_config["pixel_std"]),
    )
    _load_safetensors(
        image_decoder, model_dir / "image_decoder" / "diffusion_pytorch_model.safetensors"
    )

    mask_image_model = TransformerMAE(
        **_public_config(model_dir / "mask_image_model" / "config.json", TransformerMAE)
    )
    _load_safetensors(
        mask_image_model,
        model_dir / "mask_image_model" / "diffusion_pytorch_model.safetensors",
    )
    scheduler = MAEScheduler.from_pretrained(str(model_dir / "scheduler"), local_files_only=True)

    text_encoder = tokenizer = None
    if load_text:
        from transformers import CLIPTextModel, CLIPTokenizer

        text_encoder = CLIPTextModel.from_pretrained(
            str(model_dir / "text_encoder"), local_files_only=True
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            str(model_dir / "tokenizer"), local_files_only=True
        )
    return image_encoder, image_decoder, mask_image_model, scheduler, text_encoder, tokenizer


def encode_fixed_prompt(text_encoder, tokenizer):
    import torch

    inputs = tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = None
    if getattr(text_encoder.config, "use_attention_mask", False):
        attention_mask = inputs.attention_mask
    with torch.no_grad():
        embedding = text_encoder(inputs.input_ids, attention_mask=attention_mask)[0]
    if tuple(embedding.shape) != (1, 77, 1024):
        raise RuntimeError(f"unexpected fixed-prompt embedding shape: {tuple(embedding.shape)}")
    return embedding.detach().cpu().float().contiguous()
