#!/usr/bin/env python3
"""Pinned component loader for the official Prompt Diffusion release."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROMPT = "an image"
NEGATIVE_PROMPT = ""
SOURCE_COMMIT = "8ecd2d126d9f9e17b80d9655bf8ed2df2e93f660"
MODEL_REVISION = "60ea7b6cc48d08487f95d06ba6ce59e15b619efd"


def load_components(source_dir: Path, model_dir: Path, dtype=torch.float32):
    """Load every runtime component from local immutable files only."""
    source_dir = source_dir.resolve()
    model_dir = model_dir.resolve()
    sys.path.insert(0, str(source_dir))
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from pipeline_prompt_diffusion import PromptDiffusionPipeline
    from promptdiffusioncontrolnet import PromptDiffusionControlNetModel

    model_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": False,
    }
    controlnet = PromptDiffusionControlNetModel.from_pretrained(
        model_dir / "controlnet", **model_kwargs
    )
    unet = UNet2DConditionModel.from_pretrained(model_dir / "unet", **model_kwargs)
    vae = AutoencoderKL.from_pretrained(model_dir / "vae", **model_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(
        model_dir / "text_encoder", **model_kwargs
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_dir / "tokenizer", local_files_only=True
    )
    scheduler = DDIMScheduler.from_pretrained(
        model_dir / "scheduler", local_files_only=True
    )
    pipeline = PromptDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    return pipeline


def load_runtime_components(source_dir: Path, model_dir: Path, dtype=torch.bfloat16):
    """Load only the components needed after prompt embeddings are frozen."""
    source_dir = source_dir.resolve()
    model_dir = model_dir.resolve()
    sys.path.insert(0, str(source_dir))
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from promptdiffusioncontrolnet import PromptDiffusionControlNetModel

    model_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": False,
    }
    controlnet = PromptDiffusionControlNetModel.from_pretrained(
        model_dir / "controlnet", **model_kwargs
    )
    unet = UNet2DConditionModel.from_pretrained(model_dir / "unet", **model_kwargs)
    vae = AutoencoderKL.from_pretrained(model_dir / "vae", **model_kwargs)
    scheduler = DDIMScheduler.from_pretrained(
        model_dir / "scheduler", local_files_only=True
    )
    return controlnet, unet, vae, scheduler


def encode_fixed_prompts(pipeline):
    """Encode the fixed positive and empty negative prompts on CPU."""
    tokens = pipeline.tokenizer(
        [PROMPT, NEGATIVE_PROMPT],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = pipeline.text_encoder(tokens.input_ids)[0].cpu()
    return embeddings[:1].contiguous(), embeddings[1:].contiguous()
