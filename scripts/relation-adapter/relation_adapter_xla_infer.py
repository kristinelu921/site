#!/usr/bin/env python3
"""RelationAdapter VisualCloze inference on a fail-closed PyTorch/XLA SPMD TPU."""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_COMMIT = "0ebdebcf5697f5881b5bf3d2fd58d2938733a4e7"
FLUX_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
ADAPTER_REVISION = "7f65ecd6c7718c6de92782dade849a4872aa986a"
SIGLIP_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
PROMPT = "Apply the demonstrated image transformation."
STEPS = 24
GUIDANCE_SCALE = 3.5
IP_SCALE = 1.0
IMAGE_SIZE = 256
CONDITIONS = ("real", "scrambled", "grey", "oracle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def validate_manifest(run_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected_pack = {
        3840: "48e062452574ff0ded61c72664445d687ecd1c02",
        512: "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2",
    }
    episodes = list(manifest["episodes"])
    assert len(episodes) in expected_pack
    assert manifest["pack_id"] == expected_pack[len(episodes)]
    assert manifest["seed"] == 42 and manifest["p"] == 2
    assert manifest["seed_rule"] == "[seed, task_seed(task), q, P]"
    assert manifest["demo_source_policy"] == "same"
    assert {row["condition"] for row in episodes} == set(CONDITIONS)
    assert len({row["episode_id"] for row in episodes}) == len(episodes)
    for row in episodes:
        path = run_dir / row["file"]
        assert sha256(path) == row["sha256"], row["episode_id"]
    return manifest, episodes


def model_seed(row: dict) -> int:
    # Identical across all four conditions and both native P=1 forwards.
    key = f"relation-adapter-v1|{row['task']}|{int(row['q'])}|P2"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF


def image_from(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, mode="RGB")


def resize_demo(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = 512.0 / max(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)


def recursive_shard_output(output, mesh) -> None:
    import torch
    import torch_xla.distributed.spmd as xs

    def visit(value):
        if isinstance(value, torch.Tensor) and value.ndim:
            spec = [None] * value.ndim
            candidates = [axis for axis, size in enumerate(value.shape) if axis > 0 and size % mesh.size() == 0]
            axis = candidates[0] if candidates else next((axis for axis, size in enumerate(value.shape) if size % mesh.size() == 0), None)
            if axis is not None:
                spec[axis] = "fsdp"
                xs.mark_sharding(value, mesh, tuple(spec))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif hasattr(value, "to_tuple"):
            visit(value.to_tuple())

    visit(output)


class MLPProjModel:
    @staticmethod
    def build(torch, cross_attention_dim: int, num_tokens: int):
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Sequential(
                    torch.nn.Linear(1152, 2304),
                    torch.nn.GELU(),
                    torch.nn.Linear(2304, cross_attention_dim * num_tokens),
                )
                self.norm = torch.nn.LayerNorm(cross_attention_dim)

            def forward(self, embeds):
                value = self.proj(embeds)
                value = value.reshape(-1, num_tokens, cross_attention_dim)
                return self.norm(value)

        return Projection()


def load_runtime(args, device, mesh):
    import torch
    import torch_xla.core.xla_model as xm
    from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
    from transformers import AutoProcessor, SiglipVisionModel

    source = str(args.source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from attention_processor import IPAFluxAttnProcessor2_0
    from src.pipeline_pe_clone import FluxPipeline
    from transformer_flux import FluxTransformer2DModel

    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(
        args.flux_dir, subfolder="transformer", torch_dtype=dtype, local_files_only=True
    )
    pipe = FluxPipeline.from_pretrained(
        args.flux_dir, transformer=transformer, torch_dtype=dtype, local_files_only=True
    )
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            PROMPT, PROMPT, device=torch.device("cpu"), max_sequence_length=512
        )

    pipe.load_lora_weights(args.adapter_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    processors = {}
    for name in transformer.attn_processors:
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            processors[name] = IPAFluxAttnProcessor2_0(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=128,
                scale=IP_SCALE,
            ).to(dtype=dtype)
        else:
            processors[name] = transformer.attn_processors[name]
    transformer.set_attn_processor(processors)

    state = torch.load(args.adapter_dir / "ip_adapter-100000.bin", map_location="cpu", weights_only=True, mmap=True)
    projection_state = state["image_proj"]
    if next(iter(projection_state)).startswith("module."):
        projection_state = {key.removeprefix("module."): value for key, value in projection_state.items()}
    image_projection = MLPProjModel.build(torch, transformer.config.joint_attention_dim, 128).to(dtype=dtype)
    image_projection.load_state_dict(projection_state, strict=True)
    adapter_state = state["ip_adapter"]
    if next(iter(adapter_state)).startswith("module."):
        adapter_state = {key.removeprefix("module."): value for key, value in adapter_state.items()}
    ip_layers = torch.nn.ModuleList(transformer.attn_processors.values())
    missing, unexpected = ip_layers.load_state_dict(adapter_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"IP adapter key mismatch missing={missing} unexpected={unexpected}")
    del state, projection_state, adapter_state, ip_layers

    image_encoder = SiglipVisionModel.from_pretrained(
        args.siglip_dir, torch_dtype=dtype, local_files_only=True
    ).eval()
    image_processor = AutoProcessor.from_pretrained(args.siglip_dir, local_files_only=True)

    # Prompt embeddings are fixed; release both text encoders before TPU model loading.
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    gc.collect()

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={FluxTransformerBlock, FluxSingleTransformerBlock},
    )
    transformer = SpmdFullyShardedDataParallel(
        transformer.eval(), mesh=mesh, auto_wrap_policy=auto_wrap, shard_output=recursive_shard_output
    )
    pipe.transformer = transformer
    pipe.vae = pipe.vae.eval().to(device)
    image_encoder = image_encoder.to(device)
    image_projection = image_projection.eval().to(device)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    xm.mark_step()
    return {
        "pipe": pipe,
        "image_encoder": image_encoder,
        "image_processor": image_processor,
        "image_projection": image_projection,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "dtype": dtype,
    }


def relation_embedding(runtime, input_image, target_image, device):
    import torch
    import torch_xla.core.xla_model as xm

    embeddings = []
    for image in (input_image, target_image):
        pixels = runtime["image_processor"](images=resize_demo(image), return_tensors="pt").pixel_values
        pixels = pixels.to(device=device, dtype=runtime["dtype"])
        with torch.no_grad():
            pooled = runtime["image_encoder"](pixels).pooler_output
            embeddings.append(runtime["image_projection"](pooled))
        xm.mark_step()
    return torch.cat(embeddings, dim=0)


def predict(runtime, packed, row, device):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    query = image_from(packed["query_input"])
    seed = model_seed(row)
    native = []
    for demo in range(2):
        relation = relation_embedding(
            runtime,
            image_from(packed["demo_inputs"][demo]),
            image_from(packed["demo_targets"][demo]),
            device,
        )
        torch_xla.manual_seed(seed, device=device)
        with torch.no_grad():
            result = runtime["pipe"](
                prompt=None,
                prompt_embeds=runtime["prompt_embeds"],
                pooled_prompt_embeds=runtime["pooled_prompt_embeds"],
                image_emb=relation,
                condition_image=query,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=STEPS,
                generator=None,
                output_type="pt",
            ).images
        xm.mark_step()
        native.append(result.float().cpu())
    averaged = torch.stack(native, dim=0).mean(dim=0)[0]
    image = (averaged.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return image, seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--flux-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--siglip-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-devices", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False)
    xr.use_spmd()
    runtime_devices = int(xr.global_runtime_device_count())
    addressable = int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if backend.upper() != "TPU" or runtime_devices != args.expected_devices or addressable != args.expected_devices:
        raise SystemExit(f"refusing fallback: backend={backend!r}, runtime={runtime_devices}, addressable={addressable}")
    mesh = xs.Mesh(np.arange(runtime_devices), (runtime_devices,), ("fsdp",))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    evidence = {
        "backend": backend,
        "device": str(device),
        "runtime_device_count": runtime_devices,
        "addressable_runtime_device_count": addressable,
        "spmd_mesh_device_ids": list(range(runtime_devices)),
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "host": socket.gethostname(),
        "tpu_name": os.environ.get("EXPECTED_TPU_NAME"),
        "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE"),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)

    run_dir = args.run_dir.resolve()
    manifest, episodes = validate_manifest(run_dir)
    if args.limit:
        episodes = episodes[: args.limit]
    runtime = load_runtime(args, device, mesh)

    expected = {
        "schema": "relation_adapter_visualcloze_inference_v1",
        "pack_id": manifest["pack_id"],
        "source_commit": SOURCE_COMMIT,
        "flux_revision": FLUX_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "siglip_revision": SIGLIP_REVISION,
        "backend": "TPU",
        "runtime_device_count": runtime_devices,
        "adapter": "p2_mean_of_two_native_relationadapter_p1_float_v1",
        "prompt": PROMPT,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "ip_scale": IP_SCALE,
    }
    partial = run_dir / "inference.partial.json"
    final = run_dir / "inference.json"
    previous_path = partial if partial.exists() else final
    rows = []
    if args.resume and previous_path.exists():
        previous = json.loads(previous_path.read_text())
        mismatch = {key: (previous.get(key), value) for key, value in expected.items() if previous.get(key) != value}
        if mismatch:
            raise SystemExit(f"resume identity mismatch: {mismatch}")
        for row in previous.get("results", []):
            output = run_dir / row["prediction"]
            if not output.is_file() or sha256(output) != row["prediction_sha256"]:
                raise SystemExit(f"invalid resume output: {row.get('episode_id')}")
            rows.append(row)
    complete = {row["episode_id"] for row in rows}
    predictions = run_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    log = dict(expected, results=rows, status="running")
    atomic_json(partial, log)

    for index, episode in enumerate(episodes, 1):
        if episode["episode_id"] in complete:
            continue
        started = time.monotonic()
        with np.load(run_dir / episode["file"], allow_pickle=False) as source:
            packed = {key: source[key].copy() for key in source.files}
        prediction, seed = predict(runtime, packed, episode, device)
        relative = Path("predictions") / episode["task"] / episode["condition"] / f"q{int(episode['q']):02d}.png"
        output = run_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp.png")
        Image.fromarray(prediction, mode="RGB").save(temp, format="PNG", optimize=False)
        os.replace(temp, output)
        result = {
            "episode_id": episode["episode_id"],
            "task": episode["task"],
            "q": int(episode["q"]),
            "condition": episode["condition"],
            "prediction": str(relative),
            "prediction_sha256": sha256(output),
            "model_seed": seed,
            "seconds": round(time.monotonic() - started, 4),
        }
        rows.append(result)
        complete.add(episode["episode_id"])
        log["results"] = rows
        atomic_json(partial, log)
        if len(rows) % args.log_every == 0:
            print(f"PROGRESS {len(rows)}/{len(episodes)} last={episode['episode_id']} seconds={result['seconds']}", flush=True)

    log["status"] = "terminal_all_images_and_metadata_committed"
    log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(final, log)
    partial.unlink(missing_ok=True)
    print(f"TERMINAL {len(rows)}/{len(episodes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
