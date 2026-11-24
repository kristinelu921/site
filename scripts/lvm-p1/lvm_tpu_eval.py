#!/usr/bin/env python3
"""Run released LVM-7B on the clean P=1 real-condition test episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lvm_runtime_config import (
    EPISODE_CONTEXT_TOKENS,
    EPISODE_TOTAL_TOKENS,
    VISUAL_PAD_EOS_TOKEN_ID,
    VQ_CODEBOOK_SIZE,
    adaptation_record,
    make_vqlm_7b_config,
)


PACK_ID = "48e062452574ff0ded61c72664445d687ecd1c02"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def official_preprocess(frame_u8: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame_u8, np.uint8)).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_context(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        frames = [data["demo_inputs"][0], data["demo_targets"][0], data["query_input"]]
    result = np.stack([official_preprocess(frame) for frame in frames])
    if result.shape != (3, 256, 256, 3) or result.dtype != np.float32:
        raise ValueError(f"bad official context tensor: {result.shape} {result.dtype}")
    return result


def output_path(output_dir: Path, episode: dict) -> Path:
    return output_dir / episode["task"] / "real" / f"q{int(episode['q']):02d}.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vq-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-devices", type=int, choices=(4, 8), required=True)
    parser.add_argument("--torch-threads", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("pack_id") != PACK_ID or manifest.get("p") != 1 or manifest.get("conditions") != ["real"]:
        raise SystemExit("refusing non-P1/real test manifest")
    if len(manifest.get("episodes", [])) != 960:
        raise SystemExit("refusing incomplete 960-query manifest")
    if manifest.get("context_frames") != 3 or manifest.get("context_tokens") != 768:
        raise SystemExit("context contract mismatch")

    sys.path.insert(0, str(args.official_repo))
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax.experimental.pjit import pjit  # noqa: PLC0415
    from jax.sharding import PartitionSpec as PS  # noqa: PLC0415
    from transformers import FlaxLogitsProcessorList, GenerationConfig  # noqa: PLC0415
    from EasyLM.checkpoint import StreamingCheckpointer  # noqa: PLC0415
    from EasyLM.jax_utils import (  # noqa: PLC0415
        get_float_dtype_by_name,
        make_shard_and_gather_fns,
        match_partition_rules,
        tree_apply,
    )
    from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM  # noqa: PLC0415
    from evaluation.vqlm_demo.vqvae_muse import VQGANModel  # noqa: PLC0415

    backend = jax.default_backend()
    devices = jax.devices()
    evidence = {
        "backend": backend,
        "device_count": len(devices),
        "devices": [str(device) for device in devices],
        "jax_version": jax.__version__,
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    if backend != "tpu" or len(devices) != args.expected_devices or any("TPU" not in str(x).upper() for x in devices):
        raise SystemExit(f"refusing unexpected backend: {evidence}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backend.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "runtime_config.json").write_text(
        json.dumps(adaptation_record(), indent=2, sort_keys=True) + "\n"
    )
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(min(4, args.torch_threads))
    vq = VQGANModel.from_pretrained(str(args.vq_checkpoint)).to("cpu").eval()

    config = make_vqlm_7b_config(LLaMAConfig)
    with jax.default_device(jax.devices("cpu")[0]):
        _, host_params = StreamingCheckpointer.load_trainstate_checkpoint(
            f"params::{args.checkpoint}", disallow_trainstate=True
        )
        model = FlaxLLaMAForCausalLM(
            config, input_shape=(1, EPISODE_TOTAL_TOKENS), seed=args.model_seed,
            dtype=jnp.float16, _do_init=False,
        )
    model_ps = match_partition_rules(LLaMAConfig.get_partition_rules(), host_params)
    shard_fns, _ = make_shard_and_gather_fns(model_ps, get_float_dtype_by_name("fp16"))
    mesh = LLaMAConfig.get_jax_mesh(f"1,{args.expected_devices},1")
    started = time.monotonic()
    with mesh:
        params = tree_apply(shard_fns, host_params)
    del host_params
    print(f"PARAMS_SHARDED seconds={time.monotonic() - started:.2f}", flush=True)

    class SuppressNonVQTokens:
        def __call__(self, input_ids, scores, cur_len):
            del input_ids, cur_len
            return scores.at[:, VQ_CODEBOOK_SIZE:].set(jnp.finfo(scores.dtype).min)

    processors = FlaxLogitsProcessorList([SuppressNonVQTokens()])
    generation_config = GenerationConfig(
        max_new_tokens=256,
        pad_token_id=VISUAL_PAD_EOS_TOKEN_ID,
        bos_token_id=1,
        eos_token_id=VISUAL_PAD_EOS_TOKEN_ID,
        do_sample=True,
        num_beams=1,
        top_k=50,
        top_p=1.0,
        temperature=1.0,
    )

    @partial(pjit, in_shardings=(model_ps, PS(), PS()), out_shardings=PS())
    def generate(params_tree, input_tokens, key):
        sequences = model.generate(
            input_tokens,
            attention_mask=jnp.ones_like(input_tokens),
            params=params_tree["params"],
            prng_key=key,
            logits_processor=processors,
            generation_config=generation_config,
        ).sequences
        return sequences[:, input_tokens.shape[1]:]

    episodes = manifest["episodes"][: args.limit or None]
    journal = args.output_dir / "inference.jsonl"
    completed = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (args.output_dir / row["prediction"]).is_file():
                    completed.add(row["episode_id"])
    pending = [row for row in episodes if row["episode_id"] not in completed]
    print(f"RUN total={len(episodes)} pending={len(pending)} P=1 condition=real", flush=True)

    for index, episode in enumerate(pending, 1):
        episode_started = time.monotonic()
        context = load_context(args.manifest.parent / episode["file"])
        with torch.no_grad():
            tensor = torch.from_numpy(context).permute(0, 3, 1, 2).contiguous()
            _, token_ids = vq.encode(tensor)
        input_ids = np.asarray(token_ids.reshape(1, -1).cpu(), dtype=np.int32)
        if input_ids.shape != (1, EPISODE_CONTEXT_TOKENS):
            raise ValueError(f"bad VQ prompt shape: {input_ids.shape}")
        encode_seconds = time.monotonic() - episode_started

        generation_started = time.monotonic()
        with mesh:
            generated = generate(params, input_ids, jax.random.PRNGKey(args.model_seed))
            generated = np.asarray(jax.device_get(generated), dtype=np.int64)
        generation_seconds = time.monotonic() - generation_started
        if generated.shape != (1, 256) or generated.min() < 0 or generated.max() >= VQ_CODEBOOK_SIZE:
            raise ValueError(f"invalid visual tokens: {generated.shape}, {generated.min()}..{generated.max()}")

        decode_started = time.monotonic()
        with torch.no_grad():
            decoded = vq.decode_code(torch.from_numpy(generated).long())
        decoded_float = np.clip(decoded[0].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
        # Exact official release export: truncating uint8 conversion, not rounding.
        image = (decoded_float * 255).astype(np.uint8)
        decode_seconds = time.monotonic() - decode_started
        if image.shape != (256, 256, 3):
            raise ValueError(f"bad native decoded image: {image.shape}")
        output = output_path(args.output_dir, episode)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.png")
        Image.fromarray(image).save(temporary)
        os.replace(temporary, output)
        record = {
            "episode_id": episode["episode_id"],
            "task": episode["task"],
            "q": episode["q"],
            "p": 1,
            "condition": "real",
            "prediction": str(output.relative_to(args.output_dir)),
            "prediction_sha256": sha256(output),
            "encode_seconds": encode_seconds,
            "generation_seconds": generation_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": time.monotonic() - episode_started,
            "model_seed": args.model_seed,
            "official_resize": "PIL RGB BICUBIC 256x256",
            "official_postprocess": "clip,*255,uint8_truncate,no_resize",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(journal, record)
        print(f"[{index}/{len(pending)}] {episode['episode_id']} total={record['total_seconds']:.2f}s", flush=True)
        if index == 1:
            (args.output_dir / "smoke_passed.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print("SMOKE_PASS " + json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
