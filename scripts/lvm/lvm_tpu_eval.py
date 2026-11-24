#!/usr/bin/env python3
"""Official EasyLM/JAX LVM generation on a single 8-chip TPU host."""
from __future__ import annotations

import argparse
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
    EPISODE_TOTAL_TOKENS,
    VISUAL_PAD_EOS_TOKEN_ID,
    VQ_CODEBOOK_SIZE,
    adaptation_record,
    make_vqlm_7b_config,
)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_context(path: Path) -> np.ndarray:
    with np.load(path) as data:
        demo_inputs = np.asarray(data["demo_inputs"], np.float32) / 255.0
        demo_targets = np.asarray(data["demo_targets"], np.float32) / 255.0
        query_input = np.asarray(data["query_input"], np.float32) / 255.0
    frames = []
    for demo_input, demo_target in zip(demo_inputs, demo_targets):
        frames.extend((demo_input, demo_target))
    frames.append(query_input)
    result = np.stack(frames)
    assert result.shape == (5, 256, 256, 3), result.shape
    return result


def output_path(output_dir: Path, episode: dict) -> Path:
    return output_dir / episode["task"] / episode["condition"] / f"q{int(episode['q']):02d}.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vq-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=32)
    args = parser.parse_args()

    sys.path.insert(0, str(args.official_repo))
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from flax.core import freeze  # noqa: PLC0415
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
    from EasyLM.models.llama.llama_model import (  # noqa: PLC0415
        LLaMAConfig,
        FlaxLLaMAForCausalLM,
    )
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
    if backend != "tpu" or len(devices) != 8 or any("TPU" not in str(x).upper() for x in devices):
        raise SystemExit(f"refusing non-TPU execution: {evidence}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backend.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(min(4, args.torch_threads))
    vq = VQGANModel.from_pretrained(str(args.vq_checkpoint)).to("cpu").eval()

    config = make_vqlm_7b_config(LLaMAConfig)
    (args.output_dir / "runtime_config.json").write_text(
        json.dumps(adaptation_record(), indent=2, sort_keys=True) + "\n"
    )
    with jax.default_device(jax.devices("cpu")[0]):
        _, host_params = StreamingCheckpointer.load_trainstate_checkpoint(
            f"params::{args.checkpoint}", disallow_trainstate=True
        )
        model = FlaxLLaMAForCausalLM(
            config,
            input_shape=(1, EPISODE_TOTAL_TOKENS),
            seed=args.model_seed,
            dtype=jnp.float16,
            _do_init=False,
        )

    model_ps = match_partition_rules(LLaMAConfig.get_partition_rules(), host_params)
    shard_fns, _ = make_shard_and_gather_fns(model_ps, get_float_dtype_by_name("fp16"))
    mesh = LLaMAConfig.get_jax_mesh("1,8,1")
    shard_started = time.monotonic()
    with mesh:
        params = tree_apply(shard_fns, host_params)
    del host_params
    print(f"PARAMS_SHARDED seconds={time.monotonic() - shard_started:.2f}", flush=True)

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
        output = model.generate(
            input_tokens,
            attention_mask=jnp.ones_like(input_tokens),
            params=params_tree["params"],
            prng_key=key,
            logits_processor=processors,
            generation_config=generation_config,
        ).sequences
        return output[:, input_tokens.shape[1]:]

    manifest = json.loads(args.manifest.read_text())
    episodes = manifest["episodes"][: args.limit or None]
    journal = args.output_dir / "inference.jsonl"
    complete = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                complete.add(json.loads(line)["episode_id"])
    pending = [ep for ep in episodes if ep["episode_id"] not in complete or not output_path(args.output_dir, ep).is_file()]
    print(f"RUN manifest={args.manifest} total={len(episodes)} pending={len(pending)}", flush=True)

    for index, episode in enumerate(pending):
        started = time.monotonic()
        context = load_context(args.manifest.parent / episode["file"])
        with torch.no_grad():
            tensor = torch.from_numpy(context).permute(0, 3, 1, 2).contiguous()
            _, token_ids = vq.encode(tensor)
        input_ids = np.asarray(token_ids.reshape(1, -1).cpu(), dtype=np.int32)
        assert input_ids.shape == (1, 1280), input_ids.shape
        encode_seconds = time.monotonic() - started

        generation_started = time.monotonic()
        with mesh:
            generated = generate(params, input_ids, jax.random.PRNGKey(args.model_seed))
            generated = np.asarray(jax.device_get(generated), dtype=np.int64)
        generation_seconds = time.monotonic() - generation_started
        if generated.shape != (1, 256) or generated.min() < 0 or generated.max() >= VQ_CODEBOOK_SIZE:
            raise ValueError(f"invalid generated visual tokens: shape={generated.shape}, range={generated.min()}..{generated.max()}")

        decode_started = time.monotonic()
        with torch.no_grad():
            decoded = vq.decode_code(torch.from_numpy(generated).long())
        image = np.rint(np.clip(decoded[0].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
        decode_seconds = time.monotonic() - decode_started
        out = output_path(args.output_dir, episode)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.png")
        Image.fromarray(image).save(tmp)
        os.replace(tmp, out)
        total_seconds = time.monotonic() - started
        record = {
            "episode_id": episode["episode_id"],
            "task": episode["task"],
            "q": episode["q"],
            "condition": episode["condition"],
            "output": str(out),
            "encode_seconds": encode_seconds,
            "generation_seconds": generation_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "model_seed": args.model_seed,
            "backend": "tpu",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(journal, record)
        print(f"[{index + 1}/{len(pending)}] {episode['episode_id']} total={total_seconds:.2f}s generate={generation_seconds:.2f}s", flush=True)
        if index == 0 and not (args.output_dir / "smoke_passed.json").exists():
            (args.output_dir / "smoke_passed.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print("SMOKE_PASS " + json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
