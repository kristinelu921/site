"""Faithful inference-time shape adaptation for the official LVM config."""
from __future__ import annotations


EPISODE_CONTEXT_TOKENS = 1280
EPISODE_NEW_TOKENS = 256
EPISODE_TOTAL_TOKENS = EPISODE_CONTEXT_TOKENS + EPISODE_NEW_TOKENS
VQ_CODEBOOK_SIZE = 8192
VISUAL_PAD_EOS_TOKEN_ID = VQ_CODEBOOK_SIZE


def make_vqlm_7b_config(config_class):
    config = config_class.load_config("vqlm_7b")
    config.bos_token_id = 1
    # IDs 0..8191 are all VQ image codes, including ID 2. Treating code 2 as
    # text EOS ends a frame early and fills its remainder with pad ID 8192.
    # Use the first suppressed special-token ID as an unreachable EOS sentinel
    # so visual generation always produces exactly 256 decodable codes.
    config.eos_token_id = VISUAL_PAD_EOS_TOKEN_ID
    config.tie_word_embeddings = False

    # Official blockwise attention requires the traced sequence length to be
    # exactly divisible by its chunk sizes. 1536 is the largest exact divisor
    # of this evaluation's complete 1280+256 episode and therefore forms one
    # mathematically exact attention block during constructor shape tracing.
    config.scan_query_chunk_size = EPISODE_TOTAL_TOKENS
    config.scan_key_chunk_size = EPISODE_TOTAL_TOKENS

    # Flax generation evaluates an initial 1280-token prompt and then 1-token
    # cached steps. No chunk >1 divides both shapes. The official non-scanned
    # branch calls the same feed_forward module directly and is mathematically
    # identical, so use it rather than changing model or sampling semantics.
    config.scan_mlp = False
    return config


def adaptation_record() -> dict:
    return {
        "schema": "lvm_vqlm_7b_visualcloze_shape_adaptation_v2",
        "context_tokens": EPISODE_CONTEXT_TOKENS,
        "new_tokens": EPISODE_NEW_TOKENS,
        "total_tokens": EPISODE_TOTAL_TOKENS,
        "scan_attention": True,
        "scan_query_chunk_size": EPISODE_TOTAL_TOKENS,
        "scan_key_chunk_size": EPISODE_TOTAL_TOKENS,
        "attention_chunks_at_total_length": 1,
        "scan_mlp": False,
        "weight_changes": 0,
        "visual_token_range": [0, VQ_CODEBOOK_SIZE - 1],
        "fixed_visual_tokens": EPISODE_NEW_TOKENS,
        "pad_token_id": VISUAL_PAD_EOS_TOKEN_ID,
        "eos_token_id": VISUAL_PAD_EOS_TOKEN_ID,
        "eos_policy": "unreachable suppressed sentinel; VQ code 2 is not text EOS",
        "sampling_changes": 1,
    }
