"""P=1 shape adaptation for the released LVM-7B EasyLM checkpoint."""
from __future__ import annotations


P = 1
CONTEXT_FRAMES = 2 * P + 1
TOKENS_PER_FRAME = 256
EPISODE_CONTEXT_TOKENS = CONTEXT_FRAMES * TOKENS_PER_FRAME
EPISODE_NEW_TOKENS = TOKENS_PER_FRAME
EPISODE_TOTAL_TOKENS = EPISODE_CONTEXT_TOKENS + EPISODE_NEW_TOKENS
VQ_CODEBOOK_SIZE = 8192
VISUAL_PAD_EOS_TOKEN_ID = VQ_CODEBOOK_SIZE


def make_vqlm_7b_config(config_class):
    config = config_class.load_config("vqlm_7b")
    config.bos_token_id = 1
    # Every ID 0..8191 is a valid visual code. The released HF config labels
    # visual code 2 as text EOS; use the first suppressed non-visual ID so a
    # frame always contains exactly 256 decodable visual tokens.
    config.eos_token_id = VISUAL_PAD_EOS_TOKEN_ID
    config.tie_word_embeddings = False
    # One mathematically exact attention block at this evaluation length.
    config.scan_query_chunk_size = EPISODE_TOTAL_TOKENS
    config.scan_key_chunk_size = EPISODE_TOTAL_TOKENS
    # Generation alternates a 768-token prefill with 1-token cached steps;
    # the official dense branch calls the identical feed-forward module.
    config.scan_mlp = False
    return config


def adaptation_record() -> dict:
    return {
        "schema": "lvm_vqlm_7b_visualcloze_p1_shape_adaptation_v1",
        "p": P,
        "context_frames": CONTEXT_FRAMES,
        "context_tokens": EPISODE_CONTEXT_TOKENS,
        "new_tokens": EPISODE_NEW_TOKENS,
        "total_tokens": EPISODE_TOTAL_TOKENS,
        "scan_attention": True,
        "scan_query_chunk_size": EPISODE_TOTAL_TOKENS,
        "scan_key_chunk_size": EPISODE_TOTAL_TOKENS,
        "scan_mlp": False,
        "weight_changes": 0,
        "visual_token_range": [0, VQ_CODEBOOK_SIZE - 1],
        "fixed_visual_tokens": EPISODE_NEW_TOKENS,
        "pad_token_id": VISUAL_PAD_EOS_TOKEN_ID,
        "eos_token_id": VISUAL_PAD_EOS_TOKEN_ID,
        "eos_policy": "unreachable suppressed sentinel; VQ code 2 remains a visual token",
        "official_preprocess": "RGB -> PIL BICUBIC 256x256 -> float32 / 255",
        "official_postprocess": "VQ decode; clamp [0,1]; *255; uint8 truncation; no resize",
    }
