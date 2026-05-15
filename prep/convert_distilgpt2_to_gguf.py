#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
#   "safetensors",
#   "gguf>=0.10",
#   "numpy",
# ]
# ///
#
# Convert HF distilgpt2 (and GPT-2 small/medium/large/xl, same arch) to a
# project-native GGUF.
#
# HF layout (Conv1D, stored [in, out]):
#   transformer.wte.weight              [vocab, 768]
#   transformer.wpe.weight              [n_ctx, 768]
#   transformer.h.N.ln_1.{weight,bias}  [768]
#   transformer.h.N.attn.c_attn.weight  [768, 2304]   concatenated Q|K|V
#   transformer.h.N.attn.c_attn.bias    [2304]
#   transformer.h.N.attn.c_proj.weight  [768, 768]    output projection
#   transformer.h.N.attn.c_proj.bias    [768]
#   transformer.h.N.ln_2.{weight,bias}  [768]
#   transformer.h.N.mlp.c_fc.weight     [768, 3072]   FFN up
#   transformer.h.N.mlp.c_fc.bias       [3072]
#   transformer.h.N.mlp.c_proj.weight   [3072, 768]   FFN down
#   transformer.h.N.mlp.c_proj.bias     [768]
#   transformer.ln_f.{weight,bias}      [768]
#
# Output GGUF: c_attn is split into separate Q/K/V tensors (per-head split
# stays in the Ruby loader since ggml/our Mat layout makes that a no-op
# slice). Everything stored row-major F32.
#
# Note: we deliberately do NOT embed the BPE tokenizer in this GGUF. The
# loader path uses pre-tokenized int IDs (see prep/tokenize_gpt2.py).

import argparse
import os
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import gguf

# ---- defaults / config ------------------------------------------------------

REPO_ID = "distilgpt2"   # HF model id; same converter handles "gpt2" etc.
OUT     = Path("data/distilgpt2-f32.gguf")
CACHE   = Path("prep/_hf_cache")

# ---- helpers ----------------------------------------------------------------

def split_qkv(c_attn: np.ndarray, n_embd: int):
    """HF c_attn output dim is [Q | K | V]. Split along the *last* axis."""
    assert c_attn.shape[-1] == 3 * n_embd, f"unexpected c_attn shape: {c_attn.shape}"
    q = c_attn[..., 0          : n_embd      ].copy()
    k = c_attn[..., n_embd     : 2 * n_embd  ].copy()
    v = c_attn[..., 2 * n_embd : 3 * n_embd  ].copy()
    return q, k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--out",     default=str(OUT))
    ap.add_argument("--cache",   default=str(CACHE))
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    # 1. Pull config + safetensors. Cached on disk.
    print(f"[1/4] downloading {args.repo_id} → {args.cache}")
    cfg_path  = hf_hub_download(args.repo_id, "config.json",      cache_dir=args.cache)
    sft_path  = hf_hub_download(args.repo_id, "model.safetensors", cache_dir=args.cache)

    import json
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_vocab  = cfg["vocab_size"]
    n_ctx    = cfg["n_ctx"]
    n_embd   = cfg["n_embd"]
    n_head   = cfg["n_head"]
    n_layer  = cfg["n_layer"]
    n_ff     = 4 * n_embd
    ln_eps   = cfg.get("layer_norm_epsilon", 1e-5)

    print(f"      vocab={n_vocab} ctx={n_ctx} d={n_embd} heads={n_head} "
          f"layers={n_layer} d_ff={n_ff} ln_eps={ln_eps}")

    # 2. Open safetensors in mmap mode (no full copy in RAM).
    print(f"[2/4] opening {sft_path}")
    sft = safe_open(sft_path, framework="numpy")

    # Tensor-name prefix sniffing: distilgpt2 uses 'transformer.h.N.…' /
    # 'transformer.wte.weight', but the official OpenAI gpt2 release
    # stores them flat as 'h.N.…' / 'wte.weight'. Detect by probing for
    # one known key.
    all_names = sft.keys()
    if "transformer.wte.weight" in all_names:
        prefix = "transformer."
    elif "wte.weight" in all_names:
        prefix = ""
    else:
        raise RuntimeError("could not find wte.weight (with or without "
                           "'transformer.' prefix) in the safetensors")
    print(f"      tensor-name prefix detected: {prefix!r}")

    def take(name: str) -> np.ndarray:
        # Caller passes the "transformer.…"-style name (matches the
        # distilgpt2 convention used in this file's mapping below);
        # we strip 'transformer.' if the actual file is flat-named.
        if prefix == "" and name.startswith("transformer."):
            name = name[len("transformer."):]
        t = sft.get_tensor(name)
        return np.ascontiguousarray(t.astype(np.float32))

    # 3. Build GGUF.
    print(f"[3/4] writing GGUF → {args.out}")
    w = gguf.GGUFWriter(args.out, "gpt2")

    w.add_context_length(n_ctx)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_block_count(n_layer)
    w.add_head_count(n_head)
    w.add_layer_norm_eps(ln_eps)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_uint32("gpt2.vocab_size", n_vocab)   # gguf has no helper for this

    # Globals
    w.add_tensor("token_embd.weight",    take("transformer.wte.weight"))
    w.add_tensor("position_embd.weight", take("transformer.wpe.weight"))
    w.add_tensor("output_norm.weight",   take("transformer.ln_f.weight"))
    w.add_tensor("output_norm.bias",     take("transformer.ln_f.bias"))

    # Per-block
    for li in range(n_layer):
        prefix_hf  = f"transformer.h.{li}"
        prefix_out = f"blk.{li}"

        # LayerNorms
        w.add_tensor(f"{prefix_out}.attn_norm.weight", take(f"{prefix_hf}.ln_1.weight"))
        w.add_tensor(f"{prefix_out}.attn_norm.bias",   take(f"{prefix_hf}.ln_1.bias"))
        w.add_tensor(f"{prefix_out}.ffn_norm.weight",  take(f"{prefix_hf}.ln_2.weight"))
        w.add_tensor(f"{prefix_out}.ffn_norm.bias",    take(f"{prefix_hf}.ln_2.bias"))

        # Attention QKV. Split c_attn into separate Q/K/V; per-head split
        # is done at load time in Ruby (each Q/K/V is shape [d_model, d_model]).
        c_attn_w = take(f"{prefix_hf}.attn.c_attn.weight")   # [768, 2304]
        c_attn_b = take(f"{prefix_hf}.attn.c_attn.bias")     # [2304]
        wq, wk, wv = split_qkv(c_attn_w, n_embd)             # each [768, 768]
        bq, bk, bv = split_qkv(c_attn_b, n_embd)             # each [768]
        w.add_tensor(f"{prefix_out}.attn_q.weight", wq)
        w.add_tensor(f"{prefix_out}.attn_q.bias",   bq)
        w.add_tensor(f"{prefix_out}.attn_k.weight", wk)
        w.add_tensor(f"{prefix_out}.attn_k.bias",   bk)
        w.add_tensor(f"{prefix_out}.attn_v.weight", wv)
        w.add_tensor(f"{prefix_out}.attn_v.bias",   bv)

        # Output projection
        w.add_tensor(f"{prefix_out}.attn_output.weight", take(f"{prefix_hf}.attn.c_proj.weight"))
        w.add_tensor(f"{prefix_out}.attn_output.bias",   take(f"{prefix_hf}.attn.c_proj.bias"))

        # FFN
        w.add_tensor(f"{prefix_out}.ffn_up.weight",   take(f"{prefix_hf}.mlp.c_fc.weight"))
        w.add_tensor(f"{prefix_out}.ffn_up.bias",     take(f"{prefix_hf}.mlp.c_fc.bias"))
        w.add_tensor(f"{prefix_out}.ffn_down.weight", take(f"{prefix_hf}.mlp.c_proj.weight"))
        w.add_tensor(f"{prefix_out}.ffn_down.bias",   take(f"{prefix_hf}.mlp.c_proj.bias"))

    # 4. Flush.
    print(f"[4/4] finalising")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = os.path.getsize(args.out)
    print(f"done — {args.out} ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
