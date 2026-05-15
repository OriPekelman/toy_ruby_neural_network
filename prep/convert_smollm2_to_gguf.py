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
# Convert HF SmolLM2-135M (or any llama-family model with the same
# architecture: TinyLlama, SmolLM2-360M, Qwen2.5-0.5B with biases off)
# to a project-native GGUF in the layout `Toy::SmolLM2` expects.
#
# HF Llama layout (nn.Linear stores [out, in], so every projection
# needs a transpose to land in our [in, out] convention):
#
#   model.embed_tokens.weight                            [V, D]      (no transpose)
#   model.norm.weight                                    [D]
#   model.layers.N.input_layernorm.weight                [D]
#   model.layers.N.post_attention_layernorm.weight       [D]
#   model.layers.N.self_attn.q_proj.weight  [n_heads * Dh,    D]
#   model.layers.N.self_attn.k_proj.weight  [n_kv    * Dh,    D]
#   model.layers.N.self_attn.v_proj.weight  [n_kv    * Dh,    D]
#   model.layers.N.self_attn.o_proj.weight  [D, D]
#   model.layers.N.mlp.gate_proj.weight     [d_ff, D]
#   model.layers.N.mlp.up_proj.weight       [d_ff, D]
#   model.layers.N.mlp.down_proj.weight     [D, d_ff]
#
# Output GGUF tensor names follow llama.cpp convention:
#   token_embd.weight, output_norm.weight,
#   blk.N.{attn_norm, attn_q, attn_k, attn_v, attn_output,
#          ffn_norm, ffn_gate, ffn_up, ffn_down}.weight
#
# No tokenizer is embedded — tokenization happens host-side
# (prep/encode_smollm2.py) and IDs go through the Ruby model as
# integers. v1 simplification; the converter writes weights only.

import argparse
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import gguf


REPO_ID = "HuggingFaceTB/SmolLM2-135M"
OUT     = Path("data/smollm2-135m-f32.gguf")
CACHE   = Path("prep/_hf_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--out",     default=str(OUT))
    ap.add_argument("--cache",   default=str(CACHE))
    args = ap.parse_args()

    os.makedirs(args.cache,            exist_ok=True)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    print(f"[1/4] downloading {args.repo_id} → {args.cache}")
    cfg_path = hf_hub_download(args.repo_id, "config.json",       cache_dir=args.cache)
    sft_path = hf_hub_download(args.repo_id, "model.safetensors", cache_dir=args.cache)

    with open(cfg_path) as f:
        cfg = json.load(f)
    n_vocab     = cfg["vocab_size"]
    n_ctx       = cfg["max_position_embeddings"]
    n_embd      = cfg["hidden_size"]
    n_head      = cfg["num_attention_heads"]
    n_kv        = cfg["num_key_value_heads"]
    n_layer     = cfg["num_hidden_layers"]
    n_ff        = cfg["intermediate_size"]
    rope_theta  = float(cfg.get("rope_theta", 10000.0))
    rms_eps     = float(cfg.get("rms_norm_eps", 1e-5))
    d_head      = n_embd // n_head

    print(f"      vocab={n_vocab} ctx={n_ctx} d={n_embd} d_head={d_head} "
          f"heads={n_head} n_kv={n_kv} layers={n_layer} d_ff={n_ff} "
          f"rope_base={rope_theta} rms_eps={rms_eps}")

    # safe_open in numpy framework chokes on bfloat16 (SmolLM2's storage
    # dtype). Read raw bytes, then decode bf16 → f32 manually: bf16 is
    # literally the upper 16 bits of f32, so left-shift uint16 by 16
    # and reinterpret as f32.
    print(f"[2/4] opening {sft_path}")

    def _bf16_to_f32(buf: bytes, shape) -> np.ndarray:
        as_u16 = np.frombuffer(buf, dtype=np.uint16)
        as_u32 = as_u16.astype(np.uint32) << 16
        return as_u32.view(np.float32).reshape(shape)

    from safetensors import deserialize
    with open(sft_path, "rb") as fh:
        raw_payload = fh.read()
    blobs = dict(deserialize(raw_payload))

    def _load_f32(name: str) -> np.ndarray:
        info = blobs[name]
        dtype, shape, buf = info["dtype"], info["shape"], info["data"]
        if dtype == "BF16":
            arr = _bf16_to_f32(buf, shape)
        elif dtype in ("F32", "F16"):
            np_dt = {"F32": np.float32, "F16": np.float16}[dtype]
            arr = np.frombuffer(buf, dtype=np_dt).reshape(shape).astype(np.float32)
        else:
            raise RuntimeError(f"unsupported dtype: {dtype} for {name}")
        return np.ascontiguousarray(arr)

    def take(name: str) -> np.ndarray:
        return _load_f32(name)

    # nn.Linear convention: weight.shape == [out, in]. We need [in, out]
    # to match our Mat layout. Single tensor → single transpose.
    def take_T(name: str) -> np.ndarray:
        return np.ascontiguousarray(_load_f32(name).T)

    print(f"[3/4] writing GGUF → {args.out}")
    w = gguf.GGUFWriter(args.out, "llama")

    w.add_context_length(n_ctx)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_block_count(n_layer)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_rope_freq_base(rope_theta)
    w.add_rope_dimension_count(d_head)
    w.add_layer_norm_rms_eps(rms_eps)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_uint32("llama.vocab_size", n_vocab)

    # Globals
    w.add_tensor("token_embd.weight",  take("model.embed_tokens.weight"))   # [V, D]
    w.add_tensor("output_norm.weight", take("model.norm.weight"))           # [D]

    # Per-block
    for li in range(n_layer):
        hf  = f"model.layers.{li}"
        out = f"blk.{li}"

        # RMSNorms (1-D, no transpose)
        w.add_tensor(f"{out}.attn_norm.weight", take(f"{hf}.input_layernorm.weight"))
        w.add_tensor(f"{out}.ffn_norm.weight",  take(f"{hf}.post_attention_layernorm.weight"))

        # Attention projections (all transposed: HF stores [out, in])
        #   q_proj: HF [n_heads*Dh, D]  →  ours [D, n_heads*Dh]   (= [D, D] for SmolLM2)
        #   k_proj: HF [n_kv*Dh,   D]  →  ours [D, n_kv*Dh]
        #   v_proj: HF [n_kv*Dh,   D]  →  ours [D, n_kv*Dh]
        #   o_proj: HF [D, D]          →  ours [D, D]
        w.add_tensor(f"{out}.attn_q.weight",      take_T(f"{hf}.self_attn.q_proj.weight"))
        w.add_tensor(f"{out}.attn_k.weight",      take_T(f"{hf}.self_attn.k_proj.weight"))
        w.add_tensor(f"{out}.attn_v.weight",      take_T(f"{hf}.self_attn.v_proj.weight"))
        w.add_tensor(f"{out}.attn_output.weight", take_T(f"{hf}.self_attn.o_proj.weight"))

        # SwiGLU FFN (all transposed)
        #   gate_proj / up_proj: HF [d_ff, D]  →  ours [D, d_ff]
        #   down_proj:           HF [D, d_ff] →  ours [d_ff, D]
        w.add_tensor(f"{out}.ffn_gate.weight", take_T(f"{hf}.mlp.gate_proj.weight"))
        w.add_tensor(f"{out}.ffn_up.weight",   take_T(f"{hf}.mlp.up_proj.weight"))
        w.add_tensor(f"{out}.ffn_down.weight", take_T(f"{hf}.mlp.down_proj.weight"))

    print(f"[4/4] finalising")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = os.path.getsize(args.out)
    print(f"done — {args.out} ({sz / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
