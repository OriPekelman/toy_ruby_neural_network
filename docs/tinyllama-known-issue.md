# TinyLlama-1.1B FFI path produces NaN logits (open)

**Status**: open. Native Mat path works; FFI (CPU and CUDA) NaNs.

## Symptoms

`demos/tinyllama_kv_cuda` (and `demos/tinyllama_kv` on CPU FFI) load
TinyLlama-1.1B-Chat-v1.0 and execute a forward pass at ~64 ms/token
(CUDA) / ~160 ms/token (CPU). Output logits are all `NaN`. Greedy
argmax over NaN picks token 0 (the `<unk>` token), so generation
produces a string of `<unk>` for every output position.

`demos/tinyllama_native` (pure Ruby Mat path on the same model) at
~4.5 s/token produces sensible logits (e.g. argmax=13, value=11.33
on the last prompt position) and reasonable continuations.

The exact same FFI code path runs `demos/smollm2_kv` /
`demos/smollm2_kv_cuda` correctly (`"Once upon a time, there was a
little girl named Lily..."`), so the FFI ops themselves are fine —
something about TinyLlama's shape exercises a different bug.

## What's been ruled out

- **Not the untied-output wiring.** Forcing the tied path on
  TinyLlama (the FFI uses `t_token_embed` for unembed instead of
  `t_output`) still produces NaN — the bug is upstream of the
  unembed step.
- **Not a load failure.** `tnn_finalize_weights` returns 0 and
  4.4 GB of weights are uploaded successfully. `model.output_proj`
  (verified via debug print) holds the right values from the GGUF.
- **Not a weight-conversion failure.** The native Mat path uses
  the same loaded weights and produces sensible logits.
- **Not a Spinel compile issue.** No warnings/errors on the new
  code; the binary builds clean.
- **Not architecture-bound.** TinyLlama uses the same Toy::SmolLM2 /
  SmolLM2KVFFI(Cuda) code as SmolLM2-135M. Both follow the
  llama-family pattern with RMSNorm + GQA + SwiGLU + RoPE.

## Suspects, not yet confirmed

- **Graph order / `tnn_add_to_graph` for 4 KV writes × 22 layers**
  may interact differently with ggml's scheduler than 3 × 30 did.
  Hypothesis: the K/V writes don't complete before the per-head
  attention reads, so attention reads zero-K and zero-V → softmax
  on near-zero scores produces a uniform distribution → matmul
  with zero-V produces zeros → eventual NaN in a later RMSNorm.
  Would explain "first decode is NaN even at pos=0".
- **Buffer overflow** in the compute ctx for the larger TinyLlama
  graph (32 Q heads × 22 layers vs 9 × 30). Estimated graph node
  count is ~8200 (under the 16384 cap), but the per-layer leaf
  count and scratch usage may differ.

## Reproduction

```sh
# on gx10 (has the GGUF + CUDA toolkit)
./prep/llama_tokens.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --ids data/tinyllama_prompt_ids.txt encode "Once upon a time"
./demos/tinyllama_kv_cuda
# → output ends in <unk><unk><unk>... ; argmax always 0

./demos/tinyllama_native
# → "Once upon a time" + sensible continuation (slow: ~4.5 s/token)
```

## Next debugging step

Add an intermediate `tnn_set_output` + download after each layer's
output (or after K-cache write) in `build_block_step`. Walk
layer-by-layer to find the first NaN. That nails down which op
introduces it.

Until then: the TinyLlama path is committed but flagged as broken on
FFI. SmolLM2-135M remains the bench-of-record for the llama-family
FFI throughput numbers.
