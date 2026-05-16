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

## Layer-count bisection — non-monotonic NaN pattern

Forcing `cfg.n_layers = N` in the demo before `realize_for` and the
model construction (so only N out of 22 layers are loaded and run):

| n_layers | logits[0..4] |
|---:|---|
| 1  | finite (0.22, -0.15, ...) |
| 2  | finite |
| 3  | finite |
| 4  | finite |
| 5  | **NaN** |
| 6–14 | finite |
| 15 | **NaN** |
| 20, 22 | **NaN** |

L=5 reproduces deterministically across multiple runs. L=10 is
consistently fine, even though it includes layers 0–4 (the same ones
that NaN in L=5). So this is *not* "layer 4 has a bad weight" — the
NaN depends on the total layer count, not which specific layers
participate.

That points away from a math/weights cause and toward a Spinel codegen
or ggml-graph-state interaction that varies with the size of the
`@kv_blocks_ffi` array or the total persistent-buffer layout. Possible
mechanisms:

* Spinel arrays with N elements may use different internal storage
  for some N (inline vs heap) and one path silently mis-aliases.
* `ggml_backend_alloc_ctx_tensors` lays out persistent tensors
  differently at different total sizes; some layouts may put
  adjacent tensors in a pattern that ggml's compute kernels
  mis-handle.
* Compute-graph buffer / scratch sizing crosses some threshold at
  specific node counts.

## Next debugging step (when revisited)

1. Verify by setting `cfg.n_layers = 5` *but* skip the actual ggml
   compute for layer 4 (return identity). If NaN goes away, layer
   4's compute IS the trigger, but only at that total count.
2. Print `model.stack[i].rn1.gamma[0]` and other selected values for
   each layer at L=5 vs L=10 to confirm same weights were loaded.
3. Add an intermediate `tnn_set_output` after layer 0, layer 2, etc.
   and compare downloaded activations across runs.
4. Check `model.norm.weight` (gamma=3.188 max value) doesn't compound
   abnormally in the FFI graph.

Until then: SmolLM2-135M remains the bench-of-record for the
llama-family FFI throughput numbers.
