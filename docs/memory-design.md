# Inference memory: avoiding the 2× duplication

## The problem

Today's inference path pays 12 bytes per weight: 8 in the Ruby `Mat`
(Float64) plus 4 in the FFI persistent buffer (f32):

```
GGUF f32 on disk (4 B/w)
        ↓ tnn_gguf_read_f32_to_doubles   (f32→f64 widen)
   Ruby Mat (8 B/w, Float64)            ← native forward uses this
        ↓ stage_transposed_and_upload   (f64→f32 narrow + transpose)
   FFI buffer (4 B/w, f32)              ← FFI / CUDA forward uses this
```

For a 7B-parameter model: **60 GB Mat + 30 GB FFI = 90 GB** before we
even touch activations. gx10's 121 GB minus the training run leaves
~60 GB available — Qwen2.5-7B does not fit at this profile.

The Mat exists for the *native* (pure Ruby) forward path. For
inference via FFI we never touch the Mat in the hot loop — it's
allocated, populated from GGUF, transposed-uploaded once, then sits
idle for the entire decode loop.

## How PyTorch does it

PyTorch's model weights ARE the device buffer:

- `safetensors` opens a file with `mmap`. Tensors point straight at
  the mapped pages. The "loaded model" is the file's page cache;
  zero copy.
- On unified-memory backends (Apple MPS, NVIDIA GB10) the device
  buffer and the host pointer reference the same physical bytes —
  the GPU sees the mmap'd weights directly. No host↔device copy.
- Weights stay at `bf16` (2 B/w) for inference. A 7B model is 14 GB
  on disk and 14 GB in memory.

The fundamental difference from us: **PyTorch tensors are the
framework's native data type**. We have two: Ruby Mat and ggml
tensor, with a copy between them.

## Three paths forward

| Approach | Bytes/weight | 7B total | Effort |
|---|---|---|---|
| **Today** | 12 (Mat f64 + FFI f32) | 90 GB | — |
| **(A)** Skip Mat for inference: GGUF → FFI directly | 4 | 30 GB | Medium |
| **(B)** Mat backed by `:float_array` (f32) + (A) | 4 | 30 GB | Larger (Mat-wide refactor) |
| **(C)** mmap GGUF; FFI tensor.data → mapped page | 0 extra RAM | 30 GB shared with file cache | Hardest |

### (A) — direct GGUF→FFI loader

Add a C function that walks each GGUF tensor and copies its f32 bytes
into the corresponding FFI persistent buffer. Skip the Ruby Mat
allocation entirely for inference-only paths.

```c
int tnn_load_persistent_from_gguf(void *gguf_handle, const char *name,
                                   void *target_tensor, int transpose);
```

When `transpose` is set, do the chunked transposed write (same as
`tnn_upload_transposed_f64` but reading from GGUF rather than a
`double *`). Otherwise just `memcpy` (1-D γ, 1-D biases).

Ruby-side: `Toy::SmolLM2Loader.load_for_inference(kv_cache, gguf_path)`
fills the FFI cache directly. The `Toy::SmolLM2` Ruby object is
never constructed (or built with 1×1 placeholder Mats just to satisfy
the type system).

Memory: **4 B/w**. Matches PyTorch + bf16.

### (B) — Mat-as-float-array

The deeper refactor: replace `Mat.flat = Array.new(n, 0.0)` with a
packed `:float_array` (f32 backed). Halves Mat memory. Native
forward becomes f32-precise — fine for inference, possibly an
accuracy issue for training (TBD).

Requires touching every Mat operation. Larger blast radius.

### (C) — mmap GGUF into ggml tensors

The PyTorch/llama.cpp model. On gx10 (unified memory) the mmap'd
file pages ARE the GPU buffer — zero copy in the literal sense.
ggml has primitives for this (`ggml_backend_alloc_ctx_tensors_from_buft`
with a mapped buffer); llama.cpp uses them.

For us: when we know the GGUF is f32 (no dequant needed), we could
mmap the file region for each tensor and pin it as the ggml tensor's
`data` pointer. The persistent context's buffer becomes the file's
mapped pages.

Caveats:
- Only works for f32 GGUFs. Quantized GGUFs need dequant on load
  (which costs us a buffer copy anyway).
- The file must stay open + mapped for the lifetime of the session.
- Unified memory architectures (MPS, GB10) benefit fully — no extra
  copy. Discrete GPUs (most NVIDIA cards) still pay the host→device
  copy.

## Recommendation

**Land (A) first.** Unlocks 7B at ~30 GB on gx10. Implementation is
straightforward (one C function + one Ruby loader method).

**(C) is the optimal long-term target** for unified-memory hosts.
Maps cleanly onto how llama.cpp loads models today; the toy code
just needs to expose it through Ruby.

**(B) is orthogonal** — it benefits training (where the Mat IS the
hot data) but isn't critical for inference once (A) lands.

## Current data point

Qwen2.5-1.5B (1.54B params) at today's 12 B/w:
- Mat: 12 GB
- FFI buffer: 6 GB
- Peak RAM during decode: ~18 GB
- Decode speed: 220 ms/token on gx10 CPU

This fits comfortably alongside the training run (57 GB used, 63 GB
free). 3B will fit (~37 GB). 7B will not without (A).
