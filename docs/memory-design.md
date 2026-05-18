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

The PyTorch/llama.cpp model. CPU-side, ggml exposes
`ggml_backend_cpu_buffer_from_ptr(void *ptr, size_t size)` — wrap an
mmap'd region as a backend buffer; tensors created within point at
the mapped pages directly. No copy.

**CUDA correction (2026-05-18):** ggml-cuda does NOT have an
equivalent BYO-pointer API. The closest is
`ggml_backend_cuda_register_host_buffer`, which just calls
`cudaHostRegister` to pin host pages so cudaMemcpy can DMA without a
bounce buffer — the device-side buffer is still a separate
allocation. GB10's unified memory means that device buffer is cheap
to allocate (via ggml-cuda's `cudaMallocManaged`), but it's still a
distinct allocation; the load-time host→device copy is unavoidable
through the public ggml API. True zero-copy on GB10 requires
patching ggml-cuda upstream.

**Transpose finding (2026-05-18):** Our converter explicitly
transposes 2D linear weights from HF's `[out, in]` to `[in, out]`
during write — a legacy of the Mat-side `[in, out]` convention. That
transpose is the only thing forcing the load-time fixup. Skipping
it (the converter's new `--ggml-native` flag) writes bytes that
match ggml's column-major `ne=[in, out]` layout directly, so the
loader can `memcpy` (or, eventually, `mmap`) without touching the
bytes. Per-head Q/K/V slices become contiguous byte ranges in the
HF-native layout — perfect for mmap.

Caveats:
- f32 GGUFs: zero-copy on CPU once mmap is wired.
- Q8_0 GGUFs: today's loader dequantizes at load (Q8 → f32) into the
  persistent buffer. To realize the 9 GB-vs-30 GB savings for 7B-Q8,
  the persistent ggml tensor must be allocated with
  `GGML_TYPE_Q8_0` directly (not f32); ggml's matmul auto-dispatches
  to Q8 kernels for mixed activation-f32 × weight-Q8. That's Phase 3
  below.
- mmap'd file must stay open for the session's lifetime.

## Recommendation

**Land (A) first.** Unlocks 7B at ~30 GB on gx10. Implementation is
straightforward (one C function + one Ruby loader method).

**(C) is the optimal long-term target** for unified-memory hosts.
Maps cleanly onto how llama.cpp loads models today; the toy code
just needs to expose it through Ruby.

**(B) is orthogonal** — it benefits training (where the Mat IS the
hot data) but isn't critical for inference once (A) lands.

## Status (2026-05-17)

(A) shipped in step 46. `tinynn/tinynn_gguf.c` gains five
direct-loader primitives (`tnn_gguf_copy_*_to_persistent`) and
`lib/toy_smollm2_loader.rb` adds `GGUFLoad.load_kv_cache_directly`.
The Ruby Mat is never allocated for inference.

Verified Qwen2.5-7B end-to-end:
- **Peak RAM during decode**: 30 GB (matches the 4 B/w prediction
  for 7.6B params).
- **Decode**: 1062 ms/token on gx10 CPU.
- **Output**: "Hello, my name is a 19-year-old male. I have been
  having a problem with my".
- Direct loader produces bit-identical first-token argmax to the
  Mat-mediated path on 0.5B and 1.5B (verified by sharing prompt
  IDs across runs and comparing `step 0 top index/val`).

## Status (2026-05-18)

Phase 1 of (C) — converter-side groundwork shipped. The
`--ggml-native` flag on `prep/convert_smollm2_to_gguf.py` writes 2D
linear weights without the legacy `[out, in] → [in, out]` transpose,
so the bytes match ggml's column-major `ne=[in, out]` layout
directly. Loader gains `GGUFLoad.load_kv_cache_directly_native`
(memcpy, no chunked staging) and `GGUFLoad.load_kv_cache_auto`
(picks via the `toy.ggml_native` metadata key). `SmolLM2KVFFICache#load_weights`
now uses the auto-dispatcher, so callers stay layout-agnostic.

Parity check on Qwen2.5-0.5B: native loader produces the SAME
first-step logit value (`top index=264 val=12.77385425567627`) and
the SAME 8-token greedy continuation as the legacy loader. Load
time drops 37% (4860 → 3071 ms) because the chunked transpose stage
is gone.

Phase 2 (real mmap) and Phase 3 (Q8-stays-Q8) are the remaining
pieces.

## Current data point

Qwen2.5-1.5B (1.54B params) at today's 12 B/w:
- Mat: 12 GB
- FFI buffer: 6 GB
- Peak RAM during decode: ~18 GB
- Decode speed: 220 ms/token on gx10 CPU

This fits comfortably alongside the training run (57 GB used, 63 GB
free). 3B will fit (~37 GB). 7B will not without (A).
