# M3: DistilGPT2 inference through Ruby → Spinel → ggml-FFI

End-to-end inference of a real HuggingFace model — DistilGPT2 (82 M
params, 6 layers, d=768, vocab=50257) — running through three
backends with verified numerical parity against PyTorch transformers.

Branch: `hf-gpt2` (12 commits beyond `main`).

```
Generate "Hello, my name is" → "Hello, my name is J.J.K. Rowling."

Native Mat (f64, single-thread Ruby):  ~9.0 s   (~1100 ms/token)
FFI full-forward (T_SEQ=16):            451 ms  (   56 ms/token)
FFI KV cache:                            94 ms  (    7.4 ms/token)
```

## Quick start

One-time setup (downloads ~700 MB across distilgpt2 weights + Python
deps; converts to ~330 MB GGUF):

```sh
make setup-ggml                        # CPU ggml build (~30 s)
./prep/convert_distilgpt2_to_gguf.py   # HF → data/distilgpt2-f32.gguf
```

Run any of the three demos:

```sh
./prep/tokens.py encode "Hello, my name is"

make distilgpt2_demo      && ./distilgpt2_demo       # native (slow)
make distilgpt2_demo_ffi  && ./distilgpt2_demo_ffi   # full-forward FFI
make distilgpt2_demo_kv   && ./distilgpt2_demo_kv    # KV-cache FFI

./prep/tokens.py decode   # any of the demos write generated IDs back
```

## Parity vs HF transformers

Reference produced by `prep/parity.py ref` (loads distilgpt2 via
PyTorch + transformers, dumps the last-row logits for the same
5-token prompt). All three Ruby paths match the reference at F32
round-trip precision:

| Path | max-abs diff | mean-abs diff | argmax | top-5 |
|---|---:|---:|---:|---:|
| Native Mat (f64) | `1.16e-4` | `1.93e-5` | ✓ | 5/5 |
| FFI full-forward (f32) | `5.42e-3` | `3.65e-3` | ✓ | 5/5 |
| FFI KV decode (f32) | `3.06e-3` | `1.36e-3` | ✓ | 5/5 |

(KV's max-abs is actually *better* than full-forward's because
per-position writes accumulate less F32 error than a full-sequence
matmul.)

Re-run: `make gpt2-parity` (native dump), `make gpt2-ffi-parity`
(full-forward dump), `make gpt2-kv-parity` (KV dump), then
`./prep/parity.py compare --ours data/ours_*_logits.txt`.

## Bench (M2 Air, single CPU worker)

`make gpt2-bench`, prompt = "Hello, my name is" (5 tokens):

| | ms/forward | forwards/sec | speedup vs native |
|---|---:|---:|---:|
| Native Mat, f64 | 1289.6 | 0.78 | 1× |
| FFI full-forward, T_SEQ=5 | 20.2 | 49.4 | 63.7× |
| FFI KV decode, pos=5..34 | 22.7 | 44.1 | 56.9× |

The bench at T_SEQ=5 doesn't fully show the KV win: both FFI paths
are dominated by graph-build overhead at this scale. As T grows,
full-forward attention is O(T²), KV per-step is O(pos):

| | T_SEQ=5 | T_SEQ=16 (from demo) |
|---|---:|---:|
| FFI full-forward | 20 ms / forward | 56 ms / forward |
| FFI KV decode | 22 ms / step (mean) | 7.4 ms / step |

At T_SEQ=16 KV is ~8× faster; the gap widens further as T grows.

## File map

```
├── prep/
│   ├── convert_distilgpt2_to_gguf.py  HF safetensors → data/*.gguf
│   ├── tokens.py                       BPE encode / decode (host-side)
│   └── parity.py                       HF reference logits + compare
│
├── lib/
│   ├── gpt2.rb           GPT2LM + GPT2Block (native, inference-only)
│   ├── gpt2_ffi.rb       GPT2FullForwardFFICache (full-forward FFI)
│   ├── gpt2_ffi_kv.rb    GPT2KVFFICache (per-step KV-cache FFI)
│   └── gguf_load.rb      Read GGUF tensors into a GPT2LM
│
├── tinynn/
│   ├── gguf_inspect.rb       List every tensor in a GGUF via FFI
│   ├── gpt2_build_smoke.rb   Toy-shape forward smoke
│   ├── gpt2_load_smoke.rb    Sentinel-value loader check
│   ├── gpt2_parity.rb        Native parity dump
│   ├── gpt2_ffi_parity.rb    Full-forward parity dump
│   ├── gpt2_kv_parity.rb     KV-cache parity dump
│   ├── kv_multi_cpy_smoke.rb 8-position single-head KV smoke
│   └── gpt2_bench.rb         Native vs FFI vs KV bench
│
├── distilgpt2_demo.rb       End-to-end generation (native)
├── distilgpt2_demo_ffi.rb   End-to-end generation (full-forward FFI)
└── distilgpt2_demo_kv.rb    End-to-end generation (KV-cache FFI)
```

## Architecture notes

### GPT2LM (lib/gpt2.rb)

Separate from `TransformerLM` because the existing class is heavy
with training scaffolding (Gradients, AdamState, layer caches, FFI
caches). For inference-only the simpler class is:

- learned token + absolute position embeddings, additive
- pre-LayerNorm with `gamma + beta` on every norm
- multi-head causal attention with `bias` on every Linear (q/k/v/o)
- GeLU FFN (the tanh approx, `gelu_new`) with bias on both Linears
- tied output embedding (`logits = x_final @ token_embedᵀ`)
- sequential residual

Per-head Q/K/V layout matches the existing `Block` so the FFI cache
can mirror what `FullForwardFFICache` already does.

### GGUF format & loader

Converter (`prep/convert_distilgpt2_to_gguf.py`) writes a llama.cpp-
compatible GGUF with naming `blk.N.{attn_norm,ffn_norm,attn_q,
attn_k,attn_v,attn_output,ffn_up,ffn_down}.{weight,bias}` plus
`token_embd.weight`, `position_embd.weight`, `output_norm.{weight,
bias}`. `c_attn` is pre-split into Q/K/V at conversion time so the
loader's per-head reshape is just a strided slice.

Loader (`lib/gguf_load.rb`) reads straight into the destination
`Mat.flat` / `Array<Float>` via `tnn_gguf_read_f32_to_doubles`
— no intermediate buffers, even for the 38.6 M-element `token_embd`.

### Full-forward FFI graph (lib/gpt2_ffi.rb)

`GPT2FullForwardFFICache` is the same shape as the existing
`FullForwardFFICache` (RMSNorm + no-bias), with two additions:

1. `tnn_layer_norm(x, gamma, beta, eps)` instead of `tnn_rms_norm`
2. Bias-adds after every Linear. ggml's `ggml_add` broadcasts via
   `ggml_can_repeat`, so the 1-D biases need no extra op — except
   the V bias must be shape `ne=[1, d_head]` (not `ne=[d_head, 1]`
   like Q/K) because the transposed-V matmul layout produces
   `ne=[T, d_head]` and the can_repeat rule cares about dim order.

Persistent T_SEQ at realize time; per-step pads input to T_SEQ.

### KV-cache decode (lib/gpt2_ffi_kv.rb)

`GPT2KVFFICache` extends the above with persistent K[block][head]
(d_head × max_T) and V[block][head] (max_T × d_head) buffers.

Per decode step at position `pos`:

1. `tnn_reset_for_rebuild(sess)` — clear compute graph, keep weights
2. Build one-position graph:
   - `get_rows(token_embed, [token_id])` + view of `pos_embed[pos]`
   - per block: LayerNorm, per-head Q/K/V matmuls, `cpy(k_new,
     view_2d(K, ..., pos))` + same for V, `matmul(K[0:pos+1], q)`,
     scale + softmax, `matmul(V[0:pos+1], attn)`, concat, out_proj,
     residual, LayerNorm, FFN, residual
   - final LayerNorm, tied unembed
3. `tnn_realize(t_logits)`, upload `token_id`, compute, download

Two primitives needed beyond the existing M2 work:

- **`tnn_add_to_graph(sess, tensor)`** (new) — exposes
  `ggml_build_forward_expand` so the `cpy(src, view_of_K)` ops are
  added to the graph even though their 1-row results aren't
  reachable from the realize target. Without this they'd be pruned
  and the K/V buffer never updates.
- **`cpy + view_2d` instead of `set_rows`/`set_2d`** — the pre-
  existing `ab_smoke_kv_attn_multi.rb` was a WIP that failed parity
  at pos≥1 because `ggml_set_2d`/`set_rows` use `ggml_dup_tensor`
  (returns a copy); the persistent K/V is never mutated.
  `ggml_cpy(src, view_of_K)` uses `ggml_view_tensor` on the
  destination, so the write lands in K's buffer. Multi-step
  attention then sees the accumulated history. Validated in
  `tinynn/kv_multi_cpy_smoke.rb` (8 positions, all pass).

Also: bumped `ctx_buf_size` in `tnn_session_new` from 8192 → 262144
tensor-header slots. The KV decode loop accumulates ~1280 compute-
graph node headers per step (6 layers × 12 heads × ~16 ops per
head + FFN/LN); old budget overflowed after ~30 rebuilds.

### Spinel name-collapse traps (recurring theme)

Spinel does whole-program type inference and unifies any name
(local var / method param / ivar) that appears with different
concrete types across the codebase. The same name in two scopes
becomes one polymorphic slot — and that boxing breaks `(void *)`
casts at FFI boundaries.

Renames needed during this work:

| was | became | reason |
|---|---|---|
| `m = GPT2LM.new(...)` (smoke) | `model = ...` | `m` was Mat in `softmax_rows!(m)` |
| `cache` in GPT2FFI / GPT2KV | `fwd_cache` / `kv_cache` | one was GPT2FullForwardFFICache, other GPT2KVFFICache |
| `tensor` param in `download_row_major` | `dl_handle` | dead `upload_transposed` mistyped its `tensor` param as mrb_int |
| `t_logits` ivar on GPT2KVStepResult | `kv_step_logits` | collided with FullForwardFFICache / GPT2FullForwardFFICache |
| `t` (Time.now) in bench | `iter_start` | `t = m.nrows` in transformer.rb |
| `null = TinyNN.tnn_null_ptr; [null]` | `[TinyNN.tnn_null_ptr]` | Spinel loses :ptr typing through local-var binding |

Diagnostic: errors like `'mrb_int' and 'sp_<OtherClass> *' in binary
expression` or `operand of type 'sp_RbVal' where arithmetic or
pointer type is required`. See
`~/.claude/projects/.../memory/feedback_spinel_var_name_collapse.md`
for the long-form note.

## Caveats / what's not in scope yet

- **Quantized GGUF unused.** Converter produces F32 (327.7 MB); the
  `tnn_gguf_read_f32_to_doubles` path dequantizes Q4_K/Q8_0/F16 via
  ggml's `to_float` traits but hasn't been exercised with a real
  model file. ~4× smaller distilgpt2 + similar arithmetic perf is
  on the table.

- **Reading hyperparams from GGUF metadata.** Caller currently
  hardcodes vocab/d_model/n_heads/etc to match what the converter
  wrote. Adding `tnn_gguf_kv_get_uint32` would close this.

- **BPE tokenizer is external.** `prep/tokens.py` is a Python
  shim (HF `tokenizers` package, rust-backed). A Ruby BPE encoder
  would let the Spinel binary be self-contained. ~200 lines of
  Ruby + a few Spinel name-collapse traps to fight.

- **No CUDA path yet for GPT-2.** `lib/tinynn_cuda.rb` already
  mirrors `lib/tinynn.rb` for the existing TransformerLM; the
  parallel `lib/gpt2_ffi_cuda.rb` is the natural next step.

- **Larger GPT-2 variants untested.** Same converter handles
  `--repo-id gpt2[-medium|-large|-xl]`. Memory cost scales linearly
  in params; gpt2-medium is ~700 MB F32 (RAM-tight on Mac at
  current disk pressure).

- **Context window is 1024** (DistilGPT2 / GPT-2 base). The
  KV-cache `max_T` parameter caps allocated buffer size; bench
  uses 32, demo uses 32. Larger contexts work but allocate more
  K/V memory per (block, head).

## Next: CUDA on gx10

`lib/tinynn_cuda.rb` and the existing `*_cuda` smokes establish the
pattern. Plan:

1. `make setup-ggml-cuda` on gx10 (already builds against ggml's
   CUDA backend with the Mac-build OpenMP/Accelerate flags
   harmless on Linux).
2. New `lib/gpt2_ffi_cuda.rb` mirroring `gpt2_ffi.rb` against the
   `TinyNNCuda` module (same op surface; differences are limited
   to backend selection and the CUDA-specific link flags).
3. Bench against the M2 Air FFI-CPU numbers and against PyTorch on
   the same hardware.

Existing CUDA mirrors (`ab_smoke_cuda`, `ab_smoke_all_cuda`,
`persistent_bench_cuda`) prove ggml-CUDA composes with the project's
FFI primitives. Mostly mechanical from here.
