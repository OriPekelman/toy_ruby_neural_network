# Toy transformer LM in Ruby (Spinel-compiled, optional CUDA)

A small decoder-only transformer language model written in Ruby and
compiled to a native binary by
[Spinel](https://github.com/matz/spinel) — matz's Ruby AOT compiler.
The point is **readability**: forward, backward, optimizer, and
training loop are meant to be readable top-to-bottom.

Trained from scratch on a 5K-line slice of TinyStories, the toy model
(d_model=32, 2 layers, 4 heads, ~30K params) produces recognizable
TinyStories-style continuations after ~30 epochs.

For real-LLM-scale workloads a single pure-Ruby matmul is 2.6 s/iter;
that's unviable. The `tinynn/` subdirectory adds an opt-in FFI bridge
to [ggml](https://github.com/ggml-org/ggml) (CPU or CUDA) that closes
the gap by 45× at LLM scale. See [`tinynn/README.md`](tinynn/README.md)
for the bridge design, op coverage, bench numbers, and CUDA story.

## Layout

```
.
├── lib/
│   ├── transformer.rb       Mat, Block, TransformerLM, Gradients, AdamState
│   ├── training.rb          LRSchedule, DataLoader, Adam, corpus readers
│   ├── tinynn.rb            FFI bridge (CPU): module TinyNN
│   ├── tinynn_cuda.rb       FFI bridge (CUDA): module TinyNNCuda
│   └── …
├── prep/                    CRuby-only — runs once, writes data/
├── data/                    Generated tokenized corpus (gitignored)
├── tinynn/                  C shim + ab_smoke + bench files (see tinynn/README.md)
├── train_tinystories.rb     Main training entrypoint (Spinel-compiled)
├── train_minimal.rb         ~85-line smoke test (Spinel-compiled)
├── Makefile                 build system + smoke targets
└── README.md                this file
```

## Architecture

```
token_ids ─▶ embed ─▶ [Block]×N ─▶ RMSNorm ─▶ unembed (tied) ─▶ logits

Block (pre-norm):
    x ─▶ RMSNorm ─▶ multi-head causal attention ─▶ + ─┐
                                                      │
                                  residual ◀──────────┘
    x'─▶ RMSNorm ─▶ FFN (Linear → GeLU → Linear) ─▶ + ─┐
                                                       │
                                  residual ◀───────────┘
```

Standard modern bits: pre-RMSNorm, multi-head attention with per-head
Q/K/V projections, causal masking, residual connections, learned
positional embeddings, GeLU FFN, **tied input/output embeddings**,
cross-entropy with the combined softmax+CE gradient, **linear warmup
+ cosine LR decay**, KV cache for incremental generation.

Both **plain SGD** and **Adam** (with bias correction) are implemented
in `lib/transformer.rb` as `apply_gradients_sgd` /
`apply_gradients_adam`. `train_minimal.rb` calls SGD directly for its
smoke-test loop; `train_tinystories.rb` uses the `Adam` wrapper in
`lib/training.rb`, which owns the m/v moment state and steps the
model.

## Usage

```sh
# 1. Prep — download, tokenize, chunk into context windows. CRuby.
ruby prep/prep_tinystories.rb --max_lines 5000 --context_length 64 \
                              --prompt "Once upon a time"

# 2. Build vendored ggml as a static archive (one-time, ~30 s).
make setup-ggml             # CPU only
make setup-ggml-cuda        # adds CUDA (defaults to sm_121 / GB10)

# 3. Build train_tinystories. Default is native (pure-Ruby Mat#matmul).
make train_tinystories
./train_tinystories

# Smoke test (build + 40 SGD steps, <30 ms):
make train_minimal && ./train_minimal
```

To enable the FFI bridge, flip `USE_FFI_MATMUL = true` at the top of
`lib/transformer.rb`. The FFN's two matmuls then dispatch through
`TinyNN.matmul` (ggml-CPU) instead of `Mat#matmul`. See
[`tinynn/README.md`](tinynn/README.md) for the CUDA flow.

## Spinel constraints (and why some idioms look unusual)

Spinel does whole-program type inference; the entire reachable Ruby
has to type-check against a single closed world. Several patterns are
workarounds for this; the load-bearing ones are listed at the top of
`lib/transformer.rb` and include:

- **Flat 1D `Float` storage** in `Mat` (Spinel can't infer
  `Array<Array<Float>>` from `Array.new(n) { Array.new(m, 0.0) }`).
- **Classes, not hashes**, for `Block` / `LayerCache` / `KV cache`
  (heterogeneous-valued hashes confuse Spinel's codegen).
- **`nrows` / `ncols`** field names (not `rows` / `cols` — the latter
  hits a name-collision in iterative inference).
- **Result wrappers** (`NormResult`, `FFResult`, `LossResult`, …) where
  one would normally use a multi-value return.
- **Warm-up call** in `train_tinystories.rb` to anchor class-method
  param inference from a top-level call site.

## What's been merged upstream

- [matz/spinel#258](https://github.com/matz/spinel/pull/258) —
  `fix(codegen): root PtrArray temp in array-of-objects literal`. Found
  while debugging a SIGSEGV in `Block#initialize`.
- [matz/spinel#473](https://github.com/matz/spinel/issues/473) —
  closed 2026-05-14: `fix(codegen): clear ptr ivars after SP_POOL_NEW
  before init body runs`. Restored end-to-end training after the
  per-class free-list pool regression in 4a7a678.
- [matz/spinel#474](https://github.com/matz/spinel/issues/474) —
  closed 2026-05-14: `feat(ffi): :float_array / :int_array specs
  for zero-copy bulk transfer`. The FFI bridge's bulk-upload primitive.

Still open:

- [ggml-org/ggml#1491](https://github.com/ggml-org/ggml/issues/1491) —
  `ggml_rms_norm_back` produces different output via
  `ggml_backend_sched_graph_compute` than via legacy
  `ggml_graph_compute_with_ctx`. Blocks `TinyNN.rms_norm_back`
  parity; not blocking training (project's native RMSNorm backward
  runs fine).

## Status

Educational, with optional acceleration for real-LM-scale work.

- Loss on TinyStories converges from ~5.3 (epoch 1) to ~3 (epoch 30)
  with the upgraded stack.
- Generations look plausibly TinyStories-shaped:
  *"once upon a time there was a little boy named tim he loved to
  play in the park with his best friends…"*
- FFI bridge: 22 ggml ops + chained FFN graph + `FullForwardFFICache`
  (entire model forward as one persistent graph) + ggml-native AdamW
  + KV-cache primitives (view + cpy) + LayerNorm verified on both CPU
  and CUDA. At "more than toy" shape (vocab=4096, d_model=384, n_heads=6,
  n_layers=6, T=128) the full forward runs **38× faster on CPU FFI**
  (1180 ms → 31 ms/iter) and **34× faster on CUDA FFI**. End-to-end
  inference works: `./inference_demo` generates greedy-sampled tokens
  via the FFI graph, parity-checked against native forward.
- HTTP API POC: `make tep_demo/api && ./tep_demo/api` builds a
  single-binary Tep+Spinel server that runs greedy generation
  through `FullForwardFFICache`. **2,738 req/s = ~13.7k tokens/sec**
  at 16-thread concurrency on this toy model (vocab=16, d_model=32,
  n_layers=2). 1.46 ms per `/generate?n=5` server-side. See
  [tep_demo/README.md](tep_demo/README.md) for full numbers and the
  `make tep_demo/hello` minimal variant.
- Full performance story and the path to fully-on-GPU training in
  [tinynn/README.md](tinynn/README.md). HTTP-server numbers,
  build details, and the inference-API blocker in
  [tep_demo/README.md](tep_demo/README.md).

Real LM training at this scale still wants careful hyperparameter
sweeps; this is a hand-built toy.
