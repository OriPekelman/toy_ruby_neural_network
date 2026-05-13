# tinynn: Mat ↔ ggml FFI bridge

A thin layer that lets the pure-Ruby `Mat`/`TransformerLM` from
`lib/transformer.rb` dispatch its hot matrix operations through
[ggml](https://github.com/ggml-org/ggml) — CPU or CUDA — via Spinel's
FFI. The educational toy in the parent directory stays pure Ruby; this
directory is opt-in acceleration for when the toy's shapes outgrow what
interpreted Ruby can keep up with.

## Why this exists

At the toy model's training-step shapes, a single `Mat#matmul` takes
~0.2 ms in compiled Spinel. Acceptable. At real-LLM scale (the
tied-unembed `(64, 512) · (512, 32000)` shape that any sensible LM
needs), the same `Mat#matmul` takes **2.6 seconds per iteration** —
training is unviable. Through this bridge, the same matmul runs in
~57 ms (45× faster), which is the difference between "can we train
this" and "no". See `persistent_bench_big.rb`.

## Layout

```
tinynn/
├── tinynn_ggml.{h,c}         C shim — opaque-handle wrapper over ggml
├── smoke.rb                  4×3 raw matmul demo (no project deps)
├── ab_smoke*.rb              Per-op parity tests (TinyNN.<op> vs Mat#<op>)
├── ab_smoke_big.rb           Transformer-shape parity + wallclock (CPU)
├── ab_smoke_big_cuda.rb      Same on CUDA
├── ab_smoke_all_cuda.rb      Consolidated CUDA parity driver
├── forward_smoke.rb          A/B against a real TransformerLM
├── persistent_bench*.rb      Persistent-session bench at toy and LLM shapes
└── README.md                 This file

lib/
├── tinynn.rb                 CPU bridge: module TinyNN
└── tinynn_cuda.rb            CUDA bridge: module TinyNNCuda
```

## Build

The Makefile in the project root drives everything. CUDA is opt-in.

```sh
# One-time: clone + build vendored ggml as a static archive.
make setup-ggml             # CPU only — ~30 s on a recent host
make setup-ggml-cuda        # adds CUDA (defaults to sm_121 / GB10)

# Compile + run all CPU parity smokes:
make test

# Individual smokes:
make smoke                  # 4×3 raw matmul demo
make ab-smoke               # matmul parity
make ab-smoke-pipeline      # gelu(h·w1)·w2 chained
make ab-smoke-big           # transformer-shape parity + wallclock
make forward-smoke          # A/B against a real TransformerLM
make persistent-bench       # 50 iters, one-shot vs persistent
make persistent-bench-big   # same at real-LLM scale

# CUDA (requires `make setup-ggml-cuda`):
make ab-smoke-cuda
make ab-smoke-all-cuda      # consolidated all-ops CUDA driver
make persistent-bench-cuda
```

To override the CUDA arch (default 121 for GB10), pass
`GGML_CUDA_ARCH=NN` on the command line; for a non-standard CUDA
install, override `CUDA_DIR=/opt/cuda` etc.

## API surface

### One-shot wrappers

The "convenience" path. Each call spins up a fresh ggml graph, runs the
op, tears down. Good for ad-hoc tests and one-off operations; bad for
hot loops (per-call graph alloc dominates).

```ruby
require_relative "lib/transformer"
require_relative "lib/tinynn"

a = Mat.new(2, 3); a.flat[0] = 1.0; ...
b = Mat.new(3, 2); ...
c = TinyNN.matmul(a, b)          # A · B
d = TinyNN.matmul_t(a, b)        # A · B^T (ggml-native)
e = TinyNN.t_matmul(a, b)        # A^T · B
sum  = TinyNN.add(a, a)
gelu = TinyNN.gelu(a)
norm = TinyNN.rms_norm(a, [1.0]*3, 1e-5)
sm   = TinyNN.softmax(a)
sc   = TinyNN.scale(a, 0.5)

rows  = TinyNN.embed_lookup(table, [2, 0, 4, 2])
grads = TinyNN.embed_back(d_out, [2, 0, 4, 2], 5)
dx    = TinyNN.softmax_back(sm, dy)
new_p = TinyNN.sgd_step(param, grad, 0.05)
```

For CUDA, swap `TinyNN.` for `TinyNNCuda.` and add
`require_relative "lib/tinynn_cuda"` instead. Same shapes, same
semantics.

### Persistent-session API

The production-shape path. Build a graph once, run it many times. This
is what real training uses. Per-iteration cost drops by 30–70× on CPU.

```ruby
sess = TinyNN.persistent_new(0)               # 0 = CPU, 1 = CUDA-if-built
ta   = TinyNN.alloc_2d(sess, T, D)            # input slot
tw   = TinyNN.alloc_2d(sess, D_out, D)        # weight slot (uploaded transposed)
tc   = TinyNN.build_matmul(sess, ta, tw)      # the op tree
TinyNN.realize(sess, tc)                       # allocates all backend buffers

TinyNN.upload_transposed(sess, tw, w_mat)      # upload weights ONCE
loop do
  TinyNN.upload_row_major(sess, ta, input_mat)
  TinyNN.compute(sess)
  output = TinyNN.download_matmul(sess, tc, T, D_out)
end

TinyNN.persistent_free(sess)
```

Mixing `build_*` calls before `realize` builds a multi-op graph that
runs as one ggml dispatch — see `ab_smoke_pipeline.rb` for the FFN
shape (`matmul → gelu → matmul`).

## Op coverage

| Op | CPU (TinyNN) | CUDA (TinyNNCuda) | One-shot smoke | Notes |
|---|---|---|---|---|
| `matmul(a, b)` (A·B)         | ✅ | ✅ | `ab-smoke` | Internally A·Bᵀ via b-transposed upload |
| `matmul_t(a, b)` (A·Bᵀ)      | ✅ | ✅ | `ab-smoke-matmul-variants` | Native to ggml |
| `t_matmul(a, b)` (Aᵀ·B)      | ✅ | ✅ | `ab-smoke-matmul-variants` | Both inputs uploaded transposed |
| `add(a, b)`                  | ✅ | ✅ | `ab-smoke-add` | Element-wise |
| `gelu(a)`                    | ✅ | ✅ | `ab-smoke-gelu` | Tanh approx, matches `feed_forward` |
| `rms_norm(x, γ, ε)`          | ✅ | ✅ | `ab-smoke-rms-norm` | Last-dim normalize + γ broadcast |
| `softmax(a)`                 | ✅ | ✅ | `ab-smoke-softmax` | Per-row along ne[0] |
| `scale(a, s)`                | ✅ | ✅ | `ab-smoke-scale` | a * scalar |
| `embed_lookup(tab, idx)`     | ✅ | ✅ | `ab-smoke-embed` | Gather rows |
| `embed_back(d_out, idx, V)`  | ✅ | ✅ | `ab-smoke-embed` | Scatter-add (accumulates on duplicates) |
| `softmax_back(a_soft, dy)`   | ✅ | ✅ | `ab-smoke-back` | Backward via `ggml_soft_max_ext_back` |
| `sgd_step(p, g, lr)`         | ✅ | ✅ | `ab-smoke-sgd` | Composed `add(p, scale(g, -lr))` |
| `ffn_pipeline(h, w1, w2)`    | ✅ | ✅ | `ab-smoke-pipeline` | `gelu(h·w1)·w2` chained |
| `transpose(a)` (standalone)  | 🟡 | — | — | `ggml_cont(ggml_transpose(...))` hits the allocator; fold into consumers |
| `rms_norm_back(x, dy, ε)`    | 🟡 | — | — | Binding exists; ggml output doesn't match its own documented formula. Open mystery, FFI binding intentionally left in place pending debug |
| `gelu_back(x, dh)`           | ❌ | ❌ | — | No ggml op; needs hand-rolled CPU/CUDA kernel |
| `cross_entropy_grad`         | ❌ | ❌ | — | Composable from softmax + scale + add; needs a one-hot helper |
| `adam_step(p, g, m, v, ...)` | ❌ | ❌ | — | Multi-component (running m, v, bias correction); ggml has `ggml_opt_step_adamw` |

13 ops verified parity on both backends; 4 remaining.

## Performance

Numbers from this host (NVIDIA GB10, sm_121, gx10):

### Toy-LM shape `(8, 16) · (16, 32)`, 50 iterations

```
one-shot   TinyNN.matmul:    23.87 ms  (477 us / iter)
persistent compute:           0.36 ms  (  7 us / iter)         66× speedup over one-shot

one-shot   TinyNNCuda.matmul: 347 ms (7.0 ms / iter)
persistent compute:           237 ms (4.7 ms / iter)            1.5× speedup, still slow
```

The persistent CPU path is now *faster than native* `Mat#matmul`
(~200 µs at this shape per `ab_smoke_big`). CUDA loses to CPU at toy
shapes — kernel-launch overhead dominates when the math is 4 KB.

### Real-LLM shape `(64, 512) · (512, 32000)` (tied-unembed-sized), 5 iterations

```
native     5 x Mat#matmul:           12893 ms  (2580 ms / iter)
persistent 5 x compute(+up/down):      287 ms  (  57 ms / iter)   45× speedup over native
```

At this scale pure Ruby is unviable. The persistent FFI path takes
57 ms/iter, of which ~37 ms is the per-element upload/download loop —
which `matz/spinel#474` (`:float_array` / `:int_array` typed FFI
specs) would collapse to a single memcpy, bringing it close to
native ggml-cpu throughput.

## Architecture notes

### `tinynn_engine` vs `tinynn_session`

The C shim caches a `tnn_engine` per `prefer_cuda` flavor (backend +
scheduler, one each for CPU and CUDA across the whole process). Each
`tnn_session_new` reuses the cached engine but builds its own
`ggml_context + cgraph + scratch buffer`. The persistent-session API
uses `tnn_session_new` exactly once and runs many `tnn_compute` calls
against the same allocated graph — the win quantified above.

### Shape conventions

Project `Mat` is row-major: `flat[i * ncols + j]` is element `(i, j)`.
ggml stores 2-D tensors column-major-ish: `ne[0]` is the fastest stride
dimension, `ne[1]` the slower. The shim translates: a project `Mat` of
shape `(rows, cols)` becomes a ggml tensor with `ne[0]=cols, ne[1]=rows`.

`ggml_mul_mat(A, B)` computes `A · Bᵀ`. To get the project's `Mat#matmul`
(plain `A · B`) we upload `B` transposed — see
`TinyNN.upload_transposed`. The matmul result has its ggml `ne[0]`
holding the *output rows* dim and `ne[1]` the *output cols*, which is
why `TinyNN.download_matmul(sess, tc, m, n)` indexes as `scratch[j*m + i]`.

### Scratch buffer

Each session owns a 16 MiB host-side scratch buffer used as the
staging area for per-element FFI uploads (`tnn_scratch_set`) before a
single bulk `ggml_backend_tensor_set`. The same bytes are reinterpreted
as `int32` when uploading row indices (`tnn_scratch_set_i32`) — cheap
hack until [`matz/spinel#474`](https://github.com/matz/spinel/issues/474)
lands typed-array FFI.

## Upstream status

- [`matz/spinel#473`](https://github.com/matz/spinel/issues/473) —
  `train_minimal` SIGBUS in `sp_gc_alloc` during backward. Blocks
  *end-to-end* training verification, not the bridge itself.
- [`matz/spinel#474`](https://github.com/matz/spinel/issues/474) —
  `:float_array` / `:int_array` FFI type specs for zero-copy bulk
  transfer. Eliminates the ~37 ms/iter upload-loop cost in the
  real-LLM-shape numbers above.

## What's not here yet

See the op-coverage table above. Roughly:

- **Custom `gelu_back` kernel** — ggml has no GeLU backward op
  (only `silu_back`). Needs a hand-rolled CPU/CUDA kernel.
- **`rms_norm_back` debug** — binding exists, FFI returns values that
  disagree with ggml's own documented formula in the source. Needs a
  standalone C reproducer to disambiguate ggml-vs-shim.
- **`adam_step`** — composable from element-wise ops + scalar
  divisions, or via `ggml_opt_step_adamw`. Multi-component state.
- **`cross_entropy_grad`** — fused `softmax(logits) - one_hot(target)`,
  composable from `TinyNN.softmax` + a one-hot helper + `TinyNN.add`.
- **GGUF / safetensors loader** — needs a wrapper around
  `gguf_init_params` (struct-by-value, which Spinel's FFI can't pass).

None of these are blocked on the upstream Spinel issues; all are real
code that just needs writing.
