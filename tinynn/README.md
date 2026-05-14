# tinynn: Mat ↔ ggml FFI bridge

A thin layer that lets the pure-Ruby `Mat` / `TransformerLM` from
`lib/transformer.rb` dispatch its hot matrix operations through
[ggml](https://github.com/ggml-org/ggml) — CPU or CUDA — via Spinel's
FFI.

## Why this exists

Pure-Ruby `Mat#matmul` at training shapes:

| Shape | Time / iter |
|---|---|
| Toy (8, 16) · (16, 32)        | 0.2 ms |
| Real-LLM (64, 512) · (512, 32K) | **2.6 s** |

LLM-scale matmul in pure interpreted Ruby is unviable. Through this
bridge the same call runs in ~29 ms (≈90× faster) on CPU FFI, and
even less on CUDA once activations stay on-device. The bridge is
**opt-in** via a `USE_FFI_MATMUL` flag in `lib/transformer.rb`; pure
Ruby remains the default.

## Layout

```
tinynn/
├── tinynn_ggml.{h,c}         C shim — opaque-handle wrapper over ggml
├── tinynn_gguf.{h,c}         GGUF loader (load + read + dequant)
├── smoke.rb                  4×3 raw matmul demo (no project deps)
├── ab_smoke*.rb              Per-op parity tests (TinyNN.<op> vs Mat#<op>)
├── ab_smoke_big.rb           Transformer-shape parity + wallclock (CPU)
├── ab_smoke_big_cuda.rb      Same on CUDA
├── ab_smoke_all_cuda.rb      Consolidated CUDA parity driver (16 ops)
├── forward_smoke.rb          A/B against a real TransformerLM
├── persistent_bench*.rb      Persistent-session bench at toy and LLM shapes
├── gguf_smoke.rb             GGUF round-trip (write + load + read)
├── rms_norm_back_probe*.c    ggml-org/ggml#1491 standalone repros
└── README.md                 this file

lib/
├── tinynn.rb                 CPU bridge: module TinyNN + FFNFFICache (CPU)
└── tinynn_cuda.rb            CUDA bridge: module TinyNNCuda + FFNFFICache (CUDA)
```

## Build

The Makefile in the project root drives everything. CUDA is opt-in.

```sh
# One-time: clone + build vendored ggml as a static archive (~30 s).
make setup-ggml             # CPU only
make setup-ggml-cuda        # adds CUDA (defaults to sm_121 / GB10)

# Override the CUDA arch:  make setup-ggml-cuda GGML_CUDA_ARCH=90
# Override CUDA toolkit path:  CUDA_DIR=/opt/cuda make setup-ggml-cuda

# Run the full CPU parity sweep:
make test

# Individual smokes:
make smoke                  # 4×3 raw matmul demo
make ab-smoke               # matmul parity
make ab-smoke-pipeline      # gelu(h·w1)·w2 chained
make ab-smoke-big           # transformer-shape parity + wallclock
make forward-smoke          # A/B against a real TransformerLM
make persistent-bench       # 50 iters, one-shot vs persistent (toy shape)
make persistent-bench-big   # same at real-LLM shape (160× speedup)
make gguf-smoke             # GGUF write+load+read round-trip

# CUDA (requires `make setup-ggml-cuda`):
make ab-smoke-cuda          # matmul-only CUDA parity
make ab-smoke-all-cuda      # consolidated 16-op CUDA parity driver
make persistent-bench-cuda  # 50 iters CUDA, one-shot vs persistent
```

## API surface

### One-shot wrappers

The convenience path. Each call spins up a fresh ggml graph, runs
the op, tears down. Good for ad-hoc tests; **slow for hot loops**
(per-call ggml_init + scheduler-alloc dominates the math).

```ruby
require_relative "lib/transformer"
require_relative "lib/tinynn"

a = Mat.new(2, 3); ...
b = Mat.new(3, 2); ...
c     = TinyNN.matmul(a, b)              # A · B
d     = TinyNN.matmul_t(a, b)            # A · Bᵀ (ggml-native)
e     = TinyNN.t_matmul(a, b)            # Aᵀ · B
sum   = TinyNN.add(a, a)
gelu  = TinyNN.gelu(a)                   # tanh approx, matches feed_forward
norm  = TinyNN.rms_norm(a, [1.0, 1.0, 1.0], 1e-5)
sm    = TinyNN.softmax(a)
sc    = TinyNN.scale(a, 0.5)

# Embedding (forward gather + backward scatter-add).
rows  = TinyNN.embed_lookup(table, [2, 0, 4, 2])
grads = TinyNN.embed_back(d_out, [2, 0, 4, 2], vocab_size: 5)

# Backward & optimizer.
dx     = TinyNN.softmax_back(sm, dy)
dx     = TinyNN.gelu_back(x, dh)                    # custom CPU kernel
dlg    = TinyNN.cross_entropy_grad(logits, targets, n_pred)
new_p  = TinyNN.sgd_step(param, grad, 0.05)
result = TinyNN.adam_step(p, g, m, v, lr, b1, b2, eps, omc1, omc2)

# GGUF I/O.
handle = TinyNN.tnn_gguf_load("model.gguf")
puts TinyNN.tnn_gguf_n_tensors(handle)
TinyNN.tnn_gguf_free(handle)
```

For CUDA, swap `TinyNN.` for `TinyNNCuda.` and `require_relative
"lib/tinynn_cuda"` instead. Same shapes, same semantics. Both
backends pass the same 16 parity tests.

### Persistent-session API

The production path. Build a graph once, run it many times. This is
what real training uses.

```ruby
sess = TinyNN.persistent_new(0)               # 0 = CPU, 1 = CUDA
ta   = TinyNN.alloc_2d(sess, T, D)            # input slot
tw   = TinyNN.alloc_2d(sess, D_out, D)        # weight slot (uploaded transposed)
tc   = TinyNN.build_matmul(sess, ta, tw)      # op tree
TinyNN.realize(sess, tc)                       # allocates all backend buffers

TinyNN.upload_transposed(sess, tw, w_mat)      # upload weights ONCE
loop do
  TinyNN.upload_row_major(sess, ta, input_mat)
  TinyNN.compute(sess)
  output = TinyNN.download_matmul(sess, tc, T, D_out)
end

TinyNN.persistent_free(sess)
```

Mixing multiple `build_*` calls before `realize` builds a multi-op
graph that runs as one ggml dispatch — see `ab_smoke_pipeline.rb`
for the FFN shape (`matmul → gelu → matmul`).

### Wiring into `feed_forward`

`TransformerLM` owns a per-block `@ffn_ffi_caches` array — one
`FFNFFICache` per layer, lazy-realized on first forward call. When
`USE_FFI_MATMUL = true` in `lib/transformer.rb`, `transformer_block_into`
calls `feed_forward_ffi(h, block, ffi_cache)` instead of
`feed_forward(h, block)`. Each cache reuses its ggml graphs across
all 40 SGD steps × 3 sequences × 2 matmuls = 240 calls per training
step that would otherwise be 240 fresh `ggml_init` + backend-alloc
cycles.

Internally `FFNFFICache` builds a single ggml session that runs the
whole FFN as one chained graph (`mul_mat → gelu → mul_mat`):

- All three result tensors (`t_pre`, `t_hidden`, `t_out`) are marked
  with `ggml_set_output` so the scheduler keeps their buffers alive
  past `compute` — otherwise ggml's allocator aliases `t_pre`'s buffer
  for the gelu output, silently making backward see the wrong values.
- Operand order `mul_mat(w_t, h)` (NOT `mul_mat(h, w_t)`) keeps the
  result.ne0 = inner-dim of the next matmul, so the chain composes
  without an intermediate `ggml_cont(ggml_transpose)`. All three
  outputs then read back as plain row-major memcpy.

For CUDA the equivalent class is `FFNFFICacheCuda` in
`lib/tinynn_cuda.rb` (different name so both modules can be loaded
into one Spinel translation unit without colliding). To switch
`train_minimal` to CUDA: change `require_relative "tinynn"` to
`"tinynn_cuda"`, sed `TinyNN.` → `TinyNNCuda.` and `FFNFFICache.new`
→ `FFNFFICacheCuda.new` in `lib/transformer.rb`, then rebuild.
A `USE_FFI_CUDA` constant for compile-time dispatch is a
straightforward future addition.

## Op coverage

All 16 ops verified parity on both backends:

| Op | CPU | CUDA | Smoke target | Notes |
|---|---|---|---|---|
| `matmul(a, b)` (A·B)         | ✅ | ✅ | `ab-smoke` | A·Bᵀ via b-transposed upload |
| `matmul_t(a, b)` (A·Bᵀ)      | ✅ | ✅ | `ab-smoke-matmul-variants` | ggml-native |
| `t_matmul(a, b)` (Aᵀ·B)      | ✅ | ✅ | `ab-smoke-matmul-variants` | both inputs uploaded transposed |
| `add(a, b)`                  | ✅ | ✅ | `ab-smoke-add` | element-wise |
| `gelu(a)`                    | ✅ | ✅ | `ab-smoke-gelu` | tanh approx, matches feed_forward |
| `rms_norm(x, γ, ε)`          | ✅ | ✅ | `ab-smoke-rms-norm` | last-dim norm + γ broadcast |
| `softmax(a)`                 | ✅ | ✅ | `ab-smoke-softmax` | per-row along ne[0] |
| `scale(a, s)`                | ✅ | ✅ | `ab-smoke-scale` | a × scalar |
| `embed_lookup(tab, idx)`     | ✅ | ✅ | `ab-smoke-embed` | gather rows |
| `embed_back(d_out, idx, V)`  | ✅ | ✅ | `ab-smoke-embed` | scatter-add (accumulates duplicates) |
| `softmax_back(a_soft, dy)`   | ✅ | ✅ | `ab-smoke-back` | `ggml_soft_max_ext_back` |
| `gelu_back(x, dh)`           | ✅ | ✅ | `ab-smoke-gelu-back` | custom CPU kernel (no ggml op exists) |
| `cross_entropy_grad`         | ✅ | ✅ | `ab-smoke-cegrad` | composed: softmax + scale + add + one-hot |
| `sgd_step(p, g, lr)`         | ✅ | ✅ | `ab-smoke-sgd` | composed: `add(p, scale(g, -lr))` |
| `adam_step(...)`             | ✅ | ✅ | `ab-smoke-adam` | custom CPU kernel (sqrt + div) |
| `ffn_pipeline(h, w1, w2)`    | ✅ | ✅ | `ab-smoke-pipeline` | `gelu(h·w1)·w2` chained |
| FFN chained graph            | ✅ | ✅ | `ab-smoke-ffncache` | `FFNFFICache`: single session, GeLU on GPU |
| `opt_step_adamw(...)`        | ✅ | ✅ | `ab-smoke-adamw-op{,-cuda}` | ggml-native AdamW; in-place update; matches plain Adam at wd=0 |
| GGUF load/write/read         | ✅ | n/a | `gguf-smoke` | F32 verified; quantized via `to_float` traits |
| `transpose(a)` (standalone)  | 🟡 | — | — | `ggml_cont(ggml_transpose)` hits scheduler alloc; fold into consumers |
| `rms_norm_back(x, dy, ε)`    | 🟡 | — | — | FFI bound; ggml output disagrees w/ docs via backend-sched — [ggml#1491](https://github.com/ggml-org/ggml/issues/1491) |

## Performance

All numbers from this host (NVIDIA GB10, sm_121, gx10, aarch64), after
Spinel #474 (`:float_array` zero-copy bulk transfer) landed.

### Toy-LM shape `(8, 16) · (16, 32)`, 50 iterations

```
one-shot   TinyNN.matmul:        18.9 ms  (380 µs / iter)
persistent compute:               0.36 ms (   7 µs / iter)    66× speedup

one-shot   TinyNNCuda.matmul:    12 ms   (240 µs / iter)
persistent compute:               1.1 ms (   22 µs / iter)   11× speedup
```

The persistent CPU path is *faster than native* `Mat#matmul`
(~200 µs at this shape). CUDA persistent at toy shape is competitive
with CPU persistent.

### Real-LLM shape `(64, 512) · (512, 32000)` (tied-unembed-sized), 5 iterations

```
native     5 x Mat#matmul:           4782 ms  (956 ms / iter)
persistent 5 x compute(+up/down):     144 ms  ( 29 ms / iter)    33× speedup over native
```

At this scale pure Ruby is unviable. Persistent FFI does ~33× the
work in the same wallclock; native ggml-cpu would be ≤10 ms/iter,
and the remaining ~19 ms is the dequantize-on-upload f64→f32 loop
in the bulk-upload primitive.

### Full-forward graph (`FullForwardFFICache`) — milestone M1

`FullForwardFFICache` runs an entire transformer model forward as one
persistent ggml graph: embed lookup → per-block (RMSNorm →
multi-head causal attention → residual → RMSNorm → FFN → residual)
→ final RMSNorm → tied unembed. Weights live in `ctx_w` (uploaded
once); per call only `token_ids` cross the FFI boundary.

```
vocab=4096 d_model=384 d_ff=1024 n_heads=6 n_layers=6 T=128, 10 iters:
  native           : 1179.9 ms/iter
  CPU FFI          :   31.0 ms/iter   38.1× speedup
  CUDA FFI         :   33.7 ms/iter   33.9× speedup
```

(CPU edges out CUDA at this scale because ggml-cpu's SIMD matmul is
fast enough that the per-step kernel-launch overhead on CUDA matches
it. CUDA pulls ahead at larger `d_model` / longer `T`.)

Parity vs native `TransformerLM.forward()` at `n_layers=2, n_heads=4,
d_model=16, d_ff=32` matches to `max_abs_diff = 1.4e-4` (within f32 +
ggml's f16-LUT GeLU tolerance).

### End-to-end train_minimal (40 SGD steps × 3 sequences, toy shape)

```
native           (USE_FFI_MATMUL=false)        :  15–26 ms  loss 0.034
CPU FFI chained (single-session FFN)           :  24 ms     loss 0.036
CUDA FFI chained                                :  604 ms    loss 0.036
(prev) CUDA FFI two-session + host-side GeLU    :  757 ms    loss 0.039
CPU FFI one-shot     (pre-persistent baseline)  : 378 ms
CUDA FFI one-shot    (pre-persistent baseline)  :3386 ms
```

CPU FFI chained matches native at toy shape. CUDA chained is 20%
faster than the two-session version and converges slightly better,
but at toy shape it's still **~30× slower than native** because (i)
FFN weights re-upload every forward call, (ii) the optimizer step
runs on host, and (iii) attention/embed/unembed aren't on GPU at
all — every FFN call ferries activations across the host↔device
boundary. The cure is fully-on-GPU training (see "What's next"
below). At LLM scale the host↔device cost is amortised across
vastly more matmul math and CUDA decisively wins.

## Architecture notes

### `tnn_engine` vs `tnn_session`

The C shim caches a `tnn_engine` per `prefer_cuda` flavor (backend
objects + scheduler, one each for CPU and CUDA across the whole
process). `tnn_session_new` allocates a fresh `ggml_context + cgraph
+ scratch buffer` and references the cached engine. With the
persistent-session API, `session_new` runs once per `FFNFFICache`
(once per block) at first forward; `tnn_compute` runs many times
against the same realized graph.

### Shape conventions

Project `Mat` is row-major: `flat[i * ncols + j]` is element `(i, j)`.
ggml stores 2-D tensors with `ne[0]` as the fastest stride dimension
and `ne[1]` slower. The shim's translation: a project `Mat(rows, cols)`
becomes a ggml tensor with `ne[0] = cols, ne[1] = rows`.

`ggml_mul_mat(A, B)` computes `A · Bᵀ` mathematically. To get
project-style `Mat#matmul` (plain `A · B`), upload `B` *transposed*
(`TinyNN.upload_transposed`). The matmul result has `ne[0] = m_A`
holding the *output rows* dim and `ne[1] = n_B` the *output cols* —
which is why the result indexes as `scratch[j*m + i]` in
`TinyNN.download_matmul`.

### Scratch buffer

Each session owns a 16 MiB host-side scratch buffer used as a staging
area. Bulk uploads use Spinel #474's `:float_array` / `:int_array`
specs to memcpy directly from `Mat.flat`'s contiguous f64 storage —
no per-element FFI calls.

## Upstream status

- [`matz/spinel#473`](https://github.com/matz/spinel/issues/473) — ✅
  closed 2026-05-14. `train_minimal` runs end-to-end again.
- [`matz/spinel#474`](https://github.com/matz/spinel/issues/474) — ✅
  closed 2026-05-14. Bulk upload primitive landed.
- [`ggml-org/ggml#1491`](https://github.com/ggml-org/ggml/issues/1491) —
  open. `ggml_rms_norm_back` correctness via backend-sched. Standalone
  C repro in `rms_norm_back_probe*.c`. Blocks `TinyNN.rms_norm_back`
  parity but not training (project's native RMSNorm backward is the
  fallback).

## What's next — fully-on-GPU training

The chained-graph refactor (item 1) keeps activations on GPU within
the FFN; the optimizer wrapper (item 2 step 1) gives us on-device
AdamW. Remaining work to move CUDA from "30× slower at toy shape"
to "wins at all scales":

1. ~~**GeLU as a ggml graph node**~~ — ✅ done. `FFNFFICache` now
   runs `mul_mat → gelu → mul_mat` as one chained graph.
   `ab-smoke-ffncache` proves parity; CUDA train_minimal dropped
   from 757 ms to 604 ms.
2. **Weights pinned on GPU**; on-device optimizer step. Step 1
   landed: `tnn_opt_step_adamw` wraps ggml's built-in AdamW op
   and matches the project's plain-Adam at f32 precision on both
   backends (`ab-smoke-adamw-op{,-cuda}`). Step 2 (wiring it into
   `FFNFFICache` with persistent weight/grad/m/v tensors and a
   second cgraph for the update) is the next refactor. Once
   landed, FFN weights never re-upload and `apply_gradients_adam`
   only ferries gradients (smaller) to GPU.
3. **Full forward as one persistent graph** — attention, embed,
   unembed — not just FFN. The single-step training loop becomes:
   upload tokens (tiny) → compute → download loss (tiny).

Each collapses a class of host↔device transfers. The expected end
state at LLM scale: persistent CUDA ≥100× faster than native.

## What's not here

- **CUDA mirrors for `gelu_back`, `adam_step`** custom kernels —
  they currently run on the session's host scratch buffer regardless
  of backend. For real GPU speed they need ggml-op composition or
  device-side kernels.
- **Quantized GGUF**: f32 verified end-to-end. Q4_K / Q8_0 / F16 etc.
  are wired through ggml's `to_float` type traits but haven't been
  exercised with a real model file.
- **Backward through FFI**: currently the backward pass uses native
  `Mat#matmul_t` / `t_matmul`. Wiring it through FFI would unblock
  fully-on-GPU training; modest extension of the persistent-session
  pattern.
