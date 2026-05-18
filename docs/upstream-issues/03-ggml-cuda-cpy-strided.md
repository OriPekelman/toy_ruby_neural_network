# Issue draft: ggml-cuda cpy ignores destination strides

**Target repo:** `ggml-org/ggml`

**Type:** Bug report (with minimal reproducer)

---

## Title

ggml-cuda: `cpy` ignores destination strides (writes contiguously into a strided view_2d slot)

## Body

### Summary

`ggml_cpy(src, dst)` on the CUDA backend writes `src` contiguously into
`dst->data` starting at the offset, regardless of `dst`'s row stride
(`nb1`). When `dst` is a `view_2d` representing a strided slot of a
2D persistent tensor (e.g. writing one column of a `(d_head, max_T)`
buffer), the CUDA cpy kernel:

- writes `ne0 * ne1` contiguous floats starting at the view's offset
- ignores `nb1`

The CPU backend handles the same call correctly (one element per
row, separated by `nb1` bytes).

This produces deterministic-but-wrong values in the destination
buffer and silently corrupts any compute graph that uses the cpy
op into a strided view — for example, every KV-cache write in a
transformer's decode step where the V cache is shaped
`[max_T, d_head]` (column-major in ggml terms).

### Minimal reproducer

```ruby
# CPU
sess  = TinyNN.tnn_session_new(0)
t_V   = TinyNN.tnn_input_2d_f32_persistent(sess, D_HEAD, MAX_T)  # ne=[max_T, d_head]
TinyNN.tnn_finalize_weights(sess)
t_src = TinyNN.tnn_input_2d_f32(sess, D_HEAD, 1)                  # ne=[1, d_head]
# Slot for "timestep pos" — strided, one element per d_head row.
t_slot = TinyNN.tnn_view_2d(sess, t_V, /*ne0=*/1, /*ne1=*/D_HEAD,
                                   /*nb1=*/MAX_T * 4, /*off=*/POS * 4)
t_cpy  = TinyNN.tnn_cpy(sess, t_src, t_slot)
TinyNN.tnn_set_output(t_cpy)
TinyNN.tnn_set_output(t_V)
TinyNN.tnn_realize(sess, t_cpy)
TinyNN.upload_row_major(sess, t_src, [[1.0, 2.0, 3.0, 4.0]])
TinyNN.tnn_compute(sess)
# Read back t_V — expected: column POS has [1.0, 2.0, 3.0, 4.0]
# distributed across the d_head rows.
```

With `MAX_T=8, D_HEAD=4, POS=3`:

CPU output (correct):
```
k=0: 0 0 0 [1] 0 0 0 0
k=1: 0 0 0 [2] 0 0 0 0
k=2: 0 0 0 [3] 0 0 0 0
k=3: 0 0 0 [4] 0 0 0 0
```

CUDA output (BROKEN):
```
k=0: 0 0 0 [1 2 3 4] 0
k=1: 0 0 0  0 0 0 0  0
k=2: 0 0 0  0 0 0 0  0
k=3: 0 0 0  0 0 0 0  0
```

The CUDA kernel honored the offset but ignored the per-row stride
(`nb1 = MAX_T * 4 = 32` bytes; should be the step between
successive `ne1` slots), instead writing 4 floats contiguously.

### Impact (real-world)

A downstream transformer inference (Qwen2.5-1.5B on NVIDIA GB10,
sm_121a, CUDA 13.0) was producing deterministic-but-wrong logits
through ggml-cuda because every V-cache write corrupted the slot.
K-cache writes are contiguous (the K layout makes the slot one
contiguous `d_head` floats, no stride required), so K was fine —
only V was being silently mangled every decode step.

After 4 days of bisection (per-op parity tests of every ggml-cuda
primitive used in the decode graph, byte-readback verification of
loaded weights, per-layer N=1..28 divergence test), this is the
sole op that produces different output on CUDA vs CPU for the
same inputs.

### Environment

- ggml: `master` at commit `5725fee` plus the local vendored
  ggml-cuda BYO-pointer patch (commits `5f3bee4`, `360dc26`,
  `2384266`) — none of which touch the cpy kernel.
- CUDA 13.0.88, driver bundled with DGX OS.
- NVIDIA GB10 (compute capability 12.1, sm_121a). Likely
  reproduces on all CUDA archs since the cpy kernel logic should
  be arch-independent.
- llama.cpp's existing inference paths probably don't hit this
  because they don't use the `view_2d + cpy` idiom for KV writes
  in the same way (or their V layout produces a contiguous slot).

### Suggested fix

ggml-cuda's `cpy_f32_f32` (and friends) kernel needs to honor the
destination tensor's nb1 (and nb2/nb3 for higher rank) when
writing. Probably analogous to how cuBLAS / cudaMemcpy2DAsync
handle strided destinations.

### Related

This blocks Phase 2 BYO-pointer mmap inference on CUDA — the
underlying mechanism works (verified via per-op parity), but
ANY transformer using a strided V cache write hits this corruption.

Downstream context: a Ruby/Spinel toy inference project where
I was implementing zero-copy `mmap`-backed weight loading on CUDA.
See [oripekelman/toy_ruby_neural_network commit `46a61c0`](https://github.com/oripekelman/toy_ruby_neural_network/commit/46a61c0)
for the bisection trail.

---

## gh command

```sh
gh issue create \
  --repo ggml-org/ggml \
  --title 'ggml-cuda: cpy ignores destination strides (writes contiguously into a strided view_2d slot)' \
  --body-file docs/upstream-issues/03-ggml-cuda-cpy-strided.md
```
