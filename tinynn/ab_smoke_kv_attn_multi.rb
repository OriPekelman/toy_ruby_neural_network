# WIP — multi-step KV-cache decode. Single-step parity works
# (ab_smoke_kv_attn) but per-step graph rebuild via
# `tnn_reset_for_rebuild` currently fails at step 1's upload with
# `GGML_ASSERT(buf != NULL && "tensor buffer not set")`. After
# `tnn_reset_for_rebuild` + `tnn_realize`, the fresh compute tensors
# in the second cycle don't have backend buffers assigned. Possibly
# the new ggml_new_graph(ctx) doesn't isolate from the previous
# cgraph's allocation state. Needs investigation — picking up in a
# fresh session.
# Multi-step single-head KV-cache attention. For each position
# p = 0..MAX_T-1, rebuilds the compute graph (V's set_2d offset
# depends on p), uploads k_new/v_new/q/mask/pos_idx, runs forward,
# downloads head_out. The persistent K/V buffers in ctx_w accumulate
# across steps. Each step's head_out is parity-checked against a
# hand-rolled native reference computed on the running K/V history.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

MAX_T  = 8
D_HEAD = 6

# Persistent K (d_head, max_T) and V (max_T, d_head).
sess = TinyNN.tnn_session_new(0)
t_K = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T,  D_HEAD)
t_V = TinyNN.tnn_input_2d_f32_persistent(sess, D_HEAD, MAX_T)
TinyNN.tnn_finalize_weights(sess)

# Zero the persistent buffers (positions > p must not contribute via
# mask, but pre-zeroing the buffer keeps the host-side reference
# math identical to the FFI math regardless of how the backend
# initialised the buffer).
zero_K = Mat.new(MAX_T, D_HEAD)
zero_V = Mat.new(D_HEAD, MAX_T)
TinyNN.upload_row_major(sess, t_K, zero_K)
TinyNN.upload_row_major(sess, t_V, zero_V)

# Host-side K/V "history" -- we maintain a parallel copy so we can
# compute the native reference at each step.
K_history = Mat.new(MAX_T, D_HEAD)
V_history = Mat.new(MAX_T, D_HEAD)

def fill_seq(rows, cols, base, stride)
  out = Mat.new(rows, cols)
  n = rows * cols
  i = 0
  while i < n
    out.flat[i] = base + i.to_f * stride
    i = i + 1
  end
  out
end

def overlay_row(mat, row, src, cols)
  i = 0
  while i < cols
    mat.flat[row * cols + i] = src.flat[i]
    i = i + 1
  end
end

def native_head_out(q, k_hist, v_hist, n_valid, d_head)
  scale = 1.0 / Math.sqrt(d_head.to_f)
  scores = Array.new(n_valid, 0.0)
  p = 0
  while p < n_valid
    s = 0.0
    f = 0
    while f < d_head
      s = s + q.flat[f] * k_hist.flat[p * d_head + f]
      f = f + 1
    end
    scores[p] = s * scale
    p = p + 1
  end
  mx = scores[0]
  i = 1
  while i < n_valid
    if scores[i] > mx
      mx = scores[i]
    end
    i = i + 1
  end
  expsum = 0.0
  i = 0
  while i < n_valid
    scores[i] = Math.exp(scores[i] - mx)
    expsum = expsum + scores[i]
    i = i + 1
  end
  i = 0
  while i < n_valid
    scores[i] = scores[i] / expsum
    i = i + 1
  end
  out = Mat.new(1, d_head)
  f = 0
  while f < d_head
    s = 0.0
    p = 0
    while p < n_valid
      s = s + scores[p] * v_hist.flat[p * d_head + f]
      p = p + 1
    end
    out.flat[f] = s
    f = f + 1
  end
  out
end

scale = 1.0 / Math.sqrt(D_HEAD.to_f)
all_ok = true
max_observed = 0.0

# Multi-step decode loop. For each position, rebuild the compute
# graph with pos baked into V's set_2d offset.
pos = 0
while pos < MAX_T
  TinyNN.tnn_reset_for_rebuild(sess)

  # Compute inputs for THIS step. They're created fresh in ctx each
  # rebuild, but the persistent K/V tensors are reused unchanged.
  t_q     = TinyNN.tnn_input_2d_f32(sess, 1,      D_HEAD)
  t_k_new = TinyNN.tnn_input_2d_f32(sess, 1,      D_HEAD)
  t_v_new = TinyNN.tnn_input_2d_f32(sess, D_HEAD, 1)
  t_idx   = TinyNN.tnn_input_1d_i32(sess, 1)
  t_mask  = TinyNN.tnn_input_2d_f32(sess, 1,      MAX_T)

  t_K_after  = TinyNN.tnn_set_rows(sess, t_K, t_k_new, t_idx)
  t_V_after  = TinyNN.tnn_set_2d(sess,  t_V, t_v_new, MAX_T * 4, pos * 4)
  t_scores   = TinyNN.tnn_matmul(sess, t_K_after, t_q)
  t_attn     = TinyNN.tnn_soft_max_ext(sess, t_scores, t_mask, scale, 0.0)
  t_head_out = TinyNN.tnn_matmul(sess, t_V_after, t_attn)
  TinyNN.tnn_set_output(t_head_out)
  TinyNN.tnn_realize(sess, t_head_out)

  # Inputs for this step.
  q_host     = fill_seq(1, D_HEAD, 0.1 + pos.to_f * 0.05, 0.01)
  k_new_host = fill_seq(1, D_HEAD, 1.0 + pos.to_f * 0.1,  0.02)
  v_new_host = fill_seq(1, D_HEAD, -0.5 + pos.to_f * 0.07, 0.03)
  mask = Mat.new(1, MAX_T)
  i = 0
  while i < MAX_T
    if i <= pos
      mask.flat[i] = 0.0
    else
      mask.flat[i] = -1.0e30
    end
    i = i + 1
  end

  TinyNN.upload_row_major(sess, t_q,     q_host)
  TinyNN.upload_row_major(sess, t_k_new, k_new_host)
  TinyNN.upload_row_major(sess, t_v_new, v_new_host)
  TinyNN.upload_int_array(sess, t_idx, [pos])
  TinyNN.upload_row_major(sess, t_mask, mask)

  TinyNN.tnn_compute(sess)
  ffi_out = TinyNN.download_row_major(sess, t_head_out, 1, D_HEAD)

  # Update host history then compute native reference.
  overlay_row(K_history, pos, k_new_host, D_HEAD)
  overlay_row(V_history, pos, v_new_host, D_HEAD)
  nat_out = native_head_out(q_host, K_history, V_history, pos + 1, D_HEAD)

  # Compare.
  max_d = 0.0
  i = 0
  while i < D_HEAD
    d = nat_out.flat[i] - ffi_out.flat[i]
    if d < 0
      d = -d
    end
    if d > max_d
      max_d = d
    end
    i = i + 1
  end
  if max_d > max_observed
    max_observed = max_d
  end
  ok = max_d < 1.0e-4
  if !ok
    all_ok = false
  end
  puts "pos=" + pos.to_s + ":  max_abs_diff=" + max_d.to_s + " " + (ok ? "OK" : "FAIL")
  pos = pos + 1
end

puts ""
puts "max across all positions: " + max_observed.to_s
puts "M2 MULTI-STEP decode parity: " + (all_ok ? "OK" : "FAIL")
TinyNN.tnn_session_free(sess)
