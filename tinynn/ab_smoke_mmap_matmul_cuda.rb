# Isolation test for the Phase 2 BYO-pointer mmap-on-CUDA bug.
# Same matmul on CPU mmap (correct baseline) vs CUDA mmap.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/tinynn_cuda"

GGUF = "data/qwen25-0.5b-native.gguf"

# Qwen2.5-0.5B shapes for blk.0.attn_q.weight (per-head slice).
D_MODEL = 896
D_HEAD  = 64

# Build the synthetic q activation (shared on both sides).
synth_q = Mat.new(1, D_MODEL)
i = 0
while i < D_MODEL
  synth_q.flat[i] = ((i % 17).to_f - 8.0) * 0.0125
  i = i + 1
end

# --- CPU mmap path (known correct from earlier parity work) ---
gguf_c = TinyNN.tnn_gguf_load(GGUF)
sess_c = TinyNN.tnn_session_new(0)
map_c  = TinyNN.tnn_gguf_mmap_base(gguf_c)
size_c = TinyNN.tnn_gguf_mmap_size(gguf_c)
rc_c   = TinyNN.tnn_session_attach_weight_mmap(sess_c, map_c, size_c)
puts "CPU attach rc=" + rc_c.to_s

q_idx_c  = TinyNN.tnn_gguf_find_index(gguf_c, "blk.0.attn_q.weight")
q_off_c  = TinyNN.tnn_gguf_tensor_file_offset(gguf_c, q_idx_c)
q_type_c = TinyNN.tnn_gguf_tensor_type(gguf_c, q_idx_c)
t_w_c    = TinyNN.tnn_input_2d_persistent_mmap(sess_c, D_HEAD, D_MODEL, q_type_c, q_off_c)
puts "CPU weight ptr null? " + (t_w_c == TinyNN.tnn_null_ptr).to_s

t_q_c    = TinyNN.tnn_input_2d_f32(sess_c, 1, D_MODEL)
t_s_c    = TinyNN.tnn_matmul(sess_c, t_w_c, t_q_c)
TinyNN.tnn_set_output(t_s_c)
TinyNN.tnn_realize(sess_c, t_s_c)
TinyNN.upload_row_major(sess_c, t_q_c, synth_q)
TinyNN.tnn_compute(sess_c)
cpu_scores = TinyNN.download_row_major(sess_c, t_s_c, D_HEAD, 1)

# --- CUDA mmap path (the suspect) ---
gguf_g = TinyNNCuda.tnn_gguf_load(GGUF)
sess_g = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(sess_g)
map_g  = TinyNNCuda.tnn_gguf_mmap_base(gguf_g)
size_g = TinyNNCuda.tnn_gguf_mmap_size(gguf_g)
rc_g   = TinyNNCuda.tnn_session_attach_weight_mmap(sess_g, map_g, size_g)
puts "CUDA attach rc=" + rc_g.to_s

q_idx_g  = TinyNNCuda.tnn_gguf_find_index(gguf_g, "blk.0.attn_q.weight")
q_off_g  = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_g, q_idx_g)
q_type_g = TinyNNCuda.tnn_gguf_tensor_type(gguf_g, q_idx_g)
t_w_g    = TinyNNCuda.tnn_input_2d_persistent_mmap(sess_g, D_HEAD, D_MODEL, q_type_g, q_off_g)
puts "CUDA weight ptr null? " + (t_w_g == TinyNNCuda.tnn_null_ptr).to_s

t_q_g    = TinyNNCuda.tnn_input_2d_f32(sess_g, 1, D_MODEL)
t_s_g    = TinyNNCuda.tnn_matmul(sess_g, t_w_g, t_q_g)
TinyNNCuda.tnn_set_output(t_s_g)
TinyNNCuda.tnn_realize(sess_g, t_s_g)
TinyNNCuda.upload_row_major(sess_g, t_q_g, synth_q)
TinyNNCuda.tnn_compute(sess_g)
gpu_scores = TinyNNCuda.download_row_major(sess_g, t_s_g, D_HEAD, 1)

# Compare.
print "cpu scores[0..5]: "
i = 0
while i < 6; print " " + cpu_scores.flat[i].to_s; i = i + 1; end
puts ""
print "gpu scores[0..5]: "
i = 0
while i < 6; print " " + gpu_scores.flat[i].to_s; i = i + 1; end
puts ""

max_diff = 0.0
n_off = 0
i = 0
while i < D_HEAD
  d = cpu_scores.flat[i] - gpu_scores.flat[i]
  if d < 0; d = -d; end
  if d > max_diff; max_diff = d; end
  if d > 1.0e-4; n_off = n_off + 1; end
  i = i + 1
end
puts "match (single): " + (max_diff < 1.0e-4 ? "true" : "false") +
     " max_abs_diff=" + max_diff.to_s

# ------------------------------------------------------------
# Second iteration: rebuild + compute again on the same session.
# This mirrors what inference does (tnn_reset_for_rebuild between
# decode_steps). If the SECOND iteration differs from CPU, the
# bug is in the reset/repeat pattern with mmap-backed weights.
# ------------------------------------------------------------
puts ""
puts "=== iter 2 (reset + rebuild) ==="

TinyNN.tnn_reset_for_rebuild(sess_c)
t_q_c2 = TinyNN.tnn_input_2d_f32(sess_c, 1, D_MODEL)
t_s_c2 = TinyNN.tnn_matmul(sess_c, t_w_c, t_q_c2)
TinyNN.tnn_set_output(t_s_c2)
TinyNN.tnn_realize(sess_c, t_s_c2)
TinyNN.upload_row_major(sess_c, t_q_c2, synth_q)
TinyNN.tnn_compute(sess_c)
cpu_scores2 = TinyNN.download_row_major(sess_c, t_s_c2, D_HEAD, 1)

TinyNNCuda.tnn_reset_for_rebuild(sess_g)
t_q_g2 = TinyNNCuda.tnn_input_2d_f32(sess_g, 1, D_MODEL)
t_s_g2 = TinyNNCuda.tnn_matmul(sess_g, t_w_g, t_q_g2)
TinyNNCuda.tnn_set_output(t_s_g2)
TinyNNCuda.tnn_realize(sess_g, t_s_g2)
TinyNNCuda.upload_row_major(sess_g, t_q_g2, synth_q)
TinyNNCuda.tnn_compute(sess_g)
gpu_scores2 = TinyNNCuda.download_row_major(sess_g, t_s_g2, D_HEAD, 1)

print "iter 2 cpu[0..3]: "
i = 0
while i < 4; print " " + cpu_scores2.flat[i].to_s; i = i + 1; end
puts ""
print "iter 2 gpu[0..3]: "
i = 0
while i < 4; print " " + gpu_scores2.flat[i].to_s; i = i + 1; end
puts ""

max_diff2 = 0.0
i = 0
while i < D_HEAD
  d = cpu_scores2.flat[i] - gpu_scores2.flat[i]
  if d < 0; d = -d; end
  if d > max_diff2; max_diff2 = d; end
  i = i + 1
end
puts "match (iter2): " + (max_diff2 < 1.0e-4 ? "true" : "false") +
     " max_abs_diff=" + max_diff2.to_s

# Also compare iter 1 vs iter 2 on each side (should be IDENTICAL —
# same inputs, same weights).
max_self_diff_cpu = 0.0
max_self_diff_gpu = 0.0
i = 0
while i < D_HEAD
  dc = cpu_scores.flat[i] - cpu_scores2.flat[i]
  dg = gpu_scores.flat[i] - gpu_scores2.flat[i]
  if dc < 0; dc = -dc; end
  if dg < 0; dg = -dg; end
  if dc > max_self_diff_cpu; max_self_diff_cpu = dc; end
  if dg > max_self_diff_gpu; max_self_diff_gpu = dg; end
  i = i + 1
end
puts "cpu iter1 vs iter2 max_abs_diff: " + max_self_diff_cpu.to_s
puts "gpu iter1 vs iter2 max_abs_diff: " + max_self_diff_gpu.to_s

# ------------------------------------------------------------
# Stress test: more like real inference — multiple matmuls in
# the same graph, each against a different mmap-backed weight,
# AND a regular ctx_w tensor (like K/V cache) in the mix.
# ------------------------------------------------------------
puts ""
puts "=== chained 3-matmul test ==="

# Reset both, build a chain: 3 sequential matmuls feeding each
# other, with the q input flowing through 3 different attn_q
# head slices.

def chain_matmul_cpu(sess, gguf, synth_q, n_chain)
  d_model = D_MODEL
  d_head  = D_HEAD
  q_idx   = TinyNN.tnn_gguf_find_index(gguf, "blk.0.attn_q.weight")
  q_off   = TinyNN.tnn_gguf_tensor_file_offset(gguf, q_idx)
  q_type  = TinyNN.tnn_gguf_tensor_type(gguf, q_idx)

  TinyNN.tnn_reset_for_rebuild(sess)
  t_q = TinyNN.tnn_input_2d_f32(sess, 1, d_model)
  t_acc = t_q   # accumulator starts as q
  per_head = d_head * d_model * 4
  h = 0
  while h < n_chain
    head_off = q_off + h * per_head
    t_w = TinyNN.tnn_input_2d_persistent_mmap(sess, d_head, d_model, q_type, head_off)
    # matmul(t_w[d_head, d_model], t_acc[d_model, 1]) -> ne=[d_head, 1]
    # Need to bring back to d_model for chaining. Use scale to keep
    # values bounded; pad by repeating elements isn't simple in ggml.
    # Easier: just compute n matmuls of fresh-from-q. We're testing
    # MULTIPLE mmap reads in one graph, not a real chain.
    t_score = TinyNN.tnn_matmul(sess, t_w, t_q)
    TinyNN.tnn_set_output(t_score)
    if h == n_chain - 1
      TinyNN.tnn_realize(sess, t_score)
      TinyNN.upload_row_major(sess, t_q, synth_q)
      TinyNN.tnn_compute(sess)
      return TinyNN.download_row_major(sess, t_score, d_head, 1)
    end
    h = h + 1
  end
  Mat.new(d_head, 1)
end

def chain_matmul_cuda(sess, gguf, synth_q, n_chain)
  d_model = D_MODEL
  d_head  = D_HEAD
  q_idx   = TinyNNCuda.tnn_gguf_find_index(gguf, "blk.0.attn_q.weight")
  q_off   = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf, q_idx)
  q_type  = TinyNNCuda.tnn_gguf_tensor_type(gguf, q_idx)

  TinyNNCuda.tnn_reset_for_rebuild(sess)
  t_q = TinyNNCuda.tnn_input_2d_f32(sess, 1, d_model)
  per_head = d_head * d_model * 4
  h = 0
  while h < n_chain
    head_off = q_off + h * per_head
    t_w = TinyNNCuda.tnn_input_2d_persistent_mmap(sess, d_head, d_model, q_type, head_off)
    t_score = TinyNNCuda.tnn_matmul(sess, t_w, t_q)
    TinyNNCuda.tnn_set_output(t_score)
    if h == n_chain - 1
      TinyNNCuda.tnn_realize(sess, t_score)
      TinyNNCuda.upload_row_major(sess, t_q, synth_q)
      TinyNNCuda.tnn_compute(sess)
      return TinyNNCuda.download_row_major(sess, t_score, d_head, 1)
    end
    h = h + 1
  end
  Mat.new(d_head, 1)
end

cpu_chain = chain_matmul_cpu(sess_c, gguf_c, synth_q, 3)
gpu_chain = chain_matmul_cuda(sess_g, gguf_g, synth_q, 3)
print "cpu chain[0..3]:"
i = 0
while i < 4; print " " + cpu_chain.flat[i].to_s; i = i + 1; end
puts ""
print "gpu chain[0..3]:"
i = 0
while i < 4; print " " + gpu_chain.flat[i].to_s; i = i + 1; end
puts ""
mx = 0.0
i = 0
while i < D_HEAD
  d = cpu_chain.flat[i] - gpu_chain.flat[i]
  if d < 0; d = -d; end
  if d > mx; mx = d; end
  i = i + 1
end
puts "chain max_abs_diff: " + mx.to_s
puts "match (chain): " + (mx < 1.0e-4 ? "true" : "false")

# ------------------------------------------------------------
# Mix test: graph uses BOTH a ctx_w_mmap tensor (the mmap'd
# weight) AND a ctx_w tensor (regular persistent — mimics K/V
# cache). Full inference does this; my prior chain tests only
# touched ctx_w_mmap.
#
# Pattern: tnn_input_2d_f32_persistent for a "K-cache-like"
# tensor (allocated normally in ctx_w with its own buffer).
# Build a graph: q -> matmul(W_mmap, q) -> add(result, K[0,:]).
# K's first row will be zero (uninitialized — see ab_smoke_kv_write
# for the same caveat) but the add still exercises the cross-ctx
# path.
# ------------------------------------------------------------
puts ""
puts "=== mix test (ctx_w_mmap weight + ctx_w persistent) ==="

# We need to CREATE the ctx_w persistent tensors BEFORE
# tnn_finalize_weights is called (which already happened implicitly).
# Hmm — actually realize_for_mmap calls finalize_weights at the end.
# For this test we need fresh sessions. Skip the redo: this test
# tells us the answer with a different bias add pattern instead.

# Trick: use a ctx_w_mmap-backed bias too (1D pointing at
# attn_q.bias's 1st-head slice in the GGUF). Add it to the matmul
# result. This exercises mmap-backed tensors of TWO DIFFERENT
# SHAPES in the same graph.

TinyNN.tnn_reset_for_rebuild(sess_c)
qb_idx_c = TinyNN.tnn_gguf_find_index(gguf_c, "blk.0.attn_q.bias")
if qb_idx_c >= 0
  qb_off_c = TinyNN.tnn_gguf_tensor_file_offset(gguf_c, qb_idx_c)
  t_b_c = TinyNN.tnn_input_1d_persistent_mmap(sess_c, D_HEAD, 0, qb_off_c)
  t_q_c3 = TinyNN.tnn_input_2d_f32(sess_c, 1, D_MODEL)
  t_s_c3 = TinyNN.tnn_matmul(sess_c, t_w_c, t_q_c3)
  t_sb_c3 = TinyNN.tnn_add(sess_c, t_s_c3, t_b_c)
  TinyNN.tnn_set_output(t_sb_c3)
  TinyNN.tnn_realize(sess_c, t_sb_c3)
  TinyNN.upload_row_major(sess_c, t_q_c3, synth_q)
  TinyNN.tnn_compute(sess_c)
  cpu_mix = TinyNN.download_row_major(sess_c, t_sb_c3, D_HEAD, 1)

  TinyNNCuda.tnn_reset_for_rebuild(sess_g)
  qb_idx_g = TinyNNCuda.tnn_gguf_find_index(gguf_g, "blk.0.attn_q.bias")
  qb_off_g = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_g, qb_idx_g)
  t_b_g = TinyNNCuda.tnn_input_1d_persistent_mmap(sess_g, D_HEAD, 0, qb_off_g)
  t_q_g3 = TinyNNCuda.tnn_input_2d_f32(sess_g, 1, D_MODEL)
  t_s_g3 = TinyNNCuda.tnn_matmul(sess_g, t_w_g, t_q_g3)
  t_sb_g3 = TinyNNCuda.tnn_add(sess_g, t_s_g3, t_b_g)
  TinyNNCuda.tnn_set_output(t_sb_g3)
  TinyNNCuda.tnn_realize(sess_g, t_sb_g3)
  TinyNNCuda.upload_row_major(sess_g, t_q_g3, synth_q)
  TinyNNCuda.tnn_compute(sess_g)
  gpu_mix = TinyNNCuda.download_row_major(sess_g, t_sb_g3, D_HEAD, 1)

  print "cpu mix[0..3]:"
  i = 0
  while i < 4; print " " + cpu_mix.flat[i].to_s; i = i + 1; end
  puts ""
  print "gpu mix[0..3]:"
  i = 0
  while i < 4; print " " + gpu_mix.flat[i].to_s; i = i + 1; end
  puts ""
  mxm = 0.0
  i = 0
  while i < D_HEAD
    d = cpu_mix.flat[i] - gpu_mix.flat[i]
    if d < 0; d = -d; end
    if d > mxm; mxm = d; end
    i = i + 1
  end
  puts "mix max_abs_diff: " + mxm.to_s
  puts "match (mix): " + (mxm < 1.0e-4 ? "true" : "false")
end

# ------------------------------------------------------------
# FULL pattern: ctx_w persistent (regular F32) tensor PLUS
# ctx_w_mmap weight tensor, both used in the same graph, then
# tnn_finalize_weights, then compute.
#
# This matches what realize_for_mmap does — allocates K/V cache
# in ctx_w via tnn_input_2d_f32_persistent AND mmap weights in
# ctx_w_mmap, then finalize.
# ------------------------------------------------------------
puts ""
puts "=== full pattern: ctx_w + ctx_w_mmap both populated ==="

# Fresh sessions for this test (the prior ones are too far along).
gguf_c2 = TinyNN.tnn_gguf_load(GGUF)
sess_c2 = TinyNN.tnn_session_new(0)
TinyNN.tnn_session_attach_weight_mmap(sess_c2,
  TinyNN.tnn_gguf_mmap_base(gguf_c2),
  TinyNN.tnn_gguf_mmap_size(gguf_c2))
qi_c2 = TinyNN.tnn_gguf_find_index(gguf_c2, "blk.0.attn_q.weight")
qo_c2 = TinyNN.tnn_gguf_tensor_file_offset(gguf_c2, qi_c2)
qt_c2 = TinyNN.tnn_gguf_tensor_type(gguf_c2, qi_c2)
t_w_full_c = TinyNN.tnn_input_2d_persistent_mmap(sess_c2, D_HEAD, D_MODEL, qt_c2, qo_c2)
# Add a ctx_w persistent tensor — like a K/V cache buffer.
t_kv_c = TinyNN.tnn_input_2d_f32_persistent(sess_c2, 4, D_HEAD)   # ne=[d_head, 4]
TinyNN.tnn_finalize_weights(sess_c2)
# Build graph: matmul(W_mmap, q) -> add to a view of K cache slot.
# Simpler: matmul(W_mmap, q), set_output, also touch K via a
# noop (multiply by 1). Both must be in the graph for the
# scheduler to see both contexts.
t_q_c4 = TinyNN.tnn_input_2d_f32(sess_c2, 1, D_MODEL)
t_s_c4 = TinyNN.tnn_matmul(sess_c2, t_w_full_c, t_q_c4)
TinyNN.tnn_set_output(t_s_c4)
# Touch K via a scale (to make sure it's in the graph).
t_k_scaled_c = TinyNN.tnn_scale(sess_c2, t_kv_c, 1.0)
TinyNN.tnn_set_output(t_k_scaled_c)
TinyNN.tnn_realize(sess_c2, t_s_c4)
TinyNN.upload_row_major(sess_c2, t_q_c4, synth_q)
TinyNN.tnn_compute(sess_c2)
cpu_full = TinyNN.download_row_major(sess_c2, t_s_c4, D_HEAD, 1)

gguf_g2 = TinyNNCuda.tnn_gguf_load(GGUF)
sess_g2 = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend (full test): " + TinyNNCuda.tnn_backend_name(sess_g2)
TinyNNCuda.tnn_session_attach_weight_mmap(sess_g2,
  TinyNNCuda.tnn_gguf_mmap_base(gguf_g2),
  TinyNNCuda.tnn_gguf_mmap_size(gguf_g2))
qi_g2 = TinyNNCuda.tnn_gguf_find_index(gguf_g2, "blk.0.attn_q.weight")
qo_g2 = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_g2, qi_g2)
qt_g2 = TinyNNCuda.tnn_gguf_tensor_type(gguf_g2, qi_g2)
t_w_full_g = TinyNNCuda.tnn_input_2d_persistent_mmap(sess_g2, D_HEAD, D_MODEL, qt_g2, qo_g2)
t_kv_g = TinyNNCuda.tnn_input_2d_f32_persistent(sess_g2, 4, D_HEAD)
TinyNNCuda.tnn_finalize_weights(sess_g2)
t_q_g4 = TinyNNCuda.tnn_input_2d_f32(sess_g2, 1, D_MODEL)
t_s_g4 = TinyNNCuda.tnn_matmul(sess_g2, t_w_full_g, t_q_g4)
TinyNNCuda.tnn_set_output(t_s_g4)
t_k_scaled_g = TinyNNCuda.tnn_scale(sess_g2, t_kv_g, 1.0)
TinyNNCuda.tnn_set_output(t_k_scaled_g)
TinyNNCuda.tnn_realize(sess_g2, t_s_g4)
TinyNNCuda.upload_row_major(sess_g2, t_q_g4, synth_q)
TinyNNCuda.tnn_compute(sess_g2)
gpu_full = TinyNNCuda.download_row_major(sess_g2, t_s_g4, D_HEAD, 1)

print "cpu full[0..3]:"
i = 0
while i < 4; print " " + cpu_full.flat[i].to_s; i = i + 1; end
puts ""
print "gpu full[0..3]:"
i = 0
while i < 4; print " " + gpu_full.flat[i].to_s; i = i + 1; end
puts ""
mxf = 0.0
i = 0
while i < D_HEAD
  d = cpu_full.flat[i] - gpu_full.flat[i]
  if d < 0; d = -d; end
  if d > mxf; mxf = d; end
  i = i + 1
end
puts "full max_abs_diff: " + mxf.to_s
puts "match (full): " + (mxf < 1.0e-4 ? "true" : "false")
