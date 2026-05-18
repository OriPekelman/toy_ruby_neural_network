# CUDA parity for KV-cache write — extends the existing CPU-only
# tinynn/ab_smoke_kv_write.rb by also running the same graph on the
# CUDA backend and comparing.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/tinynn_cuda"

MAX_T  = 4
D_HEAD = 3
POS    = 2

row_bytes = D_HEAD * 4

# Shared input.
k_new = Mat.new(1, D_HEAD)
k_new.flat[0] = 1.0
k_new.flat[1] = 2.0
k_new.flat[2] = 3.0

# --- CPU ---
sess = TinyNN.tnn_session_new(0)
t_K = TinyNN.tnn_input_2d_f32_persistent(sess, MAX_T, D_HEAD)
TinyNN.tnn_finalize_weights(sess)
t_k_new   = TinyNN.tnn_input_2d_f32(sess, 1, D_HEAD)
t_K_slot  = TinyNN.tnn_view_2d(sess, t_K, D_HEAD, 1, row_bytes, POS * row_bytes)
t_cpy     = TinyNN.tnn_cpy(sess, t_k_new, t_K_slot)
TinyNN.tnn_set_output(t_cpy)
TinyNN.tnn_set_output(t_K)
TinyNN.tnn_realize(sess, t_cpy)
TinyNN.upload_row_major(sess, t_k_new, k_new)
TinyNN.tnn_compute(sess)
TinyNN.tnn_download(sess, t_K)
cpu_slot = []; cpu_slot.pop
i = 0
while i < D_HEAD
  cpu_slot.push(TinyNN.tnn_scratch_get(sess, POS * D_HEAD + i))
  i = i + 1
end

# --- CUDA ---
sess2 = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(sess2)
t_K2 = TinyNNCuda.tnn_input_2d_f32_persistent(sess2, MAX_T, D_HEAD)
TinyNNCuda.tnn_finalize_weights(sess2)
t_k_new2  = TinyNNCuda.tnn_input_2d_f32(sess2, 1, D_HEAD)
t_K_slot2 = TinyNNCuda.tnn_view_2d(sess2, t_K2, D_HEAD, 1, row_bytes, POS * row_bytes)
t_cpy2    = TinyNNCuda.tnn_cpy(sess2, t_k_new2, t_K_slot2)
TinyNNCuda.tnn_set_output(t_cpy2)
TinyNNCuda.tnn_set_output(t_K2)
TinyNNCuda.tnn_realize(sess2, t_cpy2)
TinyNNCuda.upload_row_major(sess2, t_k_new2, k_new)
TinyNNCuda.tnn_compute(sess2)
TinyNNCuda.tnn_download(sess2, t_K2)
gpu_slot = []; gpu_slot.pop
i = 0
while i < D_HEAD
  gpu_slot.push(TinyNNCuda.tnn_scratch_get(sess2, POS * D_HEAD + i))
  i = i + 1
end

# Compare just the slot (rest is uninitialized memory).
print "expected slot:"
i = 0
while i < D_HEAD; print " " + k_new.flat[i].to_s; i = i + 1; end
puts ""
print "cpu  slot   :"
i = 0
while i < D_HEAD; print " " + cpu_slot[i].to_s; i = i + 1; end
puts ""
print "cuda slot   :"
i = 0
while i < D_HEAD; print " " + gpu_slot[i].to_s; i = i + 1; end
puts ""

max_diff = 0.0
i = 0
while i < D_HEAD
  d = cpu_slot[i] - gpu_slot[i]
  if d < 0; d = -d; end
  if d > max_diff; max_diff = d; end
  i = i + 1
end
puts "slot max_abs_diff: " + max_diff.to_s
puts "match: " + (max_diff < 1.0e-5 ? "true" : "false")
