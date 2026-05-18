# CUDA parity for matmul where one operand is a view_2d of a
# persistent tensor.
#
# In SmolLM2KV.decode_step:
#   t_K_view = view_2d(blk.t_K, d_head, pos+1, bytes_d_head, 0)
#   t_scores = matmul(t_K_view, t_q_rot)
# i.e., A is a view into the persistent K buffer; B is a compute
# tensor. Existing ab_smoke_*_cuda matmul tests only check matmul
# with freshly-allocated, contiguous, NON-view operands. This is
# the missing parity case.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/tinynn_cuda"

D_HEAD = 4
MAX_T  = 8
POSP1  = 3   # like (pos+1) when pos=2 — view covers 3 K rows

# Synthetic K-buffer values (shape ne=[d_head, max_T]) and Q vector
# (ne=[d_head, 1]).
k_buf_full = Mat.new(MAX_T, D_HEAD)
i = 0
while i < MAX_T * D_HEAD
  k_buf_full.flat[i] = (i.to_f - (MAX_T * D_HEAD).to_f * 0.5) * 0.1
  i = i + 1
end
q_in = Mat.new(1, D_HEAD)
i = 0
while i < D_HEAD
  q_in.flat[i] = (i + 1).to_f * 0.2 - 0.5
  i = i + 1
end

# --- CPU ---
cpu_sess = TinyNN.tnn_session_new(0)
t_K_c = TinyNN.tnn_input_2d_f32_persistent(cpu_sess, MAX_T, D_HEAD)
TinyNN.tnn_finalize_weights(cpu_sess)
t_q_c = TinyNN.tnn_input_2d_f32(cpu_sess, 1, D_HEAD)
# View first POSP1 timesteps of K: ne=[d_head, POSP1], stride
# bytes_d_head between timesteps, offset 0.
t_K_view_c = TinyNN.tnn_view_2d(cpu_sess, t_K_c, D_HEAD, POSP1, D_HEAD * 4, 0)
t_scores_c = TinyNN.tnn_matmul(cpu_sess, t_K_view_c, t_q_c)
TinyNN.tnn_set_output(t_scores_c)
TinyNN.tnn_realize(cpu_sess, t_scores_c)
TinyNN.upload_row_major(cpu_sess, t_K_c, k_buf_full)
TinyNN.upload_row_major(cpu_sess, t_q_c, q_in)
TinyNN.tnn_compute(cpu_sess)
cpu_scores = TinyNN.download_row_major(cpu_sess, t_scores_c, POSP1, 1)

# --- CUDA ---
gpu_sess = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(gpu_sess)
t_K_g = TinyNNCuda.tnn_input_2d_f32_persistent(gpu_sess, MAX_T, D_HEAD)
TinyNNCuda.tnn_finalize_weights(gpu_sess)
t_q_g = TinyNNCuda.tnn_input_2d_f32(gpu_sess, 1, D_HEAD)
t_K_view_g = TinyNNCuda.tnn_view_2d(gpu_sess, t_K_g, D_HEAD, POSP1, D_HEAD * 4, 0)
t_scores_g = TinyNNCuda.tnn_matmul(gpu_sess, t_K_view_g, t_q_g)
TinyNNCuda.tnn_set_output(t_scores_g)
TinyNNCuda.tnn_realize(gpu_sess, t_scores_g)
TinyNNCuda.upload_row_major(gpu_sess, t_K_g, k_buf_full)
TinyNNCuda.upload_row_major(gpu_sess, t_q_g, q_in)
TinyNNCuda.tnn_compute(gpu_sess)
gpu_scores = TinyNNCuda.download_row_major(gpu_sess, t_scores_g, POSP1, 1)

# Compare.
max_diff = 0.0
i = 0
while i < POSP1
  d = cpu_scores.flat[i] - gpu_scores.flat[i]
  if d < 0
    d = -d
  end
  if d > max_diff
    max_diff = d
  end
  i = i + 1
end
print "cpu  scores:"
i = 0
while i < POSP1
  print " " + cpu_scores.flat[i].to_s
  i = i + 1
end
puts ""
print "cuda scores:"
i = 0
while i < POSP1
  print " " + gpu_scores.flat[i].to_s
  i = i + 1
end
puts ""
puts "max_abs_diff: " + max_diff.to_s
puts "match: " + (max_diff < 1.0e-4 ? "true" : "false")

TinyNN.tnn_session_free(cpu_sess)
TinyNNCuda.tnn_session_free(gpu_sess)
