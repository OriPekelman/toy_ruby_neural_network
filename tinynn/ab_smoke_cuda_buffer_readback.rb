# Read back the CUDA-uploaded weight bytes and compare to expected
# (a CPU read of the same tensor). If bytes match, the matmul-side
# kernel is using a different buffer interpretation than CPU. If
# bytes differ, the upload itself is corrupting them.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/tinynn_cuda"

GGUF    = "data/qwen25-1.5b-native.gguf"
D_MODEL = 1536
D_HEAD  = 128

# Set up CPU + CUDA sessions, both load attn_q head-0 weight via
# the same direct-native loader. Read back.

# CPU side.
gguf_c = TinyNN.tnn_gguf_load(GGUF)
sess_c = TinyNN.tnn_session_new(0)
t_w_c  = TinyNN.tnn_input_2d_f32_persistent(sess_c, D_HEAD, D_MODEL)
TinyNN.tnn_finalize_weights(sess_c)
q_idx_c = TinyNN.tnn_gguf_find_index(gguf_c, "blk.0.attn_q.weight")
TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(gguf_c, q_idx_c, sess_c, t_w_c, 0, 12, D_MODEL, D_HEAD)
cpu_w = TinyNN.download_row_major(sess_c, t_w_c, D_HEAD, D_MODEL)
n = D_HEAD * D_MODEL

# CUDA side.
gguf_g = TinyNNCuda.tnn_gguf_load(GGUF)
sess_g = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(sess_g)
t_w_g  = TinyNNCuda.tnn_input_2d_f32_persistent(sess_g, D_HEAD, D_MODEL)
TinyNNCuda.tnn_finalize_weights(sess_g)
q_idx_g = TinyNNCuda.tnn_gguf_find_index(gguf_g, "blk.0.attn_q.weight")
TinyNNCuda.tnn_gguf_copy_head_slice_to_persistent_native(gguf_g, q_idx_g, sess_g, t_w_g, 0, 12, D_MODEL, D_HEAD)
gpu_w = TinyNNCuda.download_row_major(sess_g, t_w_g, D_HEAD, D_MODEL)

# Compare.
max_diff = 0.0
n_off = 0
first_diff = -1
i = 0
while i < n
  d = cpu_w.flat[i] - gpu_w.flat[i]
  if d < 0; d = -d; end
  if d > max_diff
    max_diff = d
    if first_diff < 0; first_diff = i; end
  end
  if d > 1.0e-6; n_off = n_off + 1; end
  i = i + 1
end

puts "n = " + n.to_s
puts "max_abs_diff: " + max_diff.to_s
puts "elements off by >1e-6: " + n_off.to_s
if first_diff >= 0
  puts "first diff at i=" + first_diff.to_s +
       " cpu=" + cpu_w.flat[first_diff].to_s +
       " cuda=" + gpu_w.flat[first_diff].to_s
end
puts "bytes_match: " + (max_diff < 1.0e-6 ? "true" : "false")
