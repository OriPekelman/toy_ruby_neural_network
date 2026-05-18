# CUDA half of the cpy-into-strided-view test. Same shapes / pattern
# as the CPU half. Tests ggml-cuda's cpy kernel with a non-contiguous
# destination — the V-cache-slot write that we suspect of being the
# remaining mmap-CUDA inference bug.

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

MAX_T  = 8
D_HEAD = 4
POS    = 3

src = Mat.new(1, D_HEAD)
src.flat[0] = 1.0
src.flat[1] = 2.0
src.flat[2] = 3.0
src.flat[3] = 4.0

sess = TinyNNCuda.tnn_session_new(1)
puts "CUDA backend: " + TinyNNCuda.tnn_backend_name(sess)

t_V = TinyNNCuda.tnn_input_2d_f32_persistent(sess, D_HEAD, MAX_T)
TinyNNCuda.tnn_finalize_weights(sess)

t_src = TinyNNCuda.tnn_input_2d_f32(sess, D_HEAD, 1)
t_slot = TinyNNCuda.tnn_view_2d(sess, t_V, 1, D_HEAD, MAX_T * 4, POS * 4)
t_cpy  = TinyNNCuda.tnn_cpy(sess, t_src, t_slot)
TinyNNCuda.tnn_set_output(t_cpy)
TinyNNCuda.tnn_set_output(t_V)
TinyNNCuda.tnn_realize(sess, t_cpy)
TinyNNCuda.upload_row_major(sess, t_src, src)
TinyNNCuda.tnn_compute(sess)

v_out = TinyNNCuda.download_row_major(sess, t_V, D_HEAD, MAX_T)

puts "V buffer (d_head x max_T):"
k = 0
while k < D_HEAD
  print "  k=" + k.to_s + ":"
  t = 0
  while t < MAX_T
    print " " + v_out.flat[k * MAX_T + t].to_s
    t = t + 1
  end
  puts ""
  k = k + 1
end

puts "expected at column pos=" + POS.to_s + ": " +
     src.flat[0].to_s + " " + src.flat[1].to_s + " " +
     src.flat[2].to_s + " " + src.flat[3].to_s
puts "actual at column pos:"
k = 0
while k < D_HEAD
  print "  k=" + k.to_s + " -> " + v_out.flat[k * MAX_T + POS].to_s + "\n"
  k = k + 1
end
