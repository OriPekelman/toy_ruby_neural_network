# CPU half of the cpy-into-strided-view test. Mirror of the V cache
# write pattern: cpy a 1D source into a strided slot inside a 2D
# persistent buffer. Read back and dump the slot's bytes.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

MAX_T  = 8
D_HEAD = 4
POS    = 3

# Known source values.
src = Mat.new(1, D_HEAD)
src.flat[0] = 1.0
src.flat[1] = 2.0
src.flat[2] = 3.0
src.flat[3] = 4.0

sess = TinyNN.tnn_session_new(0)

# Persistent V-cache-shaped buffer: ne=[max_T, d_head] via
# tnn_input_2d_f32_persistent(sess, rows=d_head, cols=max_T).
t_V = TinyNN.tnn_input_2d_f32_persistent(sess, D_HEAD, MAX_T)
TinyNN.tnn_finalize_weights(sess)

# Compute-side source: ne=[1, d_head] (matches the V matmul result
# shape from the CUDA-class build_decode_step).
t_src = TinyNN.tnn_input_2d_f32(sess, D_HEAD, 1)

# View into V at column pos: ne=[1, d_head] strided.
t_slot = TinyNN.tnn_view_2d(sess, t_V, 1, D_HEAD, MAX_T * 4, POS * 4)
t_cpy  = TinyNN.tnn_cpy(sess, t_src, t_slot)
TinyNN.tnn_set_output(t_cpy)
TinyNN.tnn_set_output(t_V)
TinyNN.tnn_realize(sess, t_cpy)
TinyNN.upload_row_major(sess, t_src, src)
TinyNN.tnn_compute(sess)

# Read back the whole V buffer.
v_out = TinyNN.download_row_major(sess, t_V, D_HEAD, MAX_T)

# V's storage: ne=[max_T, d_head], so download_row_major(rows=d_head,
# cols=max_T) gives us numpy-shape (d_head, max_T) row-major.
# Element at (k_dhead, t_pos) is at flat[k * max_T + t].
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

# The "interesting" column is POS — what got written by the cpy.
puts "expected at column pos=" + POS.to_s + ": " +
     src.flat[0].to_s + " " + src.flat[1].to_s + " " +
     src.flat[2].to_s + " " + src.flat[3].to_s
puts "actual at column pos:"
k = 0
while k < D_HEAD
  print "  k=" + k.to_s + " -> " + v_out.flat[k * MAX_T + POS].to_s + "\n"
  k = k + 1
end
