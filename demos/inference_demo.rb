# End-to-end inference demo: builds a TransformerLM with random weights,
# uploads them into a FullForwardFFICache, generates N tokens with greedy
# sampling, and parity-checks against the native forward path.
#
# This is the building block for the user's stated end-state:
# (1) full accelerated forward -- done here (FullForwardFFICache),
# (2) HF weight loading -- next (M3),
# (3) tep single-binary packaging -- final (M4).
#
# Generation is autoregressive but NOT incremental (each step re-runs
# forward over the whole padded context). M2 (KV cache) is what makes
# this efficient at long contexts; correctness here is the prerequisite.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

VOCAB    = 16
D_MODEL  = 32
D_FF     = 64
N_HEADS  = 4
N_LAYERS = 2
CONTEXT  = 16
T_SEQ    = CONTEXT

srand(42)
model = TransformerLM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
model.token_embed.fill_random(0.1)
model.pos_embed.fill_random(0.05)
i = 0
while i < D_MODEL
  model.norm_final_gamma[i] = 1.0
  i = i + 1
end
li = 0
while li < N_LAYERS
  b = model.blocks[li]
  i = 0
  while i < D_MODEL
    b.norm1_gamma[i] = 1.0
    b.norm2_gamma[i] = 1.0
    i = i + 1
  end
  b.fill_random_all(0.1)
  li = li + 1
end

# === Brief training so generation is non-degenerate ===
# 20 SGD steps on a few hand-crafted sequences (same idiom as
# train_minimal). Without training, random weights make every
# argmax pick the same token; a bit of SGD breaks the symmetry
# and the FFI/native parity still holds.
seqs_a = [0, 1, 2, 3, 4]
seqs_b = [5, 6, 7, 8, 9]
seqs_c = [10, 11, 12, 13, 14]
grads = Gradients.new(VOCAB, D_MODEL, D_FF, N_HEADS, D_MODEL / N_HEADS, N_LAYERS, CONTEXT)
step = 0
while step < 20
  grads.fill_zero
  model.forward(seqs_a)
  model.backward(seqs_a, grads)
  model.apply_gradients_sgd(grads, 0.05)
  grads.fill_zero
  model.forward(seqs_b)
  model.backward(seqs_b, grads)
  model.apply_gradients_sgd(grads, 0.05)
  grads.fill_zero
  model.forward(seqs_c)
  model.backward(seqs_c, grads)
  model.apply_gradients_sgd(grads, 0.05)
  step = step + 1
end
puts "warm-trained 20 SGD steps on [0..4]/[5..9]/[10..14]"

# === FFI cache setup ===
cache = FullForwardFFICache.new
cache.realize_for(T_SEQ, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB)

TinyNN.upload_row_major(cache.sess, cache.t_token_embed, model.token_embed)
pos_slice = Mat.new(T_SEQ, D_MODEL)
i = 0
while i < T_SEQ * D_MODEL
  pos_slice.flat[i] = model.pos_embed.flat[i]
  i = i + 1
end
TinyNN.upload_row_major(cache.sess, cache.t_pos_slice, pos_slice)
TinyNN.tnn_upload_from_float_array(cache.sess, cache.t_final_norm_gamma,
                                    model.norm_final_gamma, D_MODEL)
li = 0
while li < N_LAYERS
  blk_n = model.blocks[li]
  blk_f = cache.blocks_ffi[li]
  TinyNN.tnn_upload_from_float_array(cache.sess, blk_f.t_norm1_gamma, blk_n.norm1_gamma, D_MODEL)
  TinyNN.tnn_upload_from_float_array(cache.sess, blk_f.t_norm2_gamma, blk_n.norm2_gamma, D_MODEL)
  h = 0
  while h < N_HEADS
    TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_q[h], blk_n.w_q[h])
    TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_k[h], blk_n.w_k[h])
    TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_v[h], blk_n.w_v[h])
    h = h + 1
  end
  TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_o,   blk_n.w_o)
  TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_ff1, blk_n.w_ff1)
  TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_ff2, blk_n.w_ff2)
  li = li + 1
end

# === Helpers ===
def pad_to_t(ids, t_seq)
  out = Array.new(t_seq, 0)
  i = 0
  n = ids.length
  if n > t_seq
    n = t_seq
  end
  while i < n
    out[i] = ids[i]
    i = i + 1
  end
  out
end

def argmax_row(logits, row, vocab)
  best   = 0
  best_v = logits.flat[row * vocab]
  v = 1
  while v < vocab
    val = logits.flat[row * vocab + v]
    if val > best_v
      best_v = val
      best   = v
    end
    v = v + 1
  end
  best
end

# === Generation: native ===
def generate_native(model, prompt, n_new, t_seq, vocab)
  ids = []
  i = 0
  while i < prompt.length
    ids.push(prompt[i])
    i = i + 1
  end
  step = 0
  while step < n_new
    padded = pad_to_t(ids, t_seq)
    logits = model.forward(padded)
    last   = ids.length - 1
    next_id = argmax_row(logits, last, vocab)
    ids.push(next_id)
    step = step + 1
  end
  ids
end

# === Generation: FFI ===
def generate_ffi(cache, prompt, n_new, t_seq, vocab)
  ids = []
  i = 0
  while i < prompt.length
    ids.push(prompt[i])
    i = i + 1
  end
  step = 0
  while step < n_new
    padded = pad_to_t(ids, t_seq)
    TinyNN.upload_int_array(cache.sess, cache.t_token_ids, padded)
    TinyNN.tnn_compute(cache.sess)
    logits = TinyNN.download_row_major(cache.sess, cache.t_logits, t_seq, vocab)
    last    = ids.length - 1
    next_id = argmax_row(logits, last, vocab)
    ids.push(next_id)
    step = step + 1
  end
  ids
end

# === Run ===
prompt = [3, 7, 1]
n_new  = 5

t0 = Time.now
nat_ids = generate_native(model, prompt, n_new, T_SEQ, VOCAB)
t1 = Time.now
nat_ms = (t1 - t0) * 1000.0

t0 = Time.now
ffi_ids = generate_ffi(cache, prompt, n_new, T_SEQ, VOCAB)
t1 = Time.now
ffi_ms = (t1 - t0) * 1000.0

puts "Inference demo (greedy, n_new=" + n_new.to_s + " at T=" + T_SEQ.to_s + ")"
puts "  prompt:    " + prompt.to_s
puts "  native:    " + nat_ids.to_s + "  (" + nat_ms.to_s + " ms)"
puts "  ffi:       " + ffi_ids.to_s + "  (" + ffi_ms.to_s + " ms)"
match = true
i = 0
while i < nat_ids.length
  if nat_ids[i] != ffi_ids[i]
    match = false
  end
  i = i + 1
end
puts "  parity:    " + (match ? "OK" : "FAIL")
TinyNN.tnn_session_free(cache.sess)
