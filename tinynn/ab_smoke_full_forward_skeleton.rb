# Parity check for FullForwardFFICache's minimal skeleton (embed +
# pos_embed + tied unembed; blocks come next). Computes the same
# numbers natively and verifies bitwise-close agreement.
#
# Native math:
#   x[i,d]      = token_embed[ids[i], d] + pos_embed[i, d]
#   logits[i,v] = sum_d x[i,d] * token_embed[v, d]    (tied unembed)

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

vocab_size = 5
d_model    = 4
t_seq      = 3

token_embed = Mat.new(vocab_size, d_model)
pos_slice   = Mat.new(t_seq,      d_model)

i = 0
while i < vocab_size * d_model
  token_embed.flat[i] = i.to_f * 0.05 - 0.4
  i = i + 1
end
i = 0
while i < t_seq * d_model
  pos_slice.flat[i] = i.to_f * 0.02 - 0.1
  i = i + 1
end

ids = [2, 0, 4]

# Native x and logits.
nat_x = Mat.new(t_seq, d_model)
i = 0
while i < t_seq
  tok = ids[i]
  j = 0
  while j < d_model
    nat_x.flat[i * d_model + j] =
      token_embed.flat[tok * d_model + j] + pos_slice.flat[i * d_model + j]
    j = j + 1
  end
  i = i + 1
end

nat_logits = Mat.new(t_seq, vocab_size)
i = 0
while i < t_seq
  v = 0
  while v < vocab_size
    s = 0.0
    d = 0
    while d < d_model
      s = s + nat_x.flat[i * d_model + d] * token_embed.flat[v * d_model + d]
      d = d + 1
    end
    nat_logits.flat[i * vocab_size + v] = s
    v = v + 1
  end
  i = i + 1
end

# FFI path.
cache = FullForwardFFICache.new
cache.realize_for(t_seq, d_model, vocab_size)
TinyNN.upload_row_major(cache.sess, cache.t_token_embed, token_embed)
TinyNN.upload_row_major(cache.sess, cache.t_pos_slice,   pos_slice)
TinyNN.upload_int_array(cache.sess, cache.t_token_ids,   ids)
TinyNN.tnn_compute(cache.sess)

# t_x has ne=[d_model, T]; logical Mat(T, d_model) reads as a straight
# memcpy (data[k] = logical[k/d_model][k%d_model]).
ffi_x      = TinyNN.download_row_major(cache.sess, cache.t_x,      t_seq, d_model)
# t_logits has ne=[vocab, T]; logical Mat(T, vocab) is again a straight
# memcpy.
ffi_logits = TinyNN.download_row_major(cache.sess, cache.t_logits, t_seq, vocab_size)

def cmp(label, a, b, tol)
  max_d = 0.0
  n = a.nrows * a.ncols
  i = 0
  while i < n
    d = a.flat[i] - b.flat[i]
    if d < 0
      d = -d
    end
    if d > max_d
      max_d = d
    end
    i = i + 1
  end
  ok = max_d < tol
  puts label + " max_abs_diff=" + max_d.to_s + " " + (ok ? "OK" : "FAIL")
  ok
end

ok1 = cmp("x (embed+pos)", nat_x, ffi_x, 1.0e-5)
ok2 = cmp("logits (tied)", nat_logits, ffi_logits, 1.0e-4)

puts "M1.1 SKELETON OK: " + (ok1 && ok2).to_s
TinyNN.tnn_session_free(cache.sess)
