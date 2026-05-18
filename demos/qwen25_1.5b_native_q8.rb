# demos/qwen25_direct_native_q8.rb — Qwen2.5-1.5B with Q8 weights kept
# Q8 in memory (Phase 3 of the memory-design plan).
#
# Companion to demos/qwen25_direct_native.rb (which keeps the same
# native layout but materializes weights as f32 at load). Here the
# persistent ggml tensors are allocated as Q8_0; matmul auto-dispatches
# to Q8 kernels for mixed activation-f32 × weight-Q8.
#
# Expected RAM: ~25% of the f32 path for the 2D linear weights.
# Expected output: continuation may differ by a token or two from f32
# due to Q8 quantization noise (well-documented at ~1%).

require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"

GGUF  = "data/qwen25-1.5b-native-q8.gguf"
MAX_T = 256
N_NEW = 8

ids = [9707, 11, 847, 829, 374]

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)
wtype = GGUFLoad.detect_weight_type(GGUF)
puts "flags: untied=" + flags.untied.to_s +
     " qkv_bias=" + flags.qkv_bias.to_s +
     " weight_type=" + wtype.to_s

kv = SmolLM2KVFFICache.new
kv.set_weight_type(wtype)
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)

t0 = Time.now
kv.load_weights(GGUF)
puts "  loaded weights (native, type=" + wtype.to_s + ") in " +
     ((Time.now - t0) * 1000.0).to_s + " ms"

puts "prefilling " + ids.length.to_s + " prompt tokens..."
i = 0
while i < ids.length
  SmolLM2KV.decode_step(kv, ids[i], i)
  i = i + 1
end

puts "generating " + N_NEW.to_s + " tokens..."
n = 0
while n < N_NEW
  pos = ids.length
  last_id = ids[pos - 1]
  logits = SmolLM2KV.decode_step(kv, last_id, pos)
  best_i = 0
  best_v = logits.flat[0]
  j = 1
  while j < cfg.vocab
    v = logits.flat[j]
    if v > best_v
      best_v = v
      best_i = j
    end
    j = j + 1
  end
  if n == 0
    puts "  step 0: top index=" + best_i.to_s + " val=" + best_v.to_s
  end
  ids.push(best_i)
  n = n + 1
end

print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
