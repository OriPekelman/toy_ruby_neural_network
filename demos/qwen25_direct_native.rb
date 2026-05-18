# demos/qwen25_direct_native.rb — Qwen2.5-0.5B via the native-layout
# direct loader (--ggml-native GGUFs).
#
# Parity probe for the converter-transpose-flip work. Bytes match
# hypothesis: a GGUF written with --ggml-native, loaded via
# GGUFLoad.load_kv_cache_directly_native (plain memcpy, no transpose),
# should produce the SAME output IDs as the legacy-transposed GGUF
# loaded via load_kv_cache_directly.
#
# Run this and the existing demos/qwen25_direct.rb with the same prompt;
# the first-step top index/val + final sequence must match.

require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"

GGUF  = "data/qwen25-0.5b-native.gguf"
MAX_T = 256
N_NEW = 8

# Hardcoded prompt: "Hello, my name is" (Qwen2.5 tokenizer).
ids = [9707, 11, 847, 829, 374]

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "flags: untied=" + flags.untied.to_s + " qkv_bias=" + flags.qkv_bias.to_s

kv = SmolLM2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)

t0 = Time.now
GGUFLoad.load_kv_cache_directly_native(kv, GGUF)
puts "  loaded weights (native) in " + ((Time.now - t0) * 1000.0).to_s + " ms"

# Prefill
puts "prefilling " + ids.length.to_s + " prompt tokens..."
i = 0
while i < ids.length
  SmolLM2KV.decode_step(kv, ids[i], i)
  i = i + 1
end

# Generate N_NEW tokens, greedy
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
