# demos/llama32_1b_direct.rb — Llama-3.2-1B probe via the direct GGUF→FFI loader.
#
# Goal: surface the "L=11 vs L=16" report from docs/handoff.md.
# Uses a synthetic prompt (Array of small int IDs) — no tokenizer needed.

require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/training"

GGUF  = "data/llama-3.2-1b-f32.gguf"
MAX_T = 64
N_NEW = 4

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " rope_base=" + cfg.rope_base.to_s +
     " ctx=" + cfg.context.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "flags: untied=" + flags.untied.to_s + " qkv_bias=" + flags.qkv_bias.to_s

puts "realizing KV cache..."
kv = SmolLM2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)
puts "  kv.n_layers=" + kv.n_layers.to_s + " kv.n_heads=" + kv.n_heads.to_s + " kv.n_kv=" + kv.n_kv.to_s

t0 = Time.now
GGUFLoad.load_kv_cache_directly(kv, GGUF)
puts "  loaded weights in " + ((Time.now - t0) * 1000.0).to_s + " ms"

# Synthetic prompt: 4 small BOS-ish IDs (not semantically meaningful;
# we just want to surface any layer-count assertion in the C/Spinel path).
ids = [128000, 9906, 11, 856]
puts "prefilling " + ids.length.to_s + " synthetic prompt tokens..."
i = 0
while i < ids.length
  SmolLM2KV.decode_step(kv, ids[i], i)
  i = i + 1
end
puts "prefill OK"

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
  puts "  step " + n.to_s + " → id=" + best_i.to_s + " val=" + best_v.to_s
  ids.push(best_i)
  n = n + 1
end
