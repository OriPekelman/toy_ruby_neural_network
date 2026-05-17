# demos/qwen25_direct.rb — Qwen2.5-0.5B via the direct GGUF→FFI loader.
#
# Bypasses the Ruby Mat / Float64 intermediate. Required for 7B-class
# models; verified end-to-end here at 0.5B by argmax-comparing the
# generation against the Mat-mediated qwen25_kv demo (same prompt,
# same first 16 tokens, same decoded string).
#
# Memory: 4 B/w instead of 12 B/w (see docs/memory-design.md).

require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/training"

GGUF     = "data/qwen25-7b-q8_0.gguf"
IDS_PATH = "data/qwen25_prompt_ids.txt"
MAX_T    = 256
N_NEW    = 16

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s +
     " rope_base=" + cfg.rope_base.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "flags: untied=" + flags.untied.to_s + " qkv_bias=" + flags.qkv_bias.to_s

# --- realize KV cache + load weights directly from GGUF ---
puts "realizing KV cache (MAX_T=" + MAX_T.to_s + ")..."
kv = SmolLM2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)
t0 = Time.now
GGUFLoad.load_kv_cache_directly(kv, GGUF)
puts "  loaded weights in " + ((Time.now - t0) * 1000.0).to_s + " ms"

# --- read prompt ---
raw = ["?"]
raw.pop
File.open(IDS_PATH, "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])

# --- prefill ---
puts "prefilling " + ids.length.to_s + " prompt tokens..."
t0 = Time.now
i = 0
while i < ids.length
  SmolLM2KV.decode_step(kv, ids[i], i)
  i = i + 1
end
prefill_ms = (Time.now - t0) * 1000.0
puts "  prefill: " + prefill_ms.to_s + " ms (" +
     (prefill_ms / ids.length.to_f).to_s + " ms/token)"

# --- generation ---
puts "generating " + N_NEW.to_s + " tokens..."
t0 = Time.now
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
gen_ms = (Time.now - t0) * 1000.0
puts "  generation total: " + gen_ms.to_s + " ms  (" +
     (gen_ms / N_NEW.to_f).to_s + " ms/token)"

File.open(IDS_PATH, "w") do |out|
  n = ids.length
  k = 0
  while k < n
    out.write(ids[k].to_s)
    if k < n - 1
      out.write(" ")
    end
    k = k + 1
  end
  out.write("\n")
end
puts "wrote " + IDS_PATH
