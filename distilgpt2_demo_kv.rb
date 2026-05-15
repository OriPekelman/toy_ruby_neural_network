# End-to-end DistilGPT2 inference demo — KV cache variant.
#
# Same I/O contract as the other distilgpt2_demo_*.rb:
#   prep/tokens.py encode "Hello, my name is"
#   ./distilgpt2_demo_kv
#   prep/tokens.py decode
#
# Per-step cost is constant in prompt length (vs full-forward which
# grows linearly in T_SEQ). Pre-fill walks the prompt token-by-token,
# then generation continues with the running last-token argmax.

require_relative "lib/transformer"
require_relative "lib/gpt2"
require_relative "lib/gpt2_ffi_kv"
require_relative "lib/gguf_load"
require_relative "lib/training"

N_NEW    = 30
MAX_T    = 1024   # KV-cache capacity. GPT-2 / DistilGPT2 max context.
                  # Per-head buffer: 2 × d_head × max_T floats (~128 KB
                  # per head at max_T=1024, d_head=64). Total KV
                  # footprint at gpt2-small: 12 layers × 12 heads × 2 ×
                  # 64 × 1024 × 4 B ≈ 72 MB on top of the weights.
IDS_PATH = "data/prompt_ids.txt"
# Swap this to point at a different GPT-2 variant (data/gpt2-f32.gguf,
# data/gpt2-medium-f32.gguf, …) — the hyperparams are read from the
# file's kv metadata so the rest of the code adapts automatically.
GGUF     = "data/gpt2-f32.gguf"

def read_ids(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])
end

def write_ids(path, ids)
  File.open(path, "w") do |f|
    n = ids.length
    i = 0
    while i < n
      f.write(ids[i].to_s)
      if i < n - 1
        f.write(" ")
      end
      i = i + 1
    end
    f.write("\n")
  end
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

ids = read_ids(IDS_PATH)
puts "prompt: " + ids.length.to_s + " tokens"

puts ""
puts "reading config from " + GGUF
cfg = GPT2ConfigLoader.read(GGUF)
puts "  vocab=" + cfg.vocab_size.to_s + " d=" + cfg.d_model.to_s +
     " d_ff=" + cfg.d_ff.to_s + " heads=" + cfg.n_heads.to_s +
     " layers=" + cfg.n_layers.to_s + " ctx=" + cfg.context_length.to_s
puts ""
puts "constructing native GPT2LM (weight source)..."
model = GPT2LM.new(cfg.vocab_size, cfg.d_model, cfg.d_ff,
                    cfg.n_heads, cfg.n_layers, cfg.context_length)
ok = GGUFLoad.load_gpt2(model, GGUF)
if !ok
  puts "load failed"
else
  puts "  native loaded"
  puts ""
  puts "realizing KV cache (MAX_T=" + MAX_T.to_s + ")..."
  cache = GPT2KVFFICache.new
  cache.realize_for(MAX_T, cfg.d_model, cfg.d_ff,
                     cfg.n_heads, cfg.n_layers, cfg.vocab_size,
                     cfg.context_length)
  puts "uploading weights..."
  t0 = Time.now
  GPT2KV.upload_from(cache, model)
  puts "  uploaded: " + ((Time.now - t0) * 1000).to_s + " ms"

  # Prefill — run each prompt token through the cache, keep last logits.
  puts ""
  puts "prefilling " + ids.length.to_s + " prompt tokens..."
  prefill_ms = 0.0
  last_logits = Mat.new(1, cfg.vocab_size)
  pos = 0
  while pos < ids.length
    t_step = Time.now
    last_logits = GPT2KV.decode_step(cache, ids[pos], pos)
    prefill_ms = prefill_ms + (Time.now - t_step) * 1000.0
    pos = pos + 1
  end
  puts "  prefill: " + prefill_ms.to_s + " ms  (" +
       (prefill_ms / ids.length).to_s + " ms/token)"

  # First generated token comes from the LAST prefill step's logits.
  next_id = argmax_row(last_logits, 0, cfg.vocab_size)
  ids.push(next_id)
  puts ""
  puts "generating " + N_NEW.to_s + " tokens..."
  puts "  step 1: id=" + next_id.to_s + "  (prefill logits, no extra compute)"

  gen_ms = 0.0
  step = 1
  while step < N_NEW
    t_step = Time.now
    last_logits = GPT2KV.decode_step(cache, ids[ids.length - 1], ids.length - 1)
    dt = (Time.now - t_step) * 1000.0
    gen_ms = gen_ms + dt
    next_id = argmax_row(last_logits, 0, cfg.vocab_size)
    ids.push(next_id)
    puts "  step " + (step + 1).to_s + ": id=" + next_id.to_s + "  (" + dt.to_s + " ms)"
    step = step + 1
  end

  write_ids(IDS_PATH, ids)
  puts ""
  puts "wrote " + ids.length.to_s + " ids back to " + IDS_PATH
  puts "generation total: " + gen_ms.to_s + " ms  (" +
       (gen_ms / (N_NEW - 1)).to_s + " ms/token across " + (N_NEW - 1).to_s + " new compute steps)"
  puts "decode with:  ./prep/tokens.py decode"
end
