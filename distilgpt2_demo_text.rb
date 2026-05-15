# Fully self-contained DistilGPT2 / GPT-2 inference demo.
#
# Reads a plain-text prompt from data/prompt.txt, encodes it with the
# Ruby BPE tokenizer (lib/bpe.rb), generates N_NEW tokens through the
# KV-cache FFI path, decodes back to text, and prints the result.
# No Python at runtime — only the GGUF model file + the three BPE
# TSV tables (produced once by prep/dump_bpe.py).
#
# Usage:
#   echo "Once upon a time" > data/prompt.txt
#   make distilgpt2_demo_text && ./distilgpt2_demo_text

require_relative "lib/transformer"
require_relative "lib/gpt2"
require_relative "lib/gpt2_ffi_kv"
require_relative "lib/gguf_load"
require_relative "lib/bpe"
require_relative "lib/training"

N_NEW       = 30
MAX_T       = 1024
PROMPT_PATH = "data/prompt.txt"
GGUF        = "data/gpt2-f32.gguf"
BPE_DIR     = "data"

def read_text(path)
  out = ""
  File.open(path, "r") do |f|
    f.each_line do |line|
      out = out + line
    end
  end
  # strip trailing newline so the prompt doesn't end with one (matches
  # what tokens.py does for fairer parity).
  if out.length > 0 && out[out.length - 1] == "\n"
    out = out[0, out.length - 1]
  end
  out
end

def argmax_row(logits, vocab)
  best   = 0
  best_v = logits.flat[0]
  v = 1
  while v < vocab
    val = logits.flat[v]
    if val > best_v
      best_v = val
      best   = v
    end
    v = v + 1
  end
  best
end

puts "loading BPE tables..."
t0 = Time.now
bpe = GPT2BPE.load(BPE_DIR)
puts "  loaded in " + ((Time.now - t0) * 1000).to_s + " ms"

puts ""
puts "reading prompt from " + PROMPT_PATH
prompt_text = read_text(PROMPT_PATH)
puts "  prompt: " + prompt_text.inspect
ids = GPT2BPE.encode(prompt_text, bpe)
puts "  " + ids.length.to_s + " tokens"

puts ""
puts "reading model config from " + GGUF
cfg = GPT2ConfigLoader.read(GGUF)
puts "  vocab=" + cfg.vocab_size.to_s + " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s + " layers=" + cfg.n_layers.to_s

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
  t0 = Time.now
  GPT2KV.upload_from(cache, model)
  puts "  uploaded weights in " + ((Time.now - t0) * 1000).to_s + " ms"

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
  puts "  prefill: " + prefill_ms.to_s + " ms (" +
       (prefill_ms / ids.length).to_s + " ms/token)"

  next_id = argmax_row(last_logits, cfg.vocab_size)
  ids.push(next_id)

  puts "generating " + N_NEW.to_s + " tokens..."
  gen_ms = 0.0
  step = 1
  while step < N_NEW
    t_step = Time.now
    last_logits = GPT2KV.decode_step(cache, ids[ids.length - 1], ids.length - 1)
    dt = (Time.now - t_step) * 1000.0
    gen_ms = gen_ms + dt
    next_id = argmax_row(last_logits, cfg.vocab_size)
    ids.push(next_id)
    step = step + 1
  end
  puts "  generation total: " + gen_ms.to_s + " ms  (" +
       (gen_ms / (N_NEW - 1)).to_s + " ms/token)"

  full_text = GPT2BPE.decode(ids, bpe)
  puts ""
  puts "=== output ==="
  puts full_text
end
