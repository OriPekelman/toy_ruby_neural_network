# KV-cache parity probe. Builds a GPT2KVFFICacheCuda, runs the prompt
# through one-token-at-a-time, dumps the last-position logits to
# data/ours_kv_cuda_logits.txt. Pair with prep/parity.py compare --ours
# data/ours_kv_cuda_logits.txt to verify the KV path matches HF.

require_relative "../lib/transformer"
require_relative "../lib/gpt2"
require_relative "../lib/gpt2_ffi_kv_cuda"
require_relative "../lib/gguf_load"
require_relative "../lib/training"

VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6
CONTEXT  = 1024
MAX_T    = 32          # KV-cache capacity (≥ prompt + generation)
IDS_PATH = "data/prompt_ids.txt"
GGUF     = "data/distilgpt2-f32.gguf"
OUT      = "data/ours_kv_cuda_logits.txt"

def read_ids(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])
end

ids = read_ids(IDS_PATH)
puts "prompt: " + ids.length.to_s + " tokens"

puts "constructing native GPT2LM (weight source)..."
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
ok = GGUFLoad.load_gpt2(model, GGUF)
if !ok
  puts "load failed"
else
  puts "  native loaded"

  puts ""
  puts "realizing KV-CUDA cache (MAX_T=" + MAX_T.to_s + ")..."
  t0 = Time.now
  cache = GPT2KVFFICacheCuda.new
  cache.realize_for(MAX_T, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB, CONTEXT)
  puts "  realized: " + ((Time.now - t0) * 1000).to_s + " ms"

  puts "uploading weights..."
  t0 = Time.now
  GPT2KVCuda.upload_from(cache, model)
  puts "  uploaded: " + ((Time.now - t0) * 1000).to_s + " ms"

  puts ""
  puts "prefilling (one token at a time)..."
  last_logits = Mat.new(1, VOCAB)
  pos = 0
  total_ms = 0.0
  while pos < ids.length
    t_step = Time.now
    last_logits = GPT2KVCuda.decode_step(cache, ids[pos], pos)
    dt = (Time.now - t_step) * 1000.0
    total_ms = total_ms + dt
    puts "  pos=" + pos.to_s + "  tok=" + ids[pos].to_s + "  " + dt.to_s + " ms"
    pos = pos + 1
  end
  puts "prefill total: " + total_ms.to_s + " ms"

  # Dump last-position logits (post-prompt). This is what argmax-es to
  # the next token in greedy generation — same point as the
  # parity-test artefact for native / full-forward.
  File.open(OUT, "w") do |f|
    v = 0
    while v < VOCAB
      f.write(last_logits.flat[v].to_s)
      if v < VOCAB - 1
        f.write(" ")
      end
      v = v + 1
    end
    f.write("\n")
  end
  puts ""
  puts "wrote " + VOCAB.to_s + " logits to " + OUT
  puts "compare with: ./prep/parity.py compare --ours " + OUT

  best   = 0
  best_v = last_logits.flat[0]
  v = 1
  while v < VOCAB
    val = last_logits.flat[v]
    if val > best_v
      best_v = val
      best   = v
    end
    v = v + 1
  end
  puts "KV-CUDA argmax: id=" + best.to_s + "  logit=" + best_v.to_s
end
