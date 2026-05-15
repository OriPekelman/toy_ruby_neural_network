# demos/gpt2_pleasant.rb — DistilGPT2 inference using the Toy:: sugar.
#
# Mirrors distilgpt2_demo.rb but built on top of Toy::GPT2 (lib/toy_gpt2.rb).
# The whole model fits in ~20 lines of Ruby; this demo glues
# GGUF-load → forward → greedy-argmax → repeat.
#
# Reads pre-tokenized prompt IDs from data/prompt_ids.txt (see
# prep/tokens.py for the producer side). Writes the full ID sequence
# back so prep/tokens.py can decode it.

require_relative "../lib/toy"
require_relative "../lib/toy_gpt2"
require_relative "../lib/toy_gpt2_loader"
require_relative "../lib/training"   # parse_ids: comma- or space-separated int line

# --- config from GGUF ----
GGUF = "data/distilgpt2-f32.gguf"
ext  = GPT2ConfigLoader.read(GGUF)
cfg  = Toy::GPT2Config.new(ext.vocab_size, ext.d_model, ext.n_heads,
                           ext.d_ff, ext.n_layers, ext.context_length)
puts "config: v=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " h=" + cfg.n_heads.to_s + " L=" + cfg.n_layers.to_s

# --- build + load ----
model = Toy::GPT2.new(cfg)
GGUFLoad.load_toy_gpt2(model, GGUF)

# --- read prompt ids ----
raw = ["?"]
raw.pop
File.open("data/prompt_ids.txt", "r") do |f|
  f.each_line { |line| raw.push(line.chomp) }
end
ids = parse_ids(raw[0])
puts "prompt: " + ids.length.to_s + " tokens"

# --- greedy generation ----
N_NEW = 8
gen = 0
t0  = Time.now
while gen < N_NEW
  logits = model.forward(ids, 0)               # [T, V]
  last   = logits.nrows - 1
  base   = last * logits.ncols
  # argmax over last row
  best_i = 0
  best_v = logits.flat[base]
  j = 1
  while j < logits.ncols
    v = logits.flat[base + j]
    if v > best_v
      best_v = v
      best_i = j
    end
    j += 1
  end
  ids.push(best_i)
  gen += 1
end
dt = ((Time.now - t0) * 1000.0).round(1)
puts "generated " + N_NEW.to_s + " tokens in " + dt.to_s + " ms"

# --- write back ----
File.open("data/prompt_ids.txt", "w") do |out|
  n = ids.length
  k = 0
  while k < n
    out.write(ids[k].to_s)
    if k < n - 1
      out.write(" ")
    end
    k += 1
  end
  out.write("\n")
end
puts "wrote data/prompt_ids.txt"
