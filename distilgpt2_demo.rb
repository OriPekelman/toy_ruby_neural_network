# End-to-end DistilGPT2 inference demo.
#
# Reads pre-tokenized prompt IDs from data/prompt_ids.txt, loads
# distilgpt2-f32.gguf into a GPT2LM, runs greedy generation for
# N_NEW tokens (autoregressive, re-runs full forward each step),
# writes the full ID sequence back to data/prompt_ids.txt for
# external decoding.
#
# Workflow:
#   prep/tokens.py encode "Hello, my name is"
#   make distilgpt2_demo && ./distilgpt2_demo
#   prep/tokens.py decode
#
# Native Mat forward — no FFI yet. Slow at distilgpt2 shape (several
# seconds per token); the FFI persistent-graph version is M3 step 8.

require_relative "lib/transformer"
require_relative "lib/gpt2"
require_relative "lib/gguf_load"
require_relative "lib/training"   # for parse_ids / File reading patterns

# distilgpt2 hyperparams. Match what convert_distilgpt2_to_gguf.py wrote.
VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6
CONTEXT  = 1024

N_NEW    = 8     # number of tokens to greedy-generate
IDS_PATH = "data/prompt_ids.txt"
GGUF     = "data/distilgpt2-f32.gguf"

# Read whitespace-separated int IDs from a single-line file.
def read_ids(path)
  raw = ["?"]
  raw.pop
  File.open(path, "r") do |f|
    f.each_line { |line| raw.push(line.chomp) }
  end
  parse_ids(raw[0])   # parse_ids comes from lib/training.rb
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

puts "reading prompt IDs from " + IDS_PATH
ids = read_ids(IDS_PATH)
puts "prompt length: " + ids.length.to_s + " tokens"
puts "prompt: " + ids[0].to_s
i = 1
while i < ids.length
  puts "        " + ids[i].to_s
  i = i + 1
end

puts ""
puts "constructing GPT2LM (vocab=" + VOCAB.to_s + " d=" + D_MODEL.to_s +
     " heads=" + N_HEADS.to_s + " layers=" + N_LAYERS.to_s + ")"
t0 = Time.now
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
puts "  built in " + ((Time.now - t0) * 1000).to_s + " ms"

t0 = Time.now
ok = GGUFLoad.load_gpt2(model, GGUF)
if !ok
  puts "load failed; aborting"
else
  puts "loaded in " + ((Time.now - t0) * 1000).to_s + " ms"

  puts ""
  puts "generating " + N_NEW.to_s + " tokens (greedy)"
  step = 0
  while step < N_NEW
    t_step = Time.now
    logits = model.forward(ids, 0)
    nxt = argmax_row(logits, ids.length - 1, VOCAB)
    dt  = ((Time.now - t_step) * 1000).to_s
    puts "  step " + (step + 1).to_s + " (T=" + ids.length.to_s + " → " +
         (ids.length + 1).to_s + "): id=" + nxt.to_s + "  (" + dt + " ms)"
    ids.push(nxt)
    step = step + 1
  end

  write_ids(IDS_PATH, ids)
  puts ""
  puts "wrote " + ids.length.to_s + " ids back to " + IDS_PATH
  puts "decode with:  ./prep/tokens.py decode"
end
