# End-to-end DistilGPT2 inference demo — FFI persistent-graph variant.
#
# Same I/O contract as distilgpt2_demo.rb (reads data/prompt_ids.txt,
# greedy-generates N_NEW tokens, writes ids back), but the forward
# goes through GPT2FullForwardFFICacheCuda instead of the native Mat path.
#
# Fixed T_SEQ window with right-padding: realize once, then each step
# pads the current id sequence to T_SEQ and re-runs the full forward.
# The model is causal so logits at the actual sequence positions are
# unaffected by the padded zeros at positions ≥ len(ids). The waste is
# compute on padded positions; the KV-cache variant is the natural
# follow-up that eliminates it (and the per-step recompute).
#
# Workflow:
#   prep/tokens.py encode "Hello, my name is"
#   make distilgpt2_demo_ffi && ./distilgpt2_demo_ffi
#   prep/tokens.py decode

require_relative "lib/transformer"
require_relative "lib/gpt2"
require_relative "lib/gpt2_ffi_cuda"
require_relative "lib/gguf_load"
require_relative "lib/training"

VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6
CONTEXT  = 1024

N_NEW    = 8
T_SEQ    = 16     # prompt_len + N_NEW + small buffer
IDS_PATH = "data/prompt_ids.txt"
GGUF     = "data/distilgpt2-f32.gguf"

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

puts "reading prompt IDs from " + IDS_PATH
ids = read_ids(IDS_PATH)
puts "prompt length: " + ids.length.to_s + " tokens"
if ids.length + N_NEW > T_SEQ
  puts "WARNING: prompt(" + ids.length.to_s + ") + N_NEW(" + N_NEW.to_s +
       ") > T_SEQ(" + T_SEQ.to_s + "); increase T_SEQ."
end

puts ""
puts "constructing native GPT2LM (weight source)..."
t0 = Time.now
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
puts "  built in " + ((Time.now - t0) * 1000).to_s + " ms"

t0 = Time.now
ok = GGUFLoad.load_gpt2(model, GGUF)
if !ok
  puts "load failed; aborting"
else
  puts "  loaded GGUF in " + ((Time.now - t0) * 1000).to_s + " ms"

  puts ""
  puts "realizing FFI-CUDA graph at T_SEQ=" + T_SEQ.to_s + "..."
  t0 = Time.now
  cache = GPT2FullForwardFFICacheCuda.new
  cache.realize_for(T_SEQ, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB)
  puts "  realized in " + ((Time.now - t0) * 1000).to_s + " ms"

  puts "uploading weights..."
  t0 = Time.now
  pos_slice = GPT2FFICuda.make_pos_slice(model, T_SEQ)
  GPT2FFICuda.upload_from(cache, model, pos_slice)
  puts "  uploaded in " + ((Time.now - t0) * 1000).to_s + " ms"

  puts ""
  puts "generating " + N_NEW.to_s + " tokens (greedy, FFI)..."
  total_ms = 0.0
  step = 0
  while step < N_NEW
    cur_len = ids.length
    padded  = GPT2FFICuda.pad_ids(ids, T_SEQ)
    t_step  = Time.now
    logits  = GPT2FFICuda.forward(cache, padded)
    nxt     = argmax_row(logits, cur_len - 1, VOCAB)
    dt_ms   = (Time.now - t_step) * 1000.0
    total_ms = total_ms + dt_ms
    puts "  step " + (step + 1).to_s + " (cur_len=" + cur_len.to_s + ", T_SEQ=" +
         T_SEQ.to_s + "): id=" + nxt.to_s + "  (" + dt_ms.to_s + " ms)"
    ids.push(nxt)
    step = step + 1
  end

  write_ids(IDS_PATH, ids)
  puts ""
  puts "wrote " + ids.length.to_s + " ids to " + IDS_PATH
  puts "total generation time: " + total_ms.to_s + " ms  (" +
       (total_ms / N_NEW).to_s + " ms/token)"
  puts "decode with:  ./prep/tokens.py decode"
end
