# DistilGPT2 forward bench: native Mat vs FFI persistent-graph,
# apples-to-apples on the same prompt and same T_SEQ.
#
# Reports min/avg/max ms per forward + tokens/sec. The reported
# "tokens/sec" is the autoregressive-decode rate assuming the same
# T_SEQ each step (no KV cache); a KV-cache version would be
# substantially faster.

require_relative "../lib/transformer"
require_relative "../lib/gpt2"
require_relative "../lib/gpt2_ffi"
require_relative "../lib/gguf_load"
require_relative "../lib/training"

VOCAB    = 50257
D_MODEL  = 768
D_FF     = 3072
N_HEADS  = 12
N_LAYERS = 6
T_SEQ    = 5

N_NATIVE = 3       # native is slow; small N
N_FFI    = 30      # FFI is fast; bigger N for tighter stats
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

def stats_print(label, times_ms)
  n = times_ms.length
  if n == 0
    puts label + ": no samples"
    return
  end
  sum = 0.0
  mn  = times_ms[0]
  mx  = times_ms[0]
  i = 0
  while i < n
    v = times_ms[i]
    sum = sum + v
    if v < mn
      mn = v
    end
    if v > mx
      mx = v
    end
    i = i + 1
  end
  avg = sum / n
  tok_per_sec = 1000.0 / avg
  puts label + ":"
  puts "  n=" + n.to_s + "  avg=" + avg.to_s + " ms  min=" + mn.to_s +
       "  max=" + mx.to_s
  puts "  → " + tok_per_sec.to_s + " forwards/sec  (single-position decode)"
end

ids = read_ids(IDS_PATH)
puts "prompt: " + ids.length.to_s + " tokens  (T_SEQ=" + T_SEQ.to_s + ")"
if ids.length != T_SEQ
  puts "WARNING: prompt length " + ids.length.to_s + " != T_SEQ " + T_SEQ.to_s +
       "; re-encode with the right prompt length or change T_SEQ."
end

puts ""
puts "loading native model + FFI cache..."
t0 = Time.now
model = GPT2LM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, 1024)
GGUFLoad.load_gpt2(model, GGUF)
puts "  native ready: " + ((Time.now - t0) * 1000).to_s + " ms"

t0 = Time.now
cache = GPT2FullForwardFFICache.new
cache.realize_for(T_SEQ, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB)
pos_slice = GPT2FFI.make_pos_slice(model, T_SEQ)
GPT2FFI.upload_from(cache, model, pos_slice)
puts "  FFI ready:    " + ((Time.now - t0) * 1000).to_s + " ms"

# Warmup one of each.
puts ""
puts "warmup..."
t0 = Time.now
_ = model.forward(ids, 0)
puts "  native warmup: " + ((Time.now - t0) * 1000).to_s + " ms"
t0 = Time.now
_ = GPT2FFI.forward(cache, ids)
puts "  FFI    warmup: " + ((Time.now - t0) * 1000).to_s + " ms"

puts ""
puts "benching native (" + N_NATIVE.to_s + " forwards)..."
native_ms = [0.0]
native_ms.pop
i = 0
while i < N_NATIVE
  iter_start = Time.now
  _ = model.forward(ids, 0)
  native_ms.push((Time.now - iter_start) * 1000.0)
  i = i + 1
end

puts "benching FFI (" + N_FFI.to_s + " forwards)..."
ffi_ms = [0.0]
ffi_ms.pop
i = 0
while i < N_FFI
  iter_start = Time.now
  _ = GPT2FFI.forward(cache, ids)
  ffi_ms.push((Time.now - iter_start) * 1000.0)
  i = i + 1
end

puts ""
stats_print("native (Mat)", native_ms)
stats_print("FFI (ggml-cpu)", ffi_ms)

# Speedup ratio (using means).
nsum = 0.0
fsum = 0.0
i = 0
while i < native_ms.length
  nsum = nsum + native_ms[i]
  i = i + 1
end
i = 0
while i < ffi_ms.length
  fsum = fsum + ffi_ms[i]
  i = i + 1
end
n_avg = nsum / native_ms.length
f_avg = fsum / ffi_ms.length
puts ""
puts "speedup: " + (n_avg / f_avg).to_s + "x"
