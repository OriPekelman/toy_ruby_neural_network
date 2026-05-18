# demos/qwen25_direct_cuda_standard.rb — control test for CUDA bug.
# Same model + prompt as qwen25_direct_native_mmap_cuda.rb but uses
# the STANDARD CUDA path (realize_for + load_weights via Mat-mediated
# upload) instead of BYO-pointer mmap. If THIS produces correct
# output, the bug is in our mmap path. If THIS also produces wrong
# output, the bug is in our binary's CUDA setup (e.g., archive
# linkage or env).

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF  = "data/qwen25-1.5b-f32.gguf"
MAX_T = 256
N_NEW = 8

ids = [9707, 11, 847, 829, 374]

cfg = SmolLM2ConfigLoader.read(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s +
     " L=" + cfg.n_layers.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "flags: untied=" + flags.untied.to_s + " qkv_bias=" + flags.qkv_bias.to_s

kv = SmolLM2KVFFICacheCuda.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)
puts "backend: " + TinyNNCuda.tnn_backend_name(kv.sess)

# Load via the legacy Mat-mediated path (build a Toy::SmolLM2 model,
# upload its weights). This is the path demos/smollm2_kv_cuda.rb uses.
puts "loading via Mat-mediated upload..."
model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)
SmolLM2KVCuda.upload_from(kv, model)
puts "  loaded"

puts "prefilling " + ids.length.to_s + " prompt tokens..."
i = 0
while i < ids.length
  SmolLM2KVCuda.decode_step(kv, ids[i], i)
  i = i + 1
end

puts "generating " + N_NEW.to_s + " tokens..."
n = 0
while n < N_NEW
  pos = ids.length
  last_id = ids[pos - 1]
  logits = SmolLM2KVCuda.decode_step(kv, last_id, pos)
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

# Probe: print first 8 logit values and a few across the vocab.
# If logits are all-zero (kernel didn't run / returned zeros) but
# argmax still picks something, we'd see "ties" picking a high
# index. If logits show varied numerical activity, kernel ran but
# computed wrong numbers.
puts ""
puts "logit diag:"
pos = ids.length
last_logits = SmolLM2KV.decode_step_cuda_unused = nil   # no-op placeholder
print "  logits[0..7]: "
i = 0
ll = SmolLM2KVCuda.decode_step(kv, ids[ids.length-1], ids.length)
while i < 8
  print " " + ll.flat[i].to_s
  i = i + 1
end
puts ""
print "  logits[100, 1000, 50000, 100000, 112919]:"
print " " + ll.flat[100].to_s
print " " + ll.flat[1000].to_s
print " " + ll.flat[50000].to_s
print " " + ll.flat[100000].to_s
print " " + ll.flat[112919].to_s
puts ""
