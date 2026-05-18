# demos/qwen25_direct_native_mmap_cuda.rb — CUDA mirror of
# qwen25_direct_native_mmap.rb. Loads a native-layout GGUF and binds
# weight tensors directly at the mmap'd file pages via the vendored
# ggml_backend_cuda_buffer_from_ptr (commit 5f3bee4 in
# vendor/ggml; project commit e302d32). On GB10 unified memory the
# kernels read host-mmap pages via UVA — zero copy.
#
# F32-only on CUDA today (the V matmul flip required for Q8 hasn't
# been mirrored to the CUDA build_decode_step). Target file: a 1.5B
# f32 native GGUF.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF  = "data/qwen25-1.5b-native.gguf"
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
puts "flags: untied=" + flags.untied.to_s +
     " qkv_bias=" + flags.qkv_bias.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)

kv = SmolLM2KVFFICacheCuda.new
t0 = Time.now
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
puts "  CUDA realize + mmap in " + ((Time.now - t0) * 1000.0).to_s + " ms"
puts "  backend: " + TinyNNCuda.tnn_backend_name(kv.sess)

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
