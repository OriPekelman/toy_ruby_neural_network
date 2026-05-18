# Full inference on the native GGUF via the COPY path (not mmap).
# Skips ctx_w_mmap entirely. If CUDA produces correct output here,
# the bug is specifically in ctx_w_mmap USAGE (not its presence —
# which we already fixed via lazy init).

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF  = "data/qwen25-1.5b-native.gguf"
MAX_T = 256
N_NEW = 8

ids = [9707, 11, 847, 829, 374]

cfg = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)

kv = SmolLM2KVFFICacheCuda.new
# realize_for (NOT realize_for_mmap) — uses ctx_w only, no ctx_w_mmap.
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)
puts "backend: " + TinyNNCuda.tnn_backend_name(kv.sess)

# Use the auto-dispatcher: for a native GGUF this routes to
# GGUFLoad.load_kv_cache_directly_native (memcpy, no transpose).
# This skips ctx_w_mmap entirely — weights go into ctx_w-allocated
# regular CUDA buffers via ggml_backend_tensor_set (cudaMemcpy).
GGUFLoad.load_kv_cache_auto(kv, GGUF)
puts "  loaded via auto-dispatch (native = memcpy into ctx_w)"

i = 0
while i < ids.length
  SmolLM2KVCuda.decode_step(kv, ids[i], i)
  i = i + 1
end

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
