# Per-layer divergence test for the mmap-CUDA bug.
#
# Loads Qwen2.5-1.5B-native, but runs the forward pass with a
# REDUCED number of layers (n_layers=1, 2, 3, ...). For each N,
# compares CPU vs CUDA logits. The smallest N where CUDA diverges
# from CPU pinpoints whether the bug is:
#   - N=1: the bug is single-layer-internal (some op in layer 0)
#   - N=k (k>1): cumulative kernel state across layers
#
# Uses the same realize_for + load_kv_cache_auto path that triggers
# the wrong-output bug in full inference (28 layers → top=112919).

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"        # CPU class
require_relative "../lib/toy_smollm2_ffi_kv_cuda"   # CUDA class

GGUF  = "data/qwen25-1.5b-native.gguf"
MAX_T = 16
TOKEN = 9707     # first token of "Hello, my name is"

cfg_full = SmolLM2ConfigLoader.read(GGUF)
flags    = GGUFLoad.detect_smollm2_flags(GGUF)

# Make a per-N narrow config.
def shrink_cfg(orig, n_layers)
  Toy::SmolLM2Config.new(orig.vocab, orig.d_model, orig.n_heads,
                          orig.n_kv, orig.d_ff, n_layers,
                          orig.ctx, orig.rope_base, orig.rms_eps)
end

def run_cpu(cfg, flags)
  kv = SmolLM2KVFFICache.new
  kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                  cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                  flags.untied, flags.qkv_bias)
  kv.load_weights(GGUF)
  SmolLM2KV.decode_step(kv, TOKEN, 0)
end

def run_cuda(cfg, flags)
  kv = SmolLM2KVFFICacheCuda.new
  kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                  cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                  flags.untied, flags.qkv_bias)
  GGUFLoad.load_kv_cache_auto(kv, GGUF)
  SmolLM2KVCuda.decode_step(kv, TOKEN, 0)
end

puts "Per-layer divergence test (Qwen2.5-1.5B-native, tok=" + TOKEN.to_s + ")"
puts "cfg: vocab=" + cfg_full.vocab.to_s +
     " d=" + cfg_full.d_model.to_s +
     " L_full=" + cfg_full.n_layers.to_s
puts ""

# Test N = 1, 2, 4, 8, 16, 28 (the full layer count).
test_ns = [1, 2, 4, 8, 16, cfg_full.n_layers]
ti = 0
while ti < test_ns.length
  n_layers = test_ns[ti]
  cfg_n = shrink_cfg(cfg_full, n_layers)

  cpu_logits = run_cpu(cfg_n, flags)
  gpu_logits = run_cuda(cfg_n, flags)

  # Compare a few sample positions.
  max_diff = 0.0
  first_diff = -1
  i = 0
  v = cfg_full.vocab
  while i < v
    d = cpu_logits.flat[i] - gpu_logits.flat[i]
    if d < 0; d = -d; end
    if d > max_diff
      max_diff = d
      if first_diff < 0 && d > 1.0e-3; first_diff = i; end
    end
    i = i + 1
  end

  # Find each path's argmax.
  best_c = 0; best_cv = cpu_logits.flat[0]
  i = 1
  while i < v
    if cpu_logits.flat[i] > best_cv
      best_cv = cpu_logits.flat[i]; best_c = i
    end
    i = i + 1
  end
  best_g = 0; best_gv = gpu_logits.flat[0]
  i = 1
  while i < v
    if gpu_logits.flat[i] > best_gv
      best_gv = gpu_logits.flat[i]; best_g = i
    end
    i = i + 1
  end

  m = max_diff < 1.0e-3 ? "match" : "DIFF"
  puts "N=" + n_layers.to_s.rjust(3) + " " + m +
       "  max_abs_diff=" + max_diff.to_s.ljust(22) +
       "  cpu_argmax=" + best_c.to_s.rjust(6) +
       "  cuda_argmax=" + best_g.to_s.rjust(6)
  ti = ti + 1
end
