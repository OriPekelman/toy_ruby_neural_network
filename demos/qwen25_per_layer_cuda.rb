# CUDA half of the per-layer divergence test. Outputs first 16
# logit values for shallow N-layer forward passes through the
# native-loader CUDA path.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF  = "data/qwen25-1.5b-native.gguf"
MAX_T = 16
TOKEN = 9707

cfg_full = SmolLM2ConfigLoader.read(GGUF)
flags    = GGUFLoad.detect_smollm2_flags(GGUF)

def shrink_cfg(orig, n_layers)
  Toy::SmolLM2Config.new(orig.vocab, orig.d_model, orig.n_heads,
                          orig.n_kv, orig.d_ff, n_layers,
                          orig.ctx, orig.rope_base, orig.rms_eps)
end

test_ns = [1, 2, 4, 8, 16, cfg_full.n_layers]
ti = 0
while ti < test_ns.length
  n_layers = test_ns[ti]
  cfg_n = shrink_cfg(cfg_full, n_layers)

  kv = SmolLM2KVFFICacheCuda.new
  kv.realize_for(MAX_T, cfg_n.d_model, cfg_n.d_ff, cfg_n.n_heads, cfg_n.n_kv,
                  cfg_n.n_layers, cfg_n.vocab, cfg_n.rope_base, cfg_n.rms_eps,
                  flags.untied, flags.qkv_bias)
  GGUFLoad.load_kv_cache_auto(kv, GGUF)
  logits = SmolLM2KVCuda.decode_step(kv, TOKEN, 0)

  print "N=" + n_layers.to_s + " gpu logits[0..7]:"
  i = 0
  while i < 8
    print " " + logits.flat[i].to_s
    i = i + 1
  end
  puts ""

  best = 0; best_v = logits.flat[0]
  i = 1
  v = cfg_n.vocab
  while i < v
    if logits.flat[i] > best_v
      best_v = logits.flat[i]; best = i
    end
    i = i + 1
  end
  puts "N=" + n_layers.to_s + " gpu argmax=" + best.to_s + " val=" + best_v.to_s
  ti = ti + 1
end
