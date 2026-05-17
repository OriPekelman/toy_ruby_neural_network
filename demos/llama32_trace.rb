# demos/llama32_trace.rb — trace-tap probe of Llama-3.2-1B inference.
# Goal: find where the per-layer values diverge or explode under the
# direct-loader path. Single decode_step with the trace tap on.

require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/training"

GGUF  = "data/llama-3.2-1b-f32.gguf"
MAX_T = 32

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)

kv = SmolLM2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)
kv.load_weights(GGUF)

puts "untied output: " + flags.untied.to_s + "  (Llama 3.2 has lm_head untied)"
puts "qkv bias: " + flags.qkv_bias.to_s + "  (Llama lacks qkv bias)"

# Enable trace tap before the first decode call.
kv.enable_trace!

# Synthetic prompt: BOS-ish.
ids = [128000, 9906, 11, 856]
puts "decode step at pos 0 (BOS) with trace on:"
SmolLM2KV.decode_step(kv, ids[0], 0)
kv.dump_trace
