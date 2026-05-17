# demos/qwen_probe.rb — diagnostic for Qwen2.5 NaN-in-FFI-forward.
#
# Forces rms_eps and n_layers via env vars. Runs a single decode step
# at position 0 with a fixed token id and reports finite/NaN logit
# counts. No prompt file, no generation loop — minimum repro.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/training"

GGUF = "data/qwen25-0.5b-f32.gguf"
cfg = SmolLM2ConfigLoader.read(GGUF)
forced_eps = ENV["FORCE_EPS"]
if forced_eps != nil
  cfg.rms_eps = forced_eps.to_f
end
forced_layers = ENV["FORCE_N_LAYERS"]
if forced_layers != nil
  cfg.n_layers = forced_layers.to_i
end
forced_rope = ENV["FORCE_ROPE"]
if forced_rope != nil
  cfg.rope_base = forced_rope.to_f
end
puts "eps=" + cfg.rms_eps.to_s + " L=" + cfg.n_layers.to_s +
     " rope=" + cfg.rope_base.to_s

model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)

kv = SmolLM2KVFFICache.new
kv.realize_for(64, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                model.has_untied_output, model.stack[0].attn.has_qkv_bias)
SmolLM2KV.upload_from(kv, model)
if ENV["TRACE"] != nil
  puts "  [trace enabled]"
  kv.enable_trace!
end

logits = SmolLM2KV.decode_step(kv, 100, 0)
nans = 0
finites = 0
i = 0
while i < cfg.vocab
  v = logits.flat[i]
  if v == v && v.abs < 1.0e30
    finites = finites + 1
  else
    nans = nans + 1
  end
  i = i + 1
end
puts "finite=" + finites.to_s + " nan_or_inf=" + nans.to_s +
     " top0=" + logits.flat[0].to_s + " top1=" + logits.flat[1].to_s
