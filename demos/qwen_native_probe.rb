# demos/qwen_native_probe.rb — native (Float64 Mat) forward through
# Toy::SmolLM2 for Qwen2.5-0.5B. Bypasses FFI entirely. Slow but
# precision-safe; isolates whether the NaN is FFI f32 precision or
# something architectural (weight load / config / RoPE).

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"

GGUF = "data/qwen25-0.5b-f32.gguf"
cfg  = SmolLM2ConfigLoader.read(GGUF)
forced_layers = ENV["FORCE_N_LAYERS"]
if forced_layers != nil
  cfg.n_layers = forced_layers.to_i
end
puts "eps=" + cfg.rms_eps.to_s + " L=" + cfg.n_layers.to_s +
     " rope=" + cfg.rope_base.to_s

model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, GGUF)
puts ""

t0 = Time.now
logits = model.forward([100], 0)
elapsed = (Time.now - t0) * 1000.0
puts "native forward: " + elapsed.to_s + " ms"

nans    = 0
finites = 0
max_abs = 0.0
i = 0
n = logits.nrows * logits.ncols
while i < n
  v = logits.flat[i]
  if v == v && v.abs < 1.0e30
    finites = finites + 1
    if v.abs > max_abs
      max_abs = v.abs
    end
  else
    nans = nans + 1
  end
  i = i + 1
end
puts "logits: finite=" + finites.to_s + " nan_or_inf=" + nans.to_s +
     " max_abs=" + max_abs.to_s +
     " logit[0]=" + logits.flat[0].to_s
