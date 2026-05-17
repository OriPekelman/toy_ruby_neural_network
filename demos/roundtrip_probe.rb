require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/training"

GGUF  = "data/smollm2-135m-f32.gguf"
MAX_T = 64

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)

kv = SmolLM2KVFFICache.new
kv.realize_for(MAX_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)

t_load = Time.now
kv.load_weights(GGUF)
puts "kv.load_weights " + ((Time.now - t_load) * 1000.0).to_s + " ms"

t1 = Time.now
m_norm = kv.read_persistent_mat(kv.t_final_norm_gamma, 1, cfg.d_model)
puts "small " + ((Time.now - t1) * 1000.0).to_s + " ms gamma[0]=" + m_norm.flat[0].to_s

t2 = Time.now
m_emb = kv.read_persistent_mat(kv.t_token_embed, cfg.vocab, cfg.d_model)
puts "large " + ((Time.now - t2) * 1000.0).to_s + " ms emb[0]=" + m_emb.flat[0].to_s

hh = TinyNN.tnn_gguf_load(GGUF)
eidx = TinyNN.tnn_gguf_find_index(hh, "token_embd.weight")
oracle_mat = Mat.new(cfg.vocab, cfg.d_model)
nem = cfg.vocab * cfg.d_model
TinyNN.tnn_gguf_read_f32_to_doubles(hh, eidx, oracle_mat.flat, nem)
TinyNN.tnn_gguf_free(hh)
puts "oracle[0] = " + oracle_mat.flat[0].to_s

# bit-identical scan (avoid common local var names that collide)
kdiff = 0.0
kk = 0
while kk < nem
  delta = m_emb.flat[kk] - oracle_mat.flat[kk]
  if delta < 0.0; delta = 0.0 - delta; end
  if delta > kdiff; kdiff = delta; end
  kk = kk + 1
end
puts "max |download - oracle| = " + kdiff.to_s
