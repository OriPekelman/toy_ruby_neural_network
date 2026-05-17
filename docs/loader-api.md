# Loader API surface

There are two ways to get GGUF weights into memory in this codebase.
They are **peers**, not "old vs new". Pick the one that matches your
use case.

## 1. Mat-mediated path — `GGUFLoad.load_toy_smollm2`

For inspection, fine-tuning, parity checks, anything that wants the
full Ruby `Toy::SmolLM2` object graph.

```ruby
cfg   = SmolLM2ConfigLoader.read(gguf_path)
model = Toy::SmolLM2.new(cfg)
GGUFLoad.load_toy_smollm2(model, gguf_path)

# Inspect / modify a weight as a Ruby Mat:
qw_head0 = model.stack[3].attn.w_q[0]   # Mat[d_model, d_head]
qw_head0.flat[0] += 1.0
```

Memory cost: ~12 bytes per parameter (Float64 Mat + ggml shadow).
Caps practical size around ~3B params on a 121 GiB box.

## 2. Direct path — `kv.load_weights`

For inference at 7B-class scale, where the Float64 intermediate is
unaffordable. Skips Mat construction entirely.

```ruby
cfg   = SmolLM2ConfigLoader.read(gguf_path)
flags = GGUFLoad.detect_smollm2_flags(gguf_path)

kv = SmolLM2KVFFICache.new
kv.realize_for(max_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                flags.untied, flags.qkv_bias)
kv.load_weights(gguf_path)
```

Memory cost: 4 bytes per parameter (ggml f32 only). Verified to
~30 GiB peak for Qwen2.5-7B end-to-end.

Equivalent module form (still available, callable from anywhere):

```ruby
GGUFLoad.load_kv_cache_directly(kv, gguf_path)
```

### Mat-roundtrip — pulling weights back out

The direct path is **not** a one-way trip. You can pull any persistent
FFI tensor back into a Ruby `Mat` for inspection, export, or as the
seed for a Mat-mediated fine-tune:

```ruby
emb_mat = kv.read_persistent_mat(kv.t_token_embed, cfg.vocab, cfg.d_model)
norm    = kv.read_persistent_mat(kv.t_final_norm_gamma, 1, cfg.d_model)
qhead0  = kv.read_persistent_mat(kv.kv_blocks_ffi[3].t_w_q[0],
                                  cfg.d_model, cfg.d_model / cfg.n_heads)
```

Backed by `tnn_download_to_f64_array`, which chunks through the
scratch buffer — works on tensors of arbitrary size, not just those
that fit in scratch.

Verified bit-identical to the GGUF source on SmolLM2-135M's 28M-float
token_embed: max diff 0.0 across all elements.

## Why both, why not just one

- Mat-mediated is the **only** path with a training-side graph
  today (TransformerLM still uses it). Removing it would lock the
  codebase into inference-only territory, which is explicitly not
  the direction.
- Direct is the **only** path that scales past ~3B parameters at
  current memory budgets.
- Mat-roundtrip on the direct path closes the loop: if you started
  via the direct path for memory reasons, you can still pull
  individual tensors back into Mat space for surgery.

The medium-term goal is a training-capable FFI cache (KV-cache is
inference-shaped — separate forward/backward graphs needed for
training). Until then, the two paths coexist.

## Spinel hazards

Adding helpers near `Mat.new(rows, cols)`-shaped code is easy to
trip on Spinel's whole-program local-var unification. The
verification demo (`demos/roundtrip_probe.rb`) was rewritten to use
disambiguated local names (`hh`, `eidx`, `kk`, `kdiff`, `oracle_mat`)
after `handle` / `i` / `d` / `max_diff` / `oracle` triggered "aggregate
value used where an integer was expected" at C-compile time. If you
extend the demos, expect to play this game.
