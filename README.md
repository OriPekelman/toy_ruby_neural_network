# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

A small transformer language model in Ruby. It compiles to a native
binary (via [Spinel](https://github.com/matz/spinel), matz's Ruby AOT
compiler) and runs real, downloadable HuggingFace models — GPT-2 and
SmolLM2-135M — at output-identical fidelity to PyTorch.

The project's goal is to be **readable**: the whole forward pass of a
transformer fits on one screen, every shape is annotated inline, and
the building blocks are named after the math.

```ruby
# lib/toy_smollm2.rb — the SmolLM2 block, in full.
def forward(x, pos_start)
  x.add!(@l_attn.forward(@rn1.forward(x), pos_start))   # residual after attention
  x.add!(@l_ffn.forward(@rn2.forward(x)))               # residual after FFN
  x
end
```

If you can read that, you can read the whole model.

## Quickstart

You'll need a Ruby installation, a checked-out copy of
[Spinel](https://github.com/matz/spinel) at `~/sites/spinel`, and a C
compiler. `uv` (for the conversion scripts) installs itself if you
have it; if not, `pip install uv` first.

```sh
# One-time: clone + build ggml (the underlying math library) and convert
# SmolLM2-135M from HuggingFace into our native model file.
make setup-ggml                              # ~30 s
./prep/convert_smollm2_to_gguf.py            # ~30 s, writes data/smollm2-135m-f32.gguf
./prep/smollm2_tokens.py encode "Once upon a time"

# Build and run.
make smollm2_pleasant
./demos/smollm2_pleasant

./prep/smollm2_tokens.py decode
# → "Once upon a time, there was a little girl named Lily"
```

The model is real (135M parameters, downloaded from HuggingFace) and
the output is byte-identical to what PyTorch produces on the same
prompt.

## What you'll see when you run it

`./demos/smollm2_pleasant` prints, before generating:

```
Toy::SmolLM2 (134.52M params)
  config: vocab=49152 d_model=576 n_heads=9 n_kv=3 d_ff=1536 n_layers=30 ctx=8192 rope_base=100000
  l_token_embed: Embedding(vocab=49152, d=576)  [28.31M]
  l_rope:        RoPE(d_head=64, max_seq=8192)
  l_stack: 30 × SmolLM2Block
    rn1:    RMSNorm(d=576, eps=1e-5)
    l_attn: GQAttention(d=576, n_q=9, n_kv=3, d_head=64, group=3)
    rn2:    RMSNorm(d=576, eps=1e-5)
    l_ffn:  SwiGLU(d=576, d_ff=1536)
    (per-block params: 3.54M)
  l_final_norm: RMSNorm(d=576, eps=1e-5)
  unembed: tied to l_token_embed (logits = x · l_token_embed.T)
```

That's the entire architecture. Each line you can click through to
the corresponding ~40-line Ruby class in `lib/toy.rb`.

## Introspection

Every `Mat` (the 2D float tensor type) knows its shape and can
describe itself:

```ruby
x.shape         # → "[5, 768]"
x.info          # → "Mat[5, 768] min=-2.34 max=1.97 mean=0.012"
```

Every layer reports a one-line summary and its parameter count:

```ruby
puts ffn.summary       # → "FFN(d=768, hidden=3072, act=gelu_new)"
puts ffn.param_count   # → 4,722,432
```

Every model has `.describe` (the block shown above is the output).

To peek mid-forward at a tensor without breaking flow:

```ruby
x = Toy.tap("after attn", @attn.forward(@ln1.forward(x)))
# prints "after attn: Mat[5, 768]" and passes x through.
```

## A guided tour of the code

If you're new to transformers, here's what each piece in the
`describe` output means and where it lives in code:

| Name | What it does | Where |
|---|---|---|
| `Embedding` | Turns integer token IDs into vectors (rows of a big lookup table). | [`lib/toy.rb`](lib/toy.rb) |
| `RMSNorm` / `LayerNorm` | Normalizes a row's values so training stays stable. | [`lib/toy.rb`](lib/toy.rb) |
| `RoPE` | Encodes "where is this token in the sequence" by rotating Q/K vectors. | [`lib/toy.rb`](lib/toy.rb) |
| `CausalSelfAttention` / `GQAttention` | The attention mechanism: each token decides which earlier tokens to look at. | [`lib/toy.rb`](lib/toy.rb) |
| `FFN` / `SwiGLU` | A two-layer feedforward network with a nonlinearity in the middle. | [`lib/toy.rb`](lib/toy.rb) |
| `GPT2Block` / `SmolLM2Block` | One full layer = pre-norm → attention → residual → pre-norm → FFN → residual. | [`lib/toy_gpt2.rb`](lib/toy_gpt2.rb) / [`lib/toy_smollm2.rb`](lib/toy_smollm2.rb) |
| `GPT2` / `SmolLM2` | The full model: embed tokens → stack of N blocks → final norm → unembed back to vocab probabilities. | same |

That's it. There's no extra magic. Every shape in every line is
annotated as a comment.

## Layout

```
.
├── lib/
│   ├── toy.rb                 Reusable building blocks: LayerNorm, RMSNorm,
│   │                          Linear, Embedding, CausalSelfAttention,
│   │                          GQAttention, FFN, SwiGLU, RoPE
│   ├── toy_gpt2.rb            Toy::GPT2 — full model in ~80 lines
│   ├── toy_smollm2.rb         Toy::SmolLM2 — SmolLM2 / llama-family in ~110 lines
│   ├── toy_gpt2_loader.rb     GGUF → Toy::GPT2 weights
│   ├── toy_smollm2_loader.rb  GGUF → Toy::SmolLM2 weights
│   ├── toy_trainer.rb         Toy::Trainer — pleasant training loop wrapper
│   ├── transformer.rb         Mat (2D float tensor) + the from-scratch
│   │                          TransformerLM with forward + backward + Adam
│   ├── training.rb            LRSchedule, DataLoader, Adam, corpus readers
│   ├── gguf_load.rb           Generic GGUF read helpers (used by both loaders)
│   ├── bpe.rb                 Pure-Ruby byte-level BPE (GPT-2's tokenizer)
│   ├── tinynn.rb              FFI bridge from Ruby to ggml (CPU)
│   └── tinynn_cuda.rb         FFI bridge to ggml-cuda (GPU)
├── demos/                     End-to-end Ruby drivers (one per build target)
│   ├── smollm2_pleasant.rb    The flagship demo — SmolLM2 via Toy::*
│   ├── gpt2_pleasant.rb       Same idea, GPT-2 / DistilGPT-2
│   ├── train_pleasant.rb      Training via Toy::Trainer
│   ├── distilgpt2_demo*.rb    Older legacy demos (native / FFI / KV / CUDA paths)
│   ├── train_minimal.rb       Tiny SGD smoke (40 steps)
│   └── train_tinystories.rb   Full TinyStories training run
├── prep/                      Host-side Python helpers (model conversion,
│                              tokenization, parity reference)
├── tep_demo/                  HTTP server (OpenAI-compatible API) via Tep
├── tinynn/                    C shim wrapping ggml + per-op parity smokes
└── docs/                      Long-form notes (scout, upstream issue drafts)
```

## Architecture

Plain pre-norm decoder-only transformer. Diagram for one block:

```
token_ids ─▶ embed ─▶ [Block]×N ─▶ Norm ─▶ unembed (tied) ─▶ logits

Block (pre-norm):
    x ─▶ Norm ─▶ multi-head causal attention ─▶ + ─┐
                                                    │
                                  residual ◀────────┘
    x'─▶ Norm ─▶ FFN (Linear → activation → Linear) ─▶ + ─┐
                                                           │
                                  residual ◀──────────────┘
```

The model classes:

- **`Toy::GPT2`** — HF GPT-2 shape. LayerNorm with bias, biases on
  every Linear, GeLU FFN, tied embeddings, learned absolute position
  embeddings.
- **`Toy::SmolLM2`** — Llama family shape. RMSNorm, no biases, SwiGLU
  FFN, tied embeddings, RoPE positions, grouped-query attention.
- **`TransformerLM`** ([`lib/transformer.rb`](lib/transformer.rb)) — older
  from-scratch training-shape model with Adam + cross-entropy backward
  in pure Ruby (loosely a Llama-1 stripped down). Used by
  `demos/train_minimal` and `demos/train_tinystories`.

## How fast is it?

`demos/smollm2_pleasant` is the **native Mat path** — every matmul is
a plain Ruby double-loop. That's slow but exists so you can read it.

For real inference speed there's a parallel **FFI path** (Ruby drives
[ggml](https://github.com/ggml-org/ggml) over FFI), CPU + CUDA
backends. Current numbers (single forward pass, T_SEQ=5):

| Path | gpt2-small per-step |
|---|---:|
| `Toy::GPT2` (native Mat, f64) | ~1.7 s |
| FFI full-forward, CPU | 56 ms |
| FFI KV-cache, CPU | 14 ms |
| FFI KV-cache, CUDA (gx10) | 22 ms |

All produce identical token sequences; the KV path is parity-verified
against PyTorch `transformers` at F32 precision. Numbers in
[`HF_GPT2.md`](HF_GPT2.md). The `Toy::SmolLM2` path is currently
native-only — the FFI mirror is on the todo list.

## Quantization

The GPT-2 converter writes Q8_0 / Q4_0 / Q5_0 GGUFs too:

```sh
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 --out data/gpt2-q8_0.gguf --quantize q8_0
# 498 MB F32 → 248 MB Q8_0; byte-identical generation
```

K-quants (Q4_K_M etc.) need llama.cpp's quantizer.

## Training

```sh
make setup-ggml
make train_pleasant
./prep/prep_tinystories.rb   # tokenize the corpus (one-time)
./demos/train_pleasant       # bump EPOCHS in the file for a real run
```

The from-scratch model converges to TinyStories-shaped text:

> *once upon a time there was a little boy named tim he loved to play
> in the park…*

`Toy::Trainer` ([`lib/toy_trainer.rb`](lib/toy_trainer.rb)) wraps the
optimizer / gradients / schedule so the training loop reads as:

```ruby
trainer = Toy::Trainer.new(model)
trainer.lr_max = 1e-3

step = 0
while step < total_steps
  loss = trainer.step!(data[step % data.length])
  puts "#{step}: #{loss}" if step % 100 == 0
  step += 1
end
```

## Spinel notes (the "why is the code shaped that way" section)

[Spinel](https://github.com/matz/spinel) does whole-program type
inference. The whole reachable Ruby has to type-check against a
single closed world, which constrains how you can write code. The
patterns that come up:

- **Flat 1D `Float` storage** in `Mat` (Spinel can't infer `Array<Array<Float>>`).
- **Classes, not hashes**, for records with mixed-type values.
- **`nrows` / `ncols`** instead of `rows` / `cols` (the latter collides).
- **Result wrappers** instead of multi-value returns.
- **No `Array<Array<Int>>.pop`** — silently no-ops (issue #520).
- **Hash-Int 0 ↔ nil** — `0 != nil` is `false` (issue #521).
- **Don't reuse local-var / param names across types** — same name
  with two concrete types unifies to `sp_RbVal` and breaks both
  sites (issue #538). Drives the `l_*` / `g_*` prefix convention on
  Toy::SmolLM2 / Toy::GPT2 field names.
- **Don't reuse field names across classes with different return
  types** — same idea, accessor flavor (issue #537).

### Spinel bugs filed during this project

What's been merged upstream:

- [matz/spinel#258](https://github.com/matz/spinel/pull/258),
  [#473](https://github.com/matz/spinel/issues/473),
  [#474](https://github.com/matz/spinel/issues/474) — codegen + FFI fixes.

What I filed during the HF GPT-2 + SmolLM2 work:

- [matz/spinel#520](https://github.com/matz/spinel/issues/520) —
  `Array#pop` on `Array<Array<Int>>` is silently a no-op
- [matz/spinel#521](https://github.com/matz/spinel/issues/521) —
  stored `0` in `Hash<String, Int>` is indistinguishable from missing
- [matz/spinel#532](https://github.com/matz/spinel/issues/532) —
  `String#index` returns `-1` instead of `nil` when not found
- [matz/spinel#537](https://github.com/matz/spinel/issues/537) —
  field-name collapse across classes
- [matz/spinel#538](https://github.com/matz/spinel/issues/538) —
  local variable / param name collapse across methods

Drafts are in [`docs/spinel-issues/`](docs/spinel-issues).

Still open from the training-era work:

- [ggml-org/ggml#1491](https://github.com/ggml-org/ggml/issues/1491) —
  `ggml_rms_norm_back` mismatch via the new scheduler. Not blocking
  inference.

## Status

Three working models:

- **Training (toy):** ~30K-param TinyStories model, loss ~5.3 → ~3
  over 30 epochs. `Toy::Trainer` or the older
  `demos/train_tinystories`.
- **Inference, GPT-2 family:** distilgpt2 + gpt2-small (124 M) load
  from GGUF, generate via `Toy::GPT2` (native Mat) or via the FFI
  KV-cache path. Parity-matches HuggingFace `transformers`
  byte-for-byte on argmax sequences.
- **Inference, llama family:** SmolLM2-135M loads from GGUF, generates
  via `Toy::SmolLM2`. Native Mat only at this point — FFI mirror is
  the next perf goal.

A toy you can read top-to-bottom that happens to run real models.
