# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

A small transformer language model in Ruby. AOT-compiled to a native
binary by [Spinel](https://github.com/matz/spinel) (matz's Ruby AOT
compiler). Runs real HuggingFace models — GPT-2 and SmolLM2-135M —
at output-identical fidelity to PyTorch.

The goal is to be **readable**: the whole forward pass fits on one
screen, every shape is annotated inline, the building blocks are
named after the math.

```ruby
# lib/toy_smollm2.rb — the SmolLM2 block, in full.
def forward(x, pos_start)
  x.add!(@attn.forward(@rn1.forward(x), pos_start))   # residual after attention
  x.add!(@ffn.forward(@rn2.forward(x)))               # residual after FFN
  x
end
```

If you can read that, you can read the whole model.

## Quickstart

```sh
make setup-ggml                                # ~30 s
./prep/convert_smollm2_to_gguf.py              # ~30 s; writes data/smollm2-135m-f32.gguf
./prep/smollm2_tokens.py encode "Once upon a time"

make smollm2_kv && ./demos/smollm2_kv          # ~77 tok/s on M2 Air
./prep/smollm2_tokens.py decode
# → "Once upon a time, there was a little girl named Lily..."
```

Requires Ruby, Spinel checked out at `~/sites/spinel`, and a C
compiler. `uv` installs itself for the Python converter; or
`pip install uv` first.

## What's in the box

| Path | What |
|---|---|
| [`lib/toy.rb`](lib/toy.rb)              | Building blocks: `Mat`, `LayerNorm`, `RMSNorm`, `Linear`, `Embedding`, `CausalSelfAttention`, `GQAttention`, `FFN`, `SwiGLU`, `RoPE` |
| [`lib/toy_gpt2.rb`](lib/toy_gpt2.rb)    | `Toy::GPT2` — full HF GPT-2 in ~80 lines |
| [`lib/toy_smollm2.rb`](lib/toy_smollm2.rb) | `Toy::SmolLM2` — Llama family (SmolLM2 / TinyLlama shape) |
| [`lib/toy_smollm2_ffi_kv*.rb`](lib/)    | KV-cache FFI mirror (CPU + CUDA) — the perf path |
| [`lib/toy_trainer.rb`](lib/toy_trainer.rb) | `Toy::Trainer` — training-loop wrapper |
| [`sig/toy.rbs`](sig/toy.rbs)            | RBS type signatures (validates via `rbs validate`) |
| [`demos/`](demos/)                      | End-to-end Ruby drivers — see [`demos/README.md`](demos/README.md) |
| [`docs/`](docs/)                        | Long-form notes: bench numbers, Spinel issue drafts, design scout |

## Highlights

- **Introspection**: every `Mat` knows its shape (`x.shape` → `"[5, 768]"`),
  every layer has a `summary` + `param_count`, every model has
  `describe` and `algorithm_card`. The latter emits Phuong–Hutter style
  pseudocode (arXiv:2207.09238) with shape annotations on every line
  — what the code actually does, written like the paper.
- **Round-trip**: [`prep/card_to_code.rb`](prep/card_to_code.rb) parses
  an algorithm card back to the Ruby that constructs the model.
- **Throughput**: SmolLM2-135M reaches **77 tok/s on M2 Air CPU** and
  **89 tok/s on NVIDIA GB10 CUDA** through the FFI KV-cache path.
  Full numbers in [`docs/bench-gx10-2026-05-16.md`](docs/bench-gx10-2026-05-16.md).
- **Real models, real outputs**: SmolLM2-135M on "Once upon a time"
  produces "Once upon a time, there was a little girl named Lily…" —
  the actual canonical SmolLM2 continuation.

## Reading the rest

- [`docs/HF_GPT2.md`](HF_GPT2.md) — the long story of getting GPT-2
  to run identically to PyTorch
- [`docs/bench-gx10-2026-05-16.md`](docs/bench-gx10-2026-05-16.md) — perf
  numbers across the paths
- [`docs/scout-small-models.md`](docs/scout-small-models.md) — design
  notes for what to support next
- [`docs/tinyllama-known-issue.md`](docs/tinyllama-known-issue.md) —
  the f32-precision issue when running TinyLlama via FFI
- [`docs/spinel-issues/`](docs/spinel-issues/) — Spinel bugs filed
  during this project (all closed at time of writing)
- [`tep_demo/README.md`](tep_demo/README.md) — OpenAI-compatible HTTP
  API via Tep+Spinel

A toy you can read top-to-bottom that happens to run real models.
