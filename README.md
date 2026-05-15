# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

Decoder-only transformer LM in Ruby, AOT-compiled to a native binary
by [Spinel](https://github.com/matz/spinel) (matz's Ruby AOT
compiler). Two stories live in one repo:

1. **From-scratch training.** ~700 lines of readable Ruby — `Mat`,
   `Block`, `TransformerLM`, forward, backward, Adam — trained on a
   slice of TinyStories. The project's original purpose.
2. **Real HF model inference.** DistilGPT2 and GPT-2 (124M…) load
   from `.gguf`, run on CPU and CUDA via an FFI bridge to
   [ggml](https://github.com/ggml-org/ggml), tokenize and detokenize
   in pure Ruby BPE, and produce byte-identical output to PyTorch
   `transformers`. See [HF_GPT2.md](HF_GPT2.md).

```sh
# 30-second tour: real-model inference, self-contained binary.
make setup-ggml                                          # build ggml CPU (~30 s)
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 \    # HF → GGUF (~30 s)
                                    --out data/gpt2-f32.gguf
./prep/dump_bpe.py                                       # vocab/merges → TSV
echo "Once upon a time, in a quiet village by the sea" > data/prompt.txt
make distilgpt2_demo_text && ./distilgpt2_demo_text
# → "Once upon a time, in a quiet village by the sea, there lived a
#    curious child named Nana. She was a young girl of about five years
#    old, and she was very curious..."
```

No Python at runtime. GGUF model + three BPE TSVs + a 1.1 MB native
binary. ~14 ms/token KV-decode at distilgpt2 shape on an M2 Air.

## Layout

```
.
├── lib/
│   ├── transformer.rb         Mat, Block, TransformerLM, Gradients, AdamState
│   ├── training.rb            LRSchedule, DataLoader, Adam, corpus readers
│   ├── gpt2.rb                GPT2LM (inference-only HF-shape transformer)
│   ├── gpt2_ffi.rb            FFI full-forward graph (CPU)
│   ├── gpt2_ffi_kv.rb         FFI KV-cache decode (CPU)
│   ├── gpt2_ffi_cuda.rb       FFI full-forward (CUDA)
│   ├── gpt2_ffi_kv_cuda.rb    FFI KV-cache decode (CUDA)
│   ├── gguf_load.rb           GGUF → GPT2LM weight loader + GPT2Config
│   ├── bpe.rb                 Byte-level BPE encoder + decoder
│   ├── tinynn.rb              FFI bridge to ggml (CPU): module TinyNN
│   └── tinynn_cuda.rb         FFI bridge to ggml-cuda: module TinyNNCuda
├── prep/
│   ├── prep_tinystories.rb    Tokenized corpus for the training path
│   ├── convert_distilgpt2_to_gguf.py   HF safetensors → GGUF (incl. quants)
│   ├── dump_bpe.py            Vocab/merges/byte-chars → TSV
│   ├── tokens.py              Python BPE shim (host-side); now superseded by lib/bpe.rb
│   └── parity.py              HF transformers reference logits + diff
├── tinynn/                    C shim (ggml wrappers) + parity / bench smokes
├── train_tinystories.rb       From-scratch training entrypoint
├── train_minimal.rb           Tiny SGD smoke
├── distilgpt2_demo_text.rb    Real-model inference, text in / text out
├── HF_GPT2.md                 Long-form notes on the HF inference path
└── tinynn/README.md           FFI bridge design and per-op coverage
```

## Architecture

Plain pre-norm decoder-only transformer. Two variants live side by
side in the codebase:

```
token_ids ─▶ embed ─▶ [Block]×N ─▶ Norm ─▶ unembed (tied) ─▶ logits

Block (pre-norm):
    x ─▶ Norm ─▶ multi-head causal attention ─▶ + ─┐
                                                    │
                                  residual ◀────────┘
    x'─▶ Norm ─▶ FFN (Linear → GeLU → Linear) ─▶ + ─┐
                                                     │
                                  residual ◀─────────┘
```

- `TransformerLM` (`lib/transformer.rb`): from-scratch training-shape
  model. RMSNorm, no biases, tied embeddings, learned absolute pos
  embeddings. Forward + backward + Adam all in pure Ruby.
- `GPT2LM` (`lib/gpt2.rb`): HF-shape inference-only model. LayerNorm
  with bias, biases on every Linear, GeLU(`gelu_new`), tied
  embeddings, learned absolute pos. Loads from `.gguf`.

Bits the two share: pre-norm structure, multi-head causal attention,
sequential residuals, GeLU, tied output embedding.

## Three forward paths

`make … && ./…` table for the GPT-2 side. The from-scratch training
path has its own demos (`train_minimal`, `train_tinystories`).

| Demo | What it does | Per-step (gpt2-small, T_SEQ=5) |
|---|---|---:|
| `distilgpt2_demo` | Native Mat (f64) forward in pure Ruby | 1.7 s |
| `distilgpt2_demo_ffi` | FFI full-forward, T_SEQ-padded | 56 ms |
| `distilgpt2_demo_kv` | FFI persistent KV cache, per-step decode | 14 ms |
| `distilgpt2_demo_text` | KV cache + Ruby BPE; takes a text prompt | 14 ms |
| `distilgpt2_demo_kv_cuda` | KV decode via ggml-cuda on gx10 GB10 | 22 ms |

All five produce **identical token sequences** on the same prompt; the
KV path is parity-verified against HF `transformers`' PyTorch
reference at F32 ULP precision (`max_abs_diff ≈ 3e-3` on the final
logits, argmax + top-5 match exactly).

Full numbers in [HF_GPT2.md](HF_GPT2.md). Older training-side
performance story in [tinynn/README.md](tinynn/README.md).

## Quantization

The same converter writes Q8_0 / Q4_0 / Q5_0 GGUFs (legacy
quantizers via `gguf-py`). The existing `tnn_gguf_read_f32_to_doubles`
dequantizes via ggml's type-traits — no project changes needed.

```sh
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 --out data/gpt2-q8_0.gguf --quantize q8_0
# 498 MB F32 → 248 MB Q8_0 (2x), byte-identical generation on gpt2-small
```

K-quants (Q4_K_M etc.) need llama.cpp's quantizer or a C-side helper
wrapping `ggml_quantize_chunk`; the load path would handle them but
gguf-py can't produce them.

## Spinel: what's been merged upstream / what's open

What's in:

- [matz/spinel#258](https://github.com/matz/spinel/pull/258) —
  `fix(codegen): root PtrArray temp in array-of-objects literal`
- [matz/spinel#473](https://github.com/matz/spinel/issues/473) —
  `fix(codegen): clear ptr ivars after SP_POOL_NEW before init body runs`
- [matz/spinel#474](https://github.com/matz/spinel/issues/474) —
  `feat(ffi): :float_array / :int_array specs for zero-copy bulk transfer`

What I filed during the HF GPT-2 work:

- [matz/spinel#520](https://github.com/matz/spinel/issues/520) —
  `Array#pop` on `Array<Array<Int>>` is silently a no-op
- [matz/spinel#521](https://github.com/matz/spinel/issues/521) —
  stored `0` in `Hash<String, Int>` is indistinguishable from a
  missing key; `Int 0 != nil` evaluates to `false`

Drafts for both are in [`docs/spinel-issues/`](docs/spinel-issues).

Still open from the training-era work:

- [ggml-org/ggml#1491](https://github.com/ggml-org/ggml/issues/1491) —
  `ggml_rms_norm_back` mismatch via `ggml_backend_sched_graph_compute`
  vs legacy `ggml_graph_compute_with_ctx`. Not blocking inference.

## Spinel constraints (a partial bestiary)

Spinel does whole-program type inference; the entire reachable Ruby
has to type-check against a single closed world. Patterns that come
up:

- **Flat 1D `Float` storage** in `Mat` (`Array<Array<Float>>` from
  `Array.new(n) { Array.new(m, 0.0) }` doesn't type-infer).
- **Classes, not hashes**, for heterogeneous-valued records
  (`Block`, `LayerCache`, KV cache).
- **`nrows` / `ncols`** field names (`rows` / `cols` collide).
- **Result wrappers** (`NormResult`, `FFResult`, `LossResult`,
  `GPT2KVStepResult`, …) instead of multi-value returns.
- **No `Array<Array<Int>>.pop`** — silently no-ops (issue #520).
- **Hash-Int 0 ↔ nil** — store `value + 1` so 0 means missing
  (issue #521).
- **Don't reuse local-var / param names across types** — same name
  with two concrete types unifies to `sp_RbVal`, breaks FFI boundary
  casts. Symptoms range from "compile error" to "silently produces
  the wrong answer 6 layers deep". Documented in
  [`HF_GPT2.md`](HF_GPT2.md)'s "name-collapse" table.

## Status

Two working ends:

- **Training (toy):** ~30K-param TinyStories model, loss ~5.3 → ~3
  over 30 epochs. Generations look TinyStories-shaped
  *("once upon a time there was a little boy named tim he loved to
  play in the park…")*. `make train_minimal && ./train_minimal` is
  the 40-step SGD smoke; `make train_tinystories && ./train_tinystories`
  is the full run.
- **Inference (real):** distilgpt2 + gpt2-small load from GGUF,
  generate via KV cache, parity-match HF `transformers` byte-for-byte
  on the argmax sequence. Self-contained binary (Ruby BPE in-process).
  CPU works on Mac, CUDA works on Linux + NVIDIA. See
  [HF_GPT2.md](HF_GPT2.md) for the long story.

A toy you can read top-to-bottom that happens to run a real model.
