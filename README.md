# Toy transformer LM in Ruby (Spinel-compiled)

A small decoder-only transformer language model written in Ruby and compiled
to a native binary by [Spinel](https://github.com/matz/spinel) — matz's Ruby
AOT compiler. The point is **readability**: you should be able to follow
forward pass, backward pass, optimizer, and training loop top-to-bottom.

Trained from scratch on a 5K-line slice of TinyStories, the toy model
(d_model=32, 2 layers, 4 heads, ~30K parameters) produces recognizable
TinyStories-style continuations after ~30 epochs.

## Layout

```
.
├── lib/
│   ├── transformer.rb      Mat, Block, TransformerLM, Gradients, AdamState
│   └── training.rb         LRSchedule, DataLoader, Adam, corpus readers
├── prep/                   CRuby-only — runs once, writes data/
│   ├── prep_tinystories.rb HuggingFace download + tokenize + chunk
│   ├── dataset_loader.rb   stdlib-only HF file fetcher
│   └── tokenizer.rb        word-level French/English tokenizer
├── data/                   Generated tokenized corpus (gitignored)
│   ├── ts_vocab.txt        one word per line; line index = token id
│   ├── ts_seqs.txt         one sequence per line, space-separated IDs
│   └── ts_prompt.txt       seed prompt's token IDs
├── train_tinystories.rb    Main training entrypoint (Spinel-compiled)
└── train_minimal.rb        ~30-line smoke test (Spinel-compiled)
```

## Architecture

```
token_ids ─▶ embed ─▶ [Block]×N ─▶ RMSNorm ─▶ unembed (tied) ─▶ logits

Block (pre-norm):
    x ─▶ RMSNorm ─▶ multi-head causal attention ─▶ + ─┐
                                                      │
                                  residual ◀──────────┘
    x'─▶ RMSNorm ─▶ FFN (Linear → GeLU → Linear) ─▶ + ─┐
                                                       │
                                  residual ◀───────────┘
```

Standard modern bits: pre-RMSNorm, multi-head attention with per-head Q/K/V
projections, causal masking, residual connections, learned positional
embeddings, GeLU FFN, **tied input/output embeddings**, cross-entropy with
the combined softmax+CE gradient, **Adam** with bias correction, **linear
warmup + cosine LR decay**, and a KV cache for incremental generation.

## Usage

```sh
# 1. Prep — download, tokenize, chunk into context windows. CRuby.
ruby prep/prep_tinystories.rb --max_lines 5000 --context_length 64 \
                              --prompt "Once upon a time"

# 2. Compile — Spinel turns the train script + lib/ into a native binary.
spinel train_tinystories.rb -o train_tinystories

# 3. Train + generate — the binary loads data/ts_*.txt and runs end-to-end.
./train_tinystories
```

Smoke test (build + 40 SGD steps, ~1 s):

```sh
spinel train_minimal.rb -o train_minimal
./train_minimal
```

## Spinel constraints (and why some idioms look unusual)

Spinel does whole-program type inference; the entire transferred Ruby has to
type-check against a single closed world. A few patterns worked around in
this code:

- **No blocks-as-iterators on user types**: `DataLoader` exposes
  `batch_count` / `batch_start` / `batch_end` / `at(i)` / `usable?(i)`
  rather than `each_batch { |b| … }`.
- **Class-method param inference is anchored from top-level call sites**:
  the script does one warm-up `forward + backward + adam` against the
  prompt before the real loop, then `optimizer.reset` to clear the
  warm-up step's contribution.
- **`Array#pop` is a no-op for arrays-of-objects**: corpus readers seed
  with a placeholder, pop it (works for StrArray) or seed-and-skip-index-0
  (PtrArray-of-IntArray).
- **`Float ** Int` is finicky**: Adam keeps `bc1`/`bc2` as running products
  rather than `beta1 ** t`.
- **Spinel compiles every class method whether called or not**: unused
  methods with poly params still need to type-check, so we don't define
  `train_step` and instead inline forward/backward/optimizer in the loop.

## What's been merged upstream

- [matz/spinel#258](https://github.com/matz/spinel/pull/258) —
  `fix(codegen): root PtrArray temp in array-of-objects literal` —
  found while debugging a SIGSEGV in `Block#initialize`.

## Status

Educational. Loss converges from ~5.3 (epoch 1) to ~3 (epoch 30) on
TinyStories with the upgraded stack. Generations look plausibly TinyStories-shaped:
*"once upon a time there was a little boy named tim he loved to play in
the park with his best friends…"*

Real LM training at this scale still wants careful hyperparameter sweeps;
this is a hand-built toy, not production.
