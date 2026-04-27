# Toy neural networks in pure Ruby

Two from-scratch implementations, no dependencies beyond the Ruby standard
library. The whole point is to be readable: you should be able to follow how
backpropagation works by reading the code top to bottom.

## What's in here

| File | Lines | What it is |
|---|---:|---|
| `transformer.rb` | ~880 | Decoder-only Transformer LM with multi-head attention, RMSNorm, residuals, FFN, KV cache, Adam |
| `neural_network.rb` | ~380 | Simple autoencoder — useful as a minimal backprop teaching example |
| `tokenizer.rb` | ~30 | Word-level French tokenizer (handles `c'est`, `aujourd'hui`, etc.) |
| `run.rb` | ~110 | CLI driver |
| `test_gradients.rb` | ~110 | Numerical gradient check against the transformer's analytic backward |

The transformer is the main thing. The autoencoder is preserved because its
single forward + backward pass is the simplest possible illustration of
gradient accumulation — the same accumulator pattern then carries over to the
transformer.

## Architecture (transformer)

```
token_ids ─▶ embed ─▶ [Block]×N ─▶ RMSNorm ─▶ LM head ─▶ logits

Block (pre-norm):
    x ─▶ RMSNorm ─▶ multi-head self-attention (Q,K,V,O, causal mask) ─▶ + ─┐
                                                                           │
                                                       residual ◀──────────┘
    x'─▶ RMSNorm ─▶ FFN (Linear → ReLU → Linear) ─▶ + ─┐
                                                       │
                                  residual ◀───────────┘
```

Modern bits included: pre-norm, RMSNorm (LLaMA-style, simpler than LayerNorm),
multi-head attention with per-head Q/K/V projections, causal masking, residual
connections, learned positional embeddings, cross-entropy loss with the
combined softmax+CE gradient shortcut, Adam optimizer with bias correction,
and a KV cache for incremental autoregressive generation.

## Backpropagation, briefly

Forward pass caches every intermediate activation. Backward pass walks them
in reverse, with each layer-level helper returning `(d_input, grads_for_params)`.
Training is mini-batch SGD with explicit gradient accumulation:

1. zero an accumulator (one slot per parameter)
2. for each example: forward, backward, **add** grads into the accumulator
3. divide by batch size → mean gradient
4. apply once via Adam

Averaging the *gradients* (not the raw errors) is what makes a mini-batch step
equivalent to one step on the mean loss.

A numerical-vs-analytic gradient check (`ruby test_gradients.rb`) confirms the
backward pass: max absolute error is ~1e-8.

## Usage

```sh
# Transformer (default)
ruby run.rb --epochs 30 --learning_rate 0.005 \
            --d_model 16 --d_ff 32 --n_heads 2 --n_layers 2 \
            --context_length 8 --corpus minimal --prompt "un est" --num_tokens 4

# Autoencoder
ruby run.rb --model autoencoder --epochs 20 --learning_rate 0.05 \
            --hidden_size 8 --latent_size 4 --corpus minimal --prompt "un"
```

Models are cached on disk after training (transformer in plain text, autoencoder
in `Marshal`). Re-running the same hyperparameters loads the cached model.

## Spinel-friendly subset

The transformer (and tokenizer) avoid `Marshal`, `instance_variable_set`, and
all metaprogramming so they can plausibly compile under
[Spinel](https://github.com/matz/spinel). The autoencoder still uses stdlib
`Matrix` and `Marshal` and is CRuby-only.

## Status

Educational. The minimal corpus (5 lines, 7-token vocabulary) trains to loss
~0.1 in a fraction of a second. Larger corpora work but a real LM at this
scale would still want sweeping over learning rate, batch size, and epoch
count — this is a hand-built toy, not a production system.
