# demos/

End-to-end Ruby drivers. Source here; build target names are the
same as the file base; binaries land back here. **Run from the repo
root** so they can find `data/*.gguf`, prompts, BPE tables.

| Source | Build | What |
|---|---|---|
| `gpt2.rb`            | `make gpt2`             | DistilGPT2 / GPT-2 via `Toy::GPT2` (native Mat) |
| `smollm2.rb`         | `make smollm2`          | SmolLM2-135M via `Toy::SmolLM2` (native Mat) |
| `smollm2_kv.rb`      | `make smollm2_kv`       | SmolLM2-135M FFI KV-cache (CPU) — **77 tok/s on M2 Air** |
| `smollm2_kv_cuda.rb` | `make smollm2_kv_cuda`  | SmolLM2-135M FFI KV-cache (CUDA, GB10) — **89 tok/s** |
| `tinyllama.rb`       | `make tinyllama`        | TinyLlama-1.1B via `Toy::SmolLM2` (native Mat) |
| `tinyllama_kv.rb`    | `make tinyllama_kv`     | TinyLlama-1.1B FFI KV-cache (CPU) — known f32-precision issue |
| `tinyllama_kv_cuda.rb` | `make tinyllama_kv_cuda` | TinyLlama-1.1B FFI KV-cache (CUDA) — known f32-precision issue |
| `train.rb`           | `make train`            | TinyStories from-scratch training via `Toy::Trainer` |
| `algorithm_cards.rb` | `make algorithm_cards`  | Print the Phuong–Hutter algorithm cards (no inference) |

## Quickstart

```sh
make setup-ggml                                # one-time, builds vendored ggml
./prep/convert_smollm2_to_gguf.py              # ~30 s; writes data/smollm2-135m-f32.gguf
./prep/smollm2_tokens.py encode "Once upon a time"
make smollm2_kv && ./demos/smollm2_kv          # ~77 tok/s on M2 Air
./prep/smollm2_tokens.py decode
# → "Once upon a time, there was a little girl named Lily..."
```

CUDA demos require `make setup-ggml-cuda` plus a working CUDA toolkit
(typically run on the gx10 side of the Mac → gx10 dev workflow).

The TinyLlama FFI paths are present but produce NaN logits at full
depth due to f32 overflow on this specific checkpoint. Use
`demos/tinyllama` (native Mat) for correct output; see
[`../docs/tinyllama-known-issue.md`](../docs/tinyllama-known-issue.md).
