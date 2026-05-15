# demos/

End-to-end Ruby drivers, one per build target. Each compiles via
Spinel to a native binary in this directory; **run the binary from
the repo root** so it can find `data/*.gguf`, `data/prompt.txt`, etc.

| Source | Build target | What it does |
|---|---|---|
| `train_minimal.rb` | `make train_minimal` | 40-step SGD smoke |
| `train_tinystories.rb` | `make train_tinystories` | From-scratch toy training run |
| `inference_demo.rb` | `make inference_demo` | Random-init forward + parity (CPU FFI) |
| `inference_demo_cuda.rb` | `make inference_demo_cuda` | Same, CUDA |
| `distilgpt2_demo.rb` | `make distilgpt2_demo` | Native Mat (f64) DistilGPT2 forward |
| `distilgpt2_demo_ffi.rb` | `make distilgpt2_demo_ffi` | FFI full-forward, CPU |
| `distilgpt2_demo_kv.rb` | `make distilgpt2_demo_kv` | KV-cache decode, CPU |
| `distilgpt2_demo_text.rb` | `make distilgpt2_demo_text` | KV decode + Ruby BPE; text in / out |
| `distilgpt2_demo_ffi_cuda.rb` | `make distilgpt2_demo_ffi_cuda` | FFI full-forward, CUDA |
| `distilgpt2_demo_kv_cuda.rb` | `make distilgpt2_demo_kv_cuda` | KV-cache decode, CUDA |

## Quickstart

```sh
make setup-ggml                                          # one-time
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 \
                                    --out data/gpt2-f32.gguf
./prep/dump_bpe.py
echo "Once upon a time" > data/prompt.txt
make distilgpt2_demo_text
./demos/distilgpt2_demo_text                             # invoked from repo root
```

CUDA demos require `make setup-ggml-cuda` plus a working CUDA toolkit.
