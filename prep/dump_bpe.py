#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
# ]
# ///
#
# Download GPT-2's BPE artefacts and dump them as plain TSV that the
# Ruby BPE encoder/decoder can read without a JSON parser:
#
#   data/gpt2-bpe-vocab.tsv    <id>\t<token>      × 50257
#   data/gpt2-bpe-merges.tsv   <rank>\t<A>\t<B>   × ~50000
#
# Tokens already include GPT-2's byte→unicode encoding (so e.g. " " → "Ġ").
# The Ruby loader doesn't need to do JSON parsing or unicode escapes.

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "gpt2"   # gpt2 / distilgpt2 share the same BPE tokenizer
OUT     = Path("data")
CACHE   = Path("prep/_hf_cache")

CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

vocab_path  = hf_hub_download(REPO_ID, "vocab.json",  cache_dir=str(CACHE))
merges_path = hf_hub_download(REPO_ID, "merges.txt",  cache_dir=str(CACHE))

# vocab.json: {"token": id, …}
with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)

# Write sorted by id so the Ruby side can verify density / size easily.
by_id = sorted(vocab.items(), key=lambda kv: kv[1])
with open(OUT / "gpt2-bpe-vocab.tsv", "w", encoding="utf-8") as f:
    for tok, idx in by_id:
        f.write(f"{idx}\t{tok}\n")

# merges.txt: header line + "A B" per merge, in priority order (rank 0 first).
with open(merges_path, encoding="utf-8") as f:
    lines = f.read().splitlines()
# First line is "#version: 0.2" — skip headers.
merges = [ln for ln in lines if ln and not ln.startswith("#")]
with open(OUT / "gpt2-bpe-merges.tsv", "w", encoding="utf-8") as f:
    for rank, line in enumerate(merges):
        a, b = line.split(" ", 1)
        f.write(f"{rank}\t{a}\t{b}\n")

print(f"vocab:  {len(vocab):>6}  → {OUT/'gpt2-bpe-vocab.tsv'}")
print(f"merges: {len(merges):>6}  → {OUT/'gpt2-bpe-merges.tsv'}")

# byte→unicode table. GPT-2 maps every byte 0..255 to a 'visible'
# unicode char so tokens are pure-text and round-trip cleanly through
# tokenizer files. Reproducing the original mapping rule here so
# Ruby doesn't have to compute it (and to keep both sides in sync).
def bytes_to_unicode_map():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

bmap = bytes_to_unicode_map()
with open(OUT / "gpt2-bpe-bytechars.tsv", "w", encoding="utf-8") as f:
    for b in range(256):
        f.write(f"{b}\t{bmap[b]}\n")
print(f"bytes:  256     → {OUT/'gpt2-bpe-bytechars.tsv'}")
