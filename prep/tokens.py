#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tokenizers",
#   "huggingface_hub",
# ]
# ///
#
# GPT-2 BPE tokenizer wrapper. Two modes:
#
#   prep/tokens.py encode "Hello, my name is"
#     → writes whitespace-separated int IDs to data/prompt_ids.txt
#
#   prep/tokens.py decode
#     → reads data/prompt_ids.txt and prints the decoded string
#
# Used by tinynn/distilgpt2_demo.rb: encode the prompt, run inference,
# append generated IDs to the file, then decode to see English.
#
# Uses HF tokenizers (rust-backed, no transformers/torch dep).

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

IDS_PATH = Path("data/prompt_ids.txt")
CACHE    = Path("prep/_hf_cache")
REPO_ID  = "distilgpt2"


def get_tokenizer():
    os.makedirs(CACHE, exist_ok=True)
    # distilgpt2 has tokenizer.json (the fast-format one). Same vocab
    # as gpt2 / gpt2-medium / etc.
    path = hf_hub_download(REPO_ID, "tokenizer.json", cache_dir=str(CACHE))
    return Tokenizer.from_file(path)


def encode(prompt: str):
    tok = get_tokenizer()
    enc = tok.encode(prompt)
    ids = enc.ids
    os.makedirs(IDS_PATH.parent, exist_ok=True)
    IDS_PATH.write_text(" ".join(str(i) for i in ids) + "\n")
    print(f"wrote {len(ids)} ids to {IDS_PATH}", file=sys.stderr)
    print(" ".join(str(i) for i in ids))


def decode():
    tok = get_tokenizer()
    raw = IDS_PATH.read_text().strip()
    if not raw:
        print("(empty)", file=sys.stderr)
        return
    ids = [int(s) for s in raw.split()]
    text = tok.decode(ids)
    print(text)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("encode", "decode"):
        print("usage: tokens.py {encode <prompt> | decode}", file=sys.stderr)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "encode":
        prompt = " ".join(sys.argv[2:])
        encode(prompt)
    else:
        decode()


if __name__ == "__main__":
    main()
