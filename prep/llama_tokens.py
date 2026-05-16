#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers", "huggingface_hub", "sentencepiece"]
# ///
#
# Host-side tokenizer shim for any HuggingFace llama-family model.
# Companion to prep/smollm2_tokens.py — same shape but parameterized
# on --model so it works for TinyLlama, Qwen2.5, etc.
#
# Usage:
#   prep/llama_tokens.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --ids data/tinyllama_prompt_ids.txt encode "Once upon a time"
#   prep/llama_tokens.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --ids data/tinyllama_prompt_ids.txt decode

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id (used only for the tokenizer)")
    ap.add_argument("--ids",   required=True,
                    help="Path to the prompt-ids file (read/write)")
    ap.add_argument("action", choices=["encode", "decode"])
    ap.add_argument("text", nargs="*",
                    help="Text to encode (encode mode only)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids_path = Path(args.ids)

    if args.action == "encode":
        text = " ".join(args.text)
        ids  = tok.encode(text, add_special_tokens=False)
        ids_path.parent.mkdir(exist_ok=True)
        ids_path.write_text(" ".join(str(i) for i in ids) + "\n")
        print(f"wrote {len(ids)} ids to {ids_path}")
        print(" ".join(str(i) for i in ids))
        return

    raw = ids_path.read_text().strip()
    ids = [int(s) for s in raw.split()]
    text = tok.decode(ids, skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
