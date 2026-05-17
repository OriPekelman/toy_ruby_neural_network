#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers", "huggingface_hub", "tiktoken", "blobfile"]
# ///
#
# Host-side tokenizer for Qwen2.5. Same shape as prep/smollm2_tokens.py.
# Qwen uses tiktoken-style BPE with a different merges table; the
# Ruby side never sees tokens, only integer IDs.
#
# Usage:
#   prep/qwen25_tokens.py encode "Hello, my name is"
#     → writes data/qwen25_prompt_ids.txt
#   prep/qwen25_tokens.py decode
#     → reads data/qwen25_prompt_ids.txt and prints detokenized text

import sys
from pathlib import Path
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
IDS   = Path("data/qwen25_prompt_ids.txt")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("encode", "decode"):
        print("usage: qwen25_tokens.py encode <text>  |  decode", file=sys.stderr)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(MODEL)

    if sys.argv[1] == "encode":
        text = " ".join(sys.argv[2:])
        ids  = tok.encode(text, add_special_tokens=False)
        IDS.parent.mkdir(exist_ok=True)
        IDS.write_text(" ".join(str(i) for i in ids) + "\n")
        print(f"wrote {len(ids)} ids to {IDS}")
        print(" ".join(str(i) for i in ids))
        return

    raw = IDS.read_text().strip()
    ids = [int(s) for s in raw.split()]
    text = tok.decode(ids, skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
