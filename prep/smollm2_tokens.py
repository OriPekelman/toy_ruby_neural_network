#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers", "huggingface_hub"]
# ///
#
# Host-side tokenizer shim for SmolLM2. v1 of the SmolLM2 path uses
# Python to encode/decode (pure-Ruby SentencePiece would be a separate
# project). The Ruby model only deals with integer token IDs.
#
# Usage:
#   prep/smollm2_tokens.py encode "Once upon a time"
#     → writes data/smollm2_prompt_ids.txt (one line, space-separated ints)
#   prep/smollm2_tokens.py decode
#     → reads data/smollm2_prompt_ids.txt and prints the detokenized text

import sys
from pathlib import Path
from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"
IDS   = Path("data/smollm2_prompt_ids.txt")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("encode", "decode"):
        print("usage: smollm2_tokens.py encode <text>  |  decode", file=sys.stderr)
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

    # decode
    raw = IDS.read_text().strip()
    ids = [int(s) for s in raw.split()]
    text = tok.decode(ids, skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
