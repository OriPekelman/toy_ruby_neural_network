#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "huggingface_hub",
#   "numpy",
# ]
# ///
#
# Reference-logits parity check. Two modes:
#
#   prep/parity.py ref
#     Loads distilgpt2 via HF transformers, runs forward on the IDs in
#     data/prompt_ids.txt, dumps the last-row logits (vocab=50257) to
#     data/ref_logits.txt (whitespace-separated floats).
#
#   prep/parity.py compare
#     Reads data/ref_logits.txt and data/ours_logits.txt, prints
#     max-abs-diff, top-5 overlap, argmax check.
#
# 'ref' is slow (downloads + loads distilgpt2) — only re-run when the
# prompt changes. 'compare' is fast.

import sys
from pathlib import Path

import numpy as np

IDS_PATH  = Path("data/prompt_ids.txt")
REF_PATH  = Path("data/ref_logits.txt")
OURS_PATH = Path("data/ours_logits.txt")
CACHE     = Path("prep/_hf_cache")
REPO_ID   = "distilgpt2"


def read_ids():
    raw = IDS_PATH.read_text().strip()
    return [int(s) for s in raw.split()]


def write_logits(path: Path, vec: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match the Ruby side's "one line, space-separated, default float repr".
    # numpy.savetxt with newline=" " puts a trailing space; do it manually.
    with open(path, "w") as f:
        f.write(" ".join(repr(float(x)) for x in vec))
        f.write("\n")


def read_logits(path: Path) -> np.ndarray:
    return np.fromstring(path.read_text(), sep=" ", dtype=np.float64)


def cmd_ref():
    import torch
    from transformers import AutoModelForCausalLM

    ids = read_ids()
    print(f"prompt: {len(ids)} tokens {ids}", file=sys.stderr)

    CACHE.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        cache_dir=str(CACHE),
        torch_dtype=torch.float32,
    )
    model.eval()

    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long)
        out = model(inp)
        # logits: [1, T, vocab]; we want the last position
        last = out.logits[0, -1, :].numpy().astype(np.float64)
    print(f"ref last-row logits: shape={last.shape}  argmax={int(np.argmax(last))}  max={float(np.max(last)):.4f}",
          file=sys.stderr)
    write_logits(REF_PATH, last)
    print(f"wrote {REF_PATH}", file=sys.stderr)


def cmd_compare():
    if not REF_PATH.exists():
        print(f"missing {REF_PATH} — run: prep/parity.py ref", file=sys.stderr)
        sys.exit(2)
    if not OURS_PATH.exists():
        print(f"missing {OURS_PATH} — run: make gpt2-parity", file=sys.stderr)
        sys.exit(2)
    ref  = read_logits(REF_PATH)
    ours = read_logits(OURS_PATH)
    if ref.shape != ours.shape:
        print(f"shape mismatch: ref={ref.shape}  ours={ours.shape}", file=sys.stderr)
        sys.exit(3)

    diff   = ours - ref
    abs_d  = np.abs(diff)
    max_d  = float(abs_d.max())
    mean_d = float(abs_d.mean())

    # Find where the worst disagreement is.
    worst_idx = int(np.argmax(abs_d))

    # Argmax + top-5
    am_ref  = int(np.argmax(ref))
    am_ours = int(np.argmax(ours))
    top5_ref  = np.argsort(ref)[::-1][:5].tolist()
    top5_ours = np.argsort(ours)[::-1][:5].tolist()
    top5_overlap = len(set(top5_ref) & set(top5_ours))

    print("=== parity report ===")
    print(f"  shape:          {ref.shape}")
    print(f"  max abs diff:   {max_d:.6e}  at idx={worst_idx}")
    print(f"                  ref={ref[worst_idx]:.6f}  ours={ours[worst_idx]:.6f}")
    print(f"  mean abs diff:  {mean_d:.6e}")
    print(f"  argmax ref:     {am_ref}  (logit {ref[am_ref]:.4f})")
    print(f"  argmax ours:    {am_ours} (logit {ours[am_ours]:.4f})")
    print(f"  argmax match:   {am_ref == am_ours}")
    print(f"  top-5 ref:      {top5_ref}")
    print(f"  top-5 ours:     {top5_ours}")
    print(f"  top-5 overlap:  {top5_overlap}/5")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("ref", "compare"):
        print("usage: parity.py {ref | compare}", file=sys.stderr)
        sys.exit(2)
    {"ref": cmd_ref, "compare": cmd_compare}[sys.argv[1]]()


if __name__ == "__main__":
    main()
