#!/usr/bin/env python3
"""
Usage:
    uv run eval.py                         # full run (all 1,838 examples)
    uv run eval.py --subset 10             # only 10 % of the data
    uv run eval.py --data-dir ./my-data    # use a custom data directory
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from rich import print

# ---------------------------------------------------------------------------
# PIQA data source
# ---------------------------------------------------------------------------

PIQA_BASE_URL = "https://yonatanbisk.com/piqa/data"
PIQA_FILES = {
    "valid.jsonl": f"{PIQA_BASE_URL}/valid.jsonl",
    "valid-labels.lst": f"{PIQA_BASE_URL}/valid-labels.lst",
}

DEFAULT_DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_piqa(data_dir: Path) -> None:
    """Ensure all PIQA data files exist under *data_dir*."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in PIQA_FILES.items():
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading [bold]{name}[/] ...")
        urllib.request.urlretrieve(url, dest)
    print("[green]Data ready.[/]")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_piqa(data_dir: Path, subset: int) -> list[dict]:
    """Return PIQA validation examples, with ``label`` injected.

    *subset*: percentage (1–100).  100 means all examples.
    """
    json_path = data_dir / "valid.jsonl"
    labels_path = data_dir / "valid-labels.lst"

    with open(json_path) as f:
        examples = [json.loads(line) for line in f]
    with open(labels_path) as f:
        labels = [int(line.strip()) for line in f]

    if len(examples) != len(labels):
        raise ValueError(
            f"Example/label count mismatch: {len(examples)} vs {len(labels)}"
        )

    for i, label in enumerate(labels):
        examples[i]["label"] = label

    if subset < 100:
        n = math.ceil(len(examples) * subset / 100)
        examples = examples[:n]

    return examples


# ---------------------------------------------------------------------------
# MLX inference helpers
# ---------------------------------------------------------------------------


def score_choice(model, tokenizer, ctx: str, choice: str) -> float:
    """Log-probability of *choice* given *ctx* under the model."""
    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
    full_tokens = tokenizer.encode(ctx + " " + choice, add_special_tokens=False)
    choice_tokens = full_tokens[len(ctx_tokens) :]
    if not choice_tokens:
        return -float("inf")

    logits = model(mx.array([full_tokens]))
    log_probs = mx.log(mx.softmax(logits[0], axis=-1))

    total = 0.0
    for i, token in enumerate(choice_tokens):
        pos = len(ctx_tokens) - 1 + i
        total += log_probs[pos, token].item()
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolLM2-135M-Instruct on PIQA via MLX"
    )
    parser.add_argument("--model", default="mlx-community/SmolLM2-135M-Instruct")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory to store/load PIQA data (default: ./data)",
    )
    parser.add_argument(
        "--subset",
        default=100,
        type=int,
        help="Percentage of data to evaluate (1–100).  Default: 100 (all).",
    )
    args = parser.parse_args()

    if not 1 <= args.subset <= 100:
        parser.error("--subset must be between 1 and 100")

    data_dir = Path(args.data_dir)
    download_piqa(data_dir)
    data = load_piqa(data_dir, args.subset)

    print(f"Loading model: {args.model} ...")
    model, tokenizer, *_ = load(args.model)
    print(f"Model loaded. Evaluating {len(data)} examples ...\n")

    correct = 0
    for i, item in enumerate(data):
        s0 = score_choice(model, tokenizer, item["goal"], item["sol1"])
        s1 = score_choice(model, tokenizer, item["goal"], item["sol2"])
        pred = 0 if s0 > s1 else 1
        if pred == item["label"]:
            correct += 1
        if (i + 1) % 100 == 0 or i + 1 == len(data):
            print(f"  {i + 1}/{len(data)} ({100 * (i + 1) // len(data)}%)")
    total = len(data)
    acc = correct / total * 100

    print()
    print("=" * 40)
    print(f"Examples : {total}")
    print(f"Correct  : {correct}")
    print(f"Accuracy : {acc:.1f}%")
    print("=" * 40)
    print("(Model card: PIQA = 66.3%)")


if __name__ == "__main__":
    main()
