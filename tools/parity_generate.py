"""Parity ladder, rung 2: regenerate archived responses and diff the text.

Holds the prompts fixed and re-runs generation through `models.generate` — the
exact code path the train phase uses. Greedy decoding is deterministic, so a
faithful port should reproduce the archived text nearly character-for-character.
That makes this the sharpest test available of the three things hardest to verify
by reading code: chat-template application, BOS/prompt stripping, and left-padding
behavior in a batch.

Residual fp16/kernel nondeterminism and ten months of HF weight revisions mean
100% exact match is not the bar. The signal is the *shape* of the mismatch:

- high exact-match, a few late-token divergences  -> faithful port, numeric drift
- every response sharing a mangled prefix/suffix  -> the stripping bug
- length-correlated corruption across a batch     -> the left-padding bug
- wholesale different text                        -> template or system-prompt mismatch

Batch size matters and defaults to the notebook's 32: padding is per-batch, so a
different batch size changes the padding and would confound the comparison.

Usage:
    python tools/parity_generate.py [--limit N] [--batch-size 32]
"""

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer.config import DEFAULT_SYS
from tools.parity_rejudge import DEFAULT_LOG, parse_archive

SNAPSHOT = "datasets/Snapshots/log_103_comparison_200.json"
MODEL_KEY = "qwen-1.8b"
MAX_TOKENS = 128


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--limit", type=int, help="only the first N prompts")
    p.add_argument("--batch-size", type=int, default=32,
                   help="must match the archived run (32) for a fair comparison")
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    archive = parse_archive(root / DEFAULT_LOG)
    prompts = [q for q, _, _ in archive]
    expected = [r for _, r, _ in archive]
    if args.limit:
        prompts, expected = prompts[: args.limit], expected[: args.limit]

    from src.bias_steer import models
    from src.bias_steer.registry import MODELS

    print(f"loading {MODEL_KEY} ...", flush=True)
    loaded = models.load_model(MODELS[MODEL_KEY])
    print(f"  device={loaded.device}  n_layers={loaded.model.cfg.n_layers}  "
          f"d_model={loaded.model.cfg.d_model}", flush=True)
    print(f"  bos_token={loaded.tokenizer.bos_token!r}  "
          f"pad_token={loaded.tokenizer.pad_token!r}", flush=True)

    got: list[str] = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i: i + args.batch_size]
        got.extend(models.generate(loaded, batch, MAX_TOKENS, DEFAULT_SYS))
        print(f"  generated {len(got)}/{len(prompts)}", flush=True)

    exact = sum(g.strip() == e.strip() for g, e in zip(got, expected))
    sims = [similarity(g.strip(), e.strip()) for g, e in zip(got, expected)]
    prefixes = [common_prefix(g.strip(), e.strip()) for g, e in zip(got, expected)]

    print(f"\n=== rung 2: generation parity (n={len(got)}) ===")
    print(f"  exact match      : {exact}/{len(got)} ({exact / len(got):.1%})")
    print(f"  mean similarity  : {sum(sims) / len(sims):.3f}")
    print(f"  median similarity: {sorted(sims)[len(sims) // 2]:.3f}")
    print(f"  mean common prefix chars: {sum(prefixes) / len(prefixes):.0f}")
    print(f"  responses with empty output: {sum(not g.strip() for g in got)}")

    # The padding bug, if present, correlates corruption with prompt length within
    # a batch: the longest prompt in a batch gets no padding, the shortest most.
    print("\n  similarity by prompt-length quartile (padding-bug tell):")
    order = sorted(range(len(got)), key=lambda i: len(prompts[i]))
    q = max(1, len(order) // 4)
    for label, chunk in [("shortest 25%", order[:q]), ("longest 25%", order[-q:])]:
        m = sum(sims[i] for i in chunk) / len(chunk)
        print(f"    {label}: mean similarity {m:.3f}")

    worst = sorted(range(len(got)), key=lambda i: sims[i])[:3]
    print("\n  worst 3 mismatches:")
    for i in worst:
        print(f"    [{i}] similarity {sims[i]:.3f}  prompt: {prompts[i][:80]}")
        print(f"        archive: {expected[i][:150]!r}")
        print(f"        got    : {got[i][:150]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
