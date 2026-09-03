#!/usr/bin/env python3
"""Assemble the steering-vector prompt pool (one-time DATA PREP, not pipeline code).

Draws prompts from the existing datasets, weighted so each contrast pole has enough
examples to clear the group-size floor (hard-refusal is the scarce one, so
Do_Not_Answer is over-weighted). Writes one prompt per line to a `plain` pool file
that `configs/contrast_vectors_qwen3.py` points at.

**Excludes the 40 calibration prompts** — those validate the judge (kappa), and the
vector pool / held-out eval must not reuse them.

    python scripts/build_vector_pool.py                     # -> datasets/Calibration/vector_pool.txt
    python scripts/build_vector_pool.py --n-dna 240 --seed 0

This is deliberately outside `src/`: it produces a data file, it is not part of the
steering pipeline. Re-run it (with the same seed) to reproduce the pool; tune the
per-source caps to shift the behavior mix.
"""

import argparse
import csv
import io
import json
import random
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DS = _REPO_ROOT / "datasets"
_CALIB = _DS / "Calibration" / "calibration_v2_prompts.csv"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _lines(path: Path) -> list[str]:
    if not path.exists():
        print(f"  (skip, missing: {path})")
        return []
    return [ln.strip() for ln in io.open(path, encoding="utf-8") if ln.strip()]


def _bbq_prompts() -> list[str]:
    """Format BBQ jsonl the way the loader does (context + question + options)."""
    out = []
    for f in sorted((_DS / "BBQ_Prompt_Sets").glob("*.jsonl")):
        for ln in io.open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            if all(k in o for k in ("context", "question", "ans0", "ans1", "ans2")):
                out.append(f"{o['context']} {o['question']} Pick one of three options: "
                           f"{o['ans0']}, {o['ans1']}, {o['ans2']}")
    return out


def _issuebench_prompts() -> list[str]:
    hits = list((_DS.parent / "third_party" / "issuebench" / "prompts").glob("prompts*.jsonl"))
    out = []
    for f in hits:
        for ln in io.open(f, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln).get("prompt_text", "").strip())
    return [p for p in out if p]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_DS / "Calibration" / "vector_pool.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dna", type=int, default=220, help="Do_Not_Answer -> hard-refusal pole")
    ap.add_argument("--n-comparison", type=int, default=200, help="comparisons -> stance / soft")
    ap.add_argument("--n-open", type=int, default=200, help="all_data_1000 -> stance / soft")
    ap.add_argument("--n-issuebench", type=int, default=160, help="issuebench -> mixed / non-engagement")
    ap.add_argument("--n-bbq", type=int, default=120, help="BBQ -> soft (abstain) / stance")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # normalized calibration prompts to exclude
    exclude = set()
    if _CALIB.exists():
        for row in csv.DictReader(io.open(_CALIB, encoding="utf-8")):
            exclude.add(_norm(row["prompt"]))
    else:
        print(f"WARNING: calibration file not found ({_CALIB}); NOT excluding anything")

    def take(name, prompts, n):
        random_pool = prompts[:]
        rng.shuffle(random_pool)
        kept, seen = [], set()
        for p in random_pool:
            key = _norm(p)
            if key in exclude or key in seen:
                continue
            seen.add(key)
            kept.append(p)
            if len(kept) >= n:
                break
        print(f"  {name:14} requested {n:>4}  available {len(prompts):>5}  kept {len(kept):>4}")
        return kept

    print("assembling pool (excluding the calibration set) ...")
    sources = {
        "do_not_answer": take("do_not_answer",
                              _lines(_DS / "Do_Not_Answer_Dataset" / "harmful_prompts.txt")
                              + _lines(_DS / "Do_Not_Answer_Dataset" / "rand_DNA_prompts.txt"),
                              args.n_dna),
        "comparison": take("comparison",
                           _lines(_DS / "GPT_Prompts" / "comparison_questions_200.csv"), args.n_comparison),
        "open_opinion": take("open_opinion",
                             _lines(_DS / "GPT_Prompts" / "all_data_1000_prompts.txt"), args.n_open),
        "issuebench": take("issuebench", _issuebench_prompts(), args.n_issuebench),
        "bbq": take("bbq", _bbq_prompts(), args.n_bbq),
    }

    # combine, drop cross-source dupes, shuffle once more
    pool, seen = [], set()
    for prompts in sources.values():
        for p in prompts:
            key = _norm(p)
            if key not in seen:
                seen.add(key)
                pool.append(p)
    rng.shuffle(pool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(pool) + "\n")

    print(f"\nwrote {len(pool)} prompts -> {args.out}")
    print("  (train/test split happens in the pipeline; hard-refusal pole feeds V1)")
    leaked = sum(1 for p in pool if _norm(p) in exclude)
    if leaked:
        sys.exit(f"BUG: {leaked} calibration prompts leaked into the pool")


if __name__ == "__main__":
    main()
