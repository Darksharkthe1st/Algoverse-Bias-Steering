"""Read a held-out V2 eval's results.csv and report the stance<->soft-refusal shift.

The eval run (configs/apply_v2min_heldout_qwen3.py) labels every INITIAL /
STEERED_POS / STEERED_NEG response with the FULL 9-way judge v2.1. summary.md's
"steering quality" line is meaningless under a 9-way judge (it assumes 2 labels),
so this script is the real readout:

  * collapse each 9-way verdict to the behaviour view (judges.v2.collapse) and pool
    stance-factual + stance-evaluative -> `stance` (the SAME pooling the V2 vector
    was built from),
  * per condition, print the behaviour distribution,
  * print the per-example INITIAL->STEERED_POS and INITIAL->STEERED_NEG confusion
    over {stance, soft-refusal} (the per-example distributions the 2026 steering-claim
    bar asks for, CLAUDE.md §5),
  * headline: does +opinion raise the stance rate, and -neutral raise the
    soft-refusal rate, vs INITIAL.

Usage (from the repo root, so `src` is importable — no torch needed):

    python scripts/analyze_v2min_eval.py runs/<eval-id>/results.csv
"""

import csv
import sys
from collections import Counter, defaultdict

from src.bias_steer.judges.v2 import collapse, STANCE_POOL

STANCE = "stance"


def pooled(verdict: str) -> str:
    """9-way verdict -> behaviour view, with the two stance labels pooled to `stance`."""
    c = collapse(verdict)
    return STANCE if c in STANCE_POOL else c


def load(path):
    """results.csv -> {example_id: {condition: pooled_label}} and the condition order seen."""
    by_ex = defaultdict(dict)
    conditions = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            if cond not in conditions:
                conditions.append(cond)
            by_ex[row["example_id"]][cond] = pooled(row["verdict"])
    return by_ex, conditions


def dist(by_ex, cond):
    c = Counter(labels[cond] for labels in by_ex.values() if cond in labels)
    return c, sum(c.values())


def rate(counter, total, label):
    return (counter.get(label, 0) / total) if total else 0.0


def confusion(by_ex, from_cond, to_cond, labels=(STANCE, "soft-refusal")):
    """Per-example from_cond->to_cond transitions restricted to `labels` on both ends."""
    m = Counter()
    for ex in by_ex.values():
        if from_cond in ex and to_cond in ex:
            m[(ex[from_cond], ex[to_cond])] += 1
    return m


def _print_dist(name, counter, total):
    print(f"\n{name}  (n={total})")
    for label, n in counter.most_common():
        print(f"    {label:<16} {n:>4}  {n / total:6.1%}" if total else f"    {label:<16} {n:>4}")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    by_ex, conditions = load(argv[1])

    # Conditions are the schema strings (initial / steered_pos / steered_neg); match
    # by substring so this is robust to exact casing.
    def find(sub):
        return next((c for c in conditions if sub in c.lower()), None)

    init, pos, neg = find("init"), find("pos"), find("neg")

    for cond in (init, pos, neg):
        if cond is None:
            continue
        c, n = dist(by_ex, cond)
        _print_dist(cond, c, n)

    if init and pos:
        ic, itot = dist(by_ex, init)
        pc, ptot = dist(by_ex, pos)
        print(f"\n+opinion (STEERED_POS):  stance {rate(ic, itot, STANCE):.1%} -> "
              f"{rate(pc, ptot, STANCE):.1%}")
        print("  INITIAL->STEERED_POS confusion over {stance, soft-refusal}:")
        for (a, b), k in confusion(by_ex, init, pos).most_common():
            print(f"    {a:<14} -> {b:<14} {k:>4}")

    if init and neg:
        ic, itot = dist(by_ex, init)
        nc, ntot = dist(by_ex, neg)
        print(f"\n-neutral (STEERED_NEG):  soft-refusal {rate(ic, itot, 'soft-refusal'):.1%} -> "
              f"{rate(nc, ntot, 'soft-refusal'):.1%}")
        print("  INITIAL->STEERED_NEG confusion over {stance, soft-refusal}:")
        for (a, b), k in confusion(by_ex, init, neg).most_common():
            print(f"    {a:<14} -> {b:<14} {k:>4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
