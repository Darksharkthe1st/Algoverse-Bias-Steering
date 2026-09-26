"""Item-bootstrap CIs and paired per-item comparisons on a judged CSV.

Lane B item 2 (`docs/HANDOFF_lane_a.md`): every reported rate carries an
item-bootstrap CI, because the judge drifts ±1–2/100 even at temperature 0 and a
single-run margin inside that band is not a result.

Two modes, one file, because they answer the same question at different
granularities:

    # a rate per arm, with a CI
    python scripts/bootstrap_ci.py runs/<id>/judged_v2.1.csv --positive stance

    # the paired steer-vs-prompt margin: beat_rate + the 2x2 that explains it
    python scripts/bootstrap_ci.py runs/<id>/judged_v2.1.csv --positive stance \
        --paired steered_pos,prompt_pos

    # two runs on the SAME items: opinion direction vs the Exp-2 random control
    python scripts/bootstrap_ci.py runs/<exp1>/judged_v2.1.csv runs/<exp2>/judged_v2.1.csv \
        --positive stance --paired <exp1>:steered_pos,<exp2>:steered_pos

`--positive` names the target behaviour and accepts a POOL of labels, because the
v2.1 rubric splits "took a side" into `stance-factual` and `stance-evaluative`. The
pool `stance` is the natural shorthand for their union (`docs/judges/judge_v2.1.md`,
the V2/V3 contrast rows) — pooling has to be explicit, since silently reporting only
`stance-evaluative` would undercount taking a side on a factual item.

Several judged CSVs may be given at once. When they come from different runs, arms are
tagged `<run>:<condition>` so two runs' `steered_pos` cannot be silently pooled — which
is what makes the Exp-1-vs-Exp-2 comparison (same 200 items, same dose, opinion
direction vs random control) a per-item paired test rather than two separate rates.

The paired mode is the honest form of "did the vector beat prompting": it resamples
ITEMS (not arms) so the CI carries the per-item pairing, and it prints the four
McNemar cells. A near-zero margin with many discordant pairs means the two methods
are COMPLEMENTARY — each winning a different subset — which is a different finding
from "they behave the same", and the aggregate margin alone cannot tell them apart.

Deliberately duplicates no logic from `metrics.beat_rate`: that computes over live
`Result` objects inside a run, this computes over a judged CSV after the fact (a
re-judge under a different judge version, an archived run). Both are checked against
each other in `tests/test_bootstrap_ci.py`.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

# The label pools a caller may name instead of a single label. `stance` is the
# union the rubric's own contrast table uses; `hedge` is the behaviour this project
# studies (CLAUDE.md: the behaviour is *hedging*, "soft refusal" is retired as a
# term but remains the v2.1 label slug).
LABEL_POOLS = {
    "stance": ("stance-factual", "stance-evaluative"),
    "hedge": ("soft-refusal",),
    "refusal": ("hard-refusal",),
}
# Judge-side extraction failure. Never a behaviour, never folded into one; an item
# whose verdict is this is EXCLUDED from a rate's denominator and counted separately.
UNMATCHED = "nonsense"


def resolve_positive(name: str) -> tuple[str, ...]:
    """A label pool name, or a comma-separated list of literal labels."""
    if name in LABEL_POOLS:
        return LABEL_POOLS[name]
    return tuple(s.strip() for s in name.split(",") if s.strip())


def load(csv_path, label_col: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv_path}: no rows")
    if label_col not in rows[0]:
        raise SystemExit(
            f"{csv_path}: no column {label_col!r} (has: {', '.join(rows[0])})")
    return rows


def load_many(csv_paths, label_col: str) -> list[dict]:
    """Rows from several judged CSVs, with arms tagged by run when runs differ.

    Two runs both have a `steered_pos`; pooling them would average a treatment with
    its control. So as soon as more than one run is present, every condition becomes
    `<run>:<condition>` — including the ones that do not collide, because a naming
    scheme that changes per row is worse than one that is always explicit.
    """
    rows: list[dict] = []
    for path in csv_paths:
        rows.extend(load(path, label_col))
    runs = {r.get("run") for r in rows}
    if len(csv_paths) > 1 and len(runs) > 1:
        for r in rows:
            r["condition"] = f"{r.get('run')}:{r['condition']}"
    return rows


def bootstrap_ci(hits, *, n_boot: int, ci: float, seed: int) -> tuple[float, float]:
    """Percentile item-bootstrap CI for the mean of `hits`."""
    n = len(hits)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = sorted(sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = boots[int(round((1 - ci) / 2 * (n_boot - 1)))]
    hi = boots[int(round((1 + ci) / 2 * (n_boot - 1)))]
    return lo, hi


def rate_per_arm(rows, positive, *, label_col, n_boot, ci, seed) -> list[dict]:
    by_arm = defaultdict(list)
    failures = defaultdict(int)
    for r in rows:
        label = r[label_col]
        if label == UNMATCHED:
            failures[r["condition"]] += 1
            continue
        by_arm[r["condition"]].append(1.0 if label in positive else 0.0)

    out = []
    for cond in sorted(by_arm):
        hits = by_arm[cond]
        lo, hi = bootstrap_ci(hits, n_boot=n_boot, ci=ci, seed=seed)
        out.append({
            "condition": cond, "n": len(hits), "rate": sum(hits) / len(hits),
            "ci_lo": lo, "ci_hi": hi, "judge_extraction_failures": failures[cond],
        })
    return out


def paired(rows, positive, *, a_cond, b_cond, label_col, n_boot, ci, seed) -> dict:
    """Per-item paired margin between two arms, plus the McNemar 2x2.

    An item counts only if BOTH arms have a usable verdict; an item where either arm
    is a judge extraction failure is dropped from the pairing (and counted), because
    pairing needs two labels and imputing one would invent a comparison.
    """
    by_ex: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        by_ex[r["example_id"]][r["condition"]] = r[label_col]

    a_hits, b_hits, dropped = [], [], 0
    for labels in by_ex.values():
        la, lb = labels.get(a_cond), labels.get(b_cond)
        if la is None or lb is None or la == UNMATCHED or lb == UNMATCHED:
            dropped += 1 if (la is not None and lb is not None) else 0
            continue
        a_hits.append(1.0 if la in positive else 0.0)
        b_hits.append(1.0 if lb in positive else 0.0)

    n = len(a_hits)
    if n == 0:
        raise SystemExit(f"no items have a usable verdict in BOTH {a_cond} and {b_cond}")
    diffs = [x - y for x, y in zip(a_hits, b_hits)]
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, ci=ci, seed=seed)

    both = sum(1 for x, y in zip(a_hits, b_hits) if x and y)
    a_only = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    b_only = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    return {
        "a": a_cond, "b": b_cond, "n": n,
        "a_rate": sum(a_hits) / n, "b_rate": sum(b_hits) / n,
        "margin": sum(diffs) / n, "ci_lo": lo, "ci_hi": hi,
        "both": both, "a_only": a_only, "b_only": b_only,
        "neither": n - both - a_only - b_only,
        "discordant": a_only + b_only, "dropped_unjudgeable_pairs": dropped,
    }


def verdict_line(res: dict, ci: float) -> str:
    if res["ci_lo"] > 0:
        call = f"{res['a']} BEATS {res['b']} (CI excludes 0)"
    elif res["ci_hi"] < 0:
        call = f"{res['b']} BEATS {res['a']} (CI excludes 0)"
    else:
        call = "INCONCLUSIVE — CI spans 0; report the bound, do not call a winner"
    return f"  -> {call}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+", type=Path,
                    help="one or more judged CSVs; with several runs, arms are tagged "
                         "<run>:<condition>")
    ap.add_argument("--label-col", default="verdict_collapsed",
                    help="which judged column to read (default the collapsed-6 view)")
    ap.add_argument("--positive", default="stance",
                    help=f"target behaviour: a pool name {sorted(LABEL_POOLS)} or a "
                         f"comma-separated list of literal labels")
    ap.add_argument("--paired", default=None, metavar="A,B",
                    help="also report the per-item paired margin between two arms, "
                         "e.g. steered_pos,prompt_pos (comma-separated, because an arm "
                         "name may itself contain the run tag's colon)")
    ap.add_argument("--B", dest="n_boot", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    positive = resolve_positive(a.positive)
    rows = load_many(a.csv, a.label_col)
    print(f"{', '.join(str(c) for c in a.csv)}\n  column={a.label_col}  "
          f"positive={{{', '.join(positive)}}}  B={a.n_boot}  {int(a.ci * 100)}% CI\n")

    print("rate per arm:")
    for r in rate_per_arm(rows, positive, label_col=a.label_col, n_boot=a.n_boot,
                          ci=a.ci, seed=a.seed):
        extra = (f"   [{r['judge_extraction_failures']} judge extraction failure(s) "
                 f"excluded]" if r["judge_extraction_failures"] else "")
        print(f"  {r['condition']:12s} n={r['n']:4d}  {r['rate']:.3f}  "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]{extra}")

    if a.paired:
        if "," not in a.paired:
            raise SystemExit("--paired takes two comma-separated arms, e.g. "
                             "steered_pos,prompt_pos")
        a_cond, _, b_cond = a.paired.partition(",")
        present = {r["condition"] for r in rows}
        missing = [c for c in (a_cond, b_cond) if c not in present]
        if missing:
            raise SystemExit(
                f"--paired names arm(s) not in the data: {', '.join(missing)}\n"
                f"  present: {', '.join(sorted(present))}")
        res = paired(rows, positive, a_cond=a_cond, b_cond=b_cond,
                     label_col=a.label_col, n_boot=a.n_boot, ci=a.ci, seed=a.seed)
        print(f"\npaired per-item, {res['a']} vs {res['b']} (n={res['n']}):")
        print(f"  {res['a']} {res['a_rate']:.3f} vs {res['b']} {res['b_rate']:.3f}  "
              f"margin {res['margin']:+.3f}  [{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]")
        print(f"  2x2: both {res['both']} · {res['a']}-only {res['a_only']} · "
              f"{res['b']}-only {res['b_only']} · neither {res['neither']}  "
              f"(discordant {res['discordant']})")
        if res["dropped_unjudgeable_pairs"]:
            print(f"  dropped {res['dropped_unjudgeable_pairs']} pair(s) with a judge "
                  f"extraction failure")
        print(verdict_line(res, a.ci))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
