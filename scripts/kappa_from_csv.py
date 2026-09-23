#!/usr/bin/env python3
"""Inter-annotator agreement from filled annotation CSVs.

Two-rater mode (Cohen's κ) is unchanged from the original script:

  py scripts/kappa_from_csv.py --a path/annotator_A.csv --b path/annotator_B.csv

Multi-rater mode (WP-21) adds Fleiss κ, per-category κ_j, Gwet's AC1 and
percentile bootstrap CIs, for WP-25's three-annotator adjudication:

  py scripts/kappa_from_csv.py --sheets wp25/sheet_a.csv wp25/sheet_b.csv wp25/sheet_c.csv
  py scripts/kappa_from_csv.py --master wp25/battery_stratified_MASTER.csv

Why AC1 is reported next to κ: Fleiss κ is depressed when one category
dominates the marginals (the "kappa paradox"). The WP-25 battery is expected to
run roughly 2:1 S2:S3, so κ can look poor at high raw agreement. Gwet's AC1 is
robust to that skew. Report both; neither is a gate.

Label column is auto-detected: ``label``, ``stratum_S2_or_S3``, or ``stratum``.
Rows with a blank label are dropped, and every statistic is computed only over
items every rater labelled.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the Greek letters and
# em dashes used below. Without this, printing results raises UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

LABEL_COLUMNS = ("label", "stratum_S2_or_S3", "stratum")


def _pick_label_column(fieldnames: list[str] | None, override: str | None) -> str:
    if override:
        return override
    names = fieldnames or []
    for cand in LABEL_COLUMNS:
        if cand in names:
            return cand
    raise SystemExit(
        f"could not find a label column in {names!r}; pass --label-col explicitly"
    )


def load_labels(path: Path, label_col: str | None = None) -> dict[str, str]:
    """item_id -> label, skipping blanks. Labels are upper-cased and stripped."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        col = _pick_label_column(reader.fieldnames, label_col)
        for row in reader:
            iid = (row.get("item_id") or "").strip()
            lab = (row.get(col) or "").strip().upper()
            if not iid or not lab:
                continue
            out[iid] = lab
    return out


def load_master(path: Path) -> dict[str, dict[str, str]]:
    """Read every ``label_*`` column out of the master CSV.

    Returns rater_name -> {item_id: label}.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if c.startswith("label_")]
        if not cols:
            raise SystemExit(f"{path} has no label_* columns")
        raters: dict[str, dict[str, str]] = {c[len("label_") :]: {} for c in cols}
        for row in reader:
            iid = (row.get("item_id") or "").strip()
            if not iid:
                continue
            for c in cols:
                lab = (row.get(c) or "").strip().upper()
                if lab:
                    raters[c[len("label_") :]][iid] = lab
    return raters


# --------------------------------------------------------------------------
# two raters — Cohen
# --------------------------------------------------------------------------


def cohens_kappa(y1: list[str], y2: list[str]) -> float:
    assert len(y1) == len(y2) and y1
    labels = sorted(set(y1) | set(y2))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(y1)
    mat = [[0] * len(labels) for _ in labels]
    for a, b in zip(y1, y2):
        mat[idx[a]][idx[b]] += 1
    po = sum(mat[i][i] for i in range(len(labels))) / n
    row = [sum(mat[i][j] for j in range(len(labels))) for i in range(len(labels))]
    col = [sum(mat[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(len(labels)))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def per_label_kappa(y1: list[str], y2: list[str], label: str) -> float:
    """Binary κ treating ``label`` vs rest."""
    b1 = ["pos" if y == label else "neg" for y in y1]
    b2 = ["pos" if y == label else "neg" for y in y2]
    return cohens_kappa(b1, b2)


# --------------------------------------------------------------------------
# N raters — Fleiss and Gwet
# --------------------------------------------------------------------------


def _counts_matrix(rows: list[list[str]], labels: list[str]) -> list[list[int]]:
    """rows[i] is the list of labels the raters gave item i."""
    idx = {l: j for j, l in enumerate(labels)}
    return [[sum(1 for lab in r if idx[lab] == j) for j in range(len(labels))] for r in rows]


def observed_agreement(rows: list[list[str]], labels: list[str]) -> float:
    """Mean per-item pairwise agreement — the P̄ shared by Fleiss and AC1."""
    mat = _counts_matrix(rows, labels)
    n_raters = len(rows[0])
    if n_raters < 2:
        raise SystemExit("need at least 2 raters")
    total = 0.0
    for counts in mat:
        agreeing = sum(c * (c - 1) for c in counts)
        total += agreeing / (n_raters * (n_raters - 1))
    return total / len(mat)


def marginals(rows: list[list[str]], labels: list[str]) -> list[float]:
    mat = _counts_matrix(rows, labels)
    n_items = len(mat)
    n_raters = len(rows[0])
    return [sum(r[j] for r in mat) / (n_items * n_raters) for j in range(len(labels))]


def fleiss_kappa(rows: list[list[str]], labels: list[str]) -> float:
    p_bar = observed_agreement(rows, labels)
    p = marginals(rows, labels)
    p_e = sum(x * x for x in p)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def gwet_ac1(rows: list[list[str]], labels: list[str]) -> float:
    """Gwet's AC1 — same observed agreement, skew-robust chance term."""
    p_bar = observed_agreement(rows, labels)
    p = marginals(rows, labels)
    k = len(labels)
    if k < 2:
        return 1.0
    p_e = sum(x * (1.0 - x) for x in p) / (k - 1)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def fleiss_kappa_per_label(rows: list[list[str]], label: str) -> float:
    """One-vs-rest Fleiss κ_j for a single category."""
    collapsed = [["POS" if x == label else "NEG" for x in r] for r in rows]
    return fleiss_kappa(collapsed, ["NEG", "POS"])


def bootstrap_ci(
    rows: list[list[str]],
    labels: list[str],
    stat,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI by resampling *items* with replacement."""
    rng = random.Random(seed)
    n = len(rows)
    vals: list[float] = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        present = sorted({lab for r in sample for lab in r})
        if len(present) < 2:
            # degenerate resample — every item landed in one category
            vals.append(1.0)
            continue
        vals.append(stat(sample, present))
    vals.sort()
    lo = vals[max(0, int((alpha / 2) * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def report_two(la: dict[str, str], lb: dict[str, str]) -> dict:
    common = sorted(set(la) & set(lb))
    if not common:
        raise SystemExit("no overlapping item_id with non-empty labels")
    y1 = [la[i] for i in common]
    y2 = [lb[i] for i in common]
    overall = cohens_kappa(y1, y2)
    print(f"n_paired={len(common)}")
    print(f"overall_kappa={overall:.4f}")
    labels = sorted(set(y1) | set(y2))
    print("per_label_kappa (one-vs-rest):")
    for lab in labels:
        k = per_label_kappa(y1, y2, lab)
        n_lab = sum(1 for a, b in zip(y1, y2) if a == lab or b == lab)
        print(f"  {lab:20s}  kappa={k:.4f}  items_touching={n_lab}")
    agree = sum(a == b for a, b in zip(y1, y2)) / len(common)
    print("agreement_rate=", agree)
    disagree = Counter((a, b) for a, b in zip(y1, y2) if a != b)
    if disagree:
        print("top disagreements (a→b):")
        for (a, b), n in disagree.most_common(12):
            print(f"  {n:3d}  {a} → {b}")
    return {"mode": "cohen", "n": len(common), "kappa": overall, "agreement_rate": agree}


def report_many(
    raters: dict[str, dict[str, str]], n_boot: int, seed: int
) -> dict:
    names = sorted(raters)
    if len(names) < 2:
        raise SystemExit(f"need >=2 raters, found {names}")
    common = sorted(set.intersection(*(set(raters[n]) for n in names)))
    if not common:
        raise SystemExit("no item_id labelled by every rater")

    rows = [[raters[n][i] for n in names] for i in common]
    labels = sorted({lab for r in rows for lab in r})

    n_items = len(rows)
    n_raters = len(names)
    unanimous = sum(1 for r in rows if len(set(r)) == 1)
    p_bar = observed_agreement(rows, labels)
    kappa = fleiss_kappa(rows, labels)
    ac1 = gwet_ac1(rows, labels)

    print(f"raters              : {n_raters}  ({', '.join(names)})")
    print(f"items labelled by all: {n_items}")
    print(f"categories          : {', '.join(labels)}")
    print()
    print(f"unanimous_items     : {unanimous}  ({unanimous / n_items:.1%})")
    print(f"disagreement_rate   : {1 - unanimous / n_items:.4f}"
          f"   ({n_items - unanimous} of {n_items} items split)")
    print(f"mean_pairwise_agree : {p_bar:.4f}")
    print()

    out: dict = {
        "mode": "fleiss",
        "raters": names,
        "n_items": n_items,
        "categories": labels,
        "unanimous_items": unanimous,
        "disagreement_rate": 1 - unanimous / n_items,
        "mean_pairwise_agreement": p_bar,
        "fleiss_kappa": kappa,
        "gwet_ac1": ac1,
    }

    if n_boot > 0:
        k_lo, k_hi = bootstrap_ci(rows, labels, fleiss_kappa, n_boot, seed)
        a_lo, a_hi = bootstrap_ci(rows, labels, gwet_ac1, n_boot, seed)
        print(f"fleiss_kappa        : {kappa:.4f}   95% CI [{k_lo:.4f}, {k_hi:.4f}]")
        print(f"gwet_ac1            : {ac1:.4f}   95% CI [{a_lo:.4f}, {a_hi:.4f}]")
        out["fleiss_kappa_ci95"] = [k_lo, k_hi]
        out["gwet_ac1_ci95"] = [a_lo, a_hi]
        out["n_boot"] = n_boot
        out["seed"] = seed
    else:
        print(f"fleiss_kappa        : {kappa:.4f}")
        print(f"gwet_ac1            : {ac1:.4f}")

    print()
    print("per_category fleiss kappa_j (one-vs-rest):")
    per: dict[str, float] = {}
    for lab in labels:
        kj = fleiss_kappa_per_label(rows, lab)
        n_lab = sum(1 for r in rows if lab in r)
        per[lab] = kj
        print(f"  {lab:12s}  kappa_j={kj:.4f}   items_touching={n_lab}")
    out["fleiss_kappa_per_category"] = per

    print()
    print("pairwise cohen kappa:")
    pairwise: dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            y1 = [raters[a][x] for x in common]
            y2 = [raters[b][x] for x in common]
            k = cohens_kappa(y1, y2)
            agree = sum(u == v for u, v in zip(y1, y2)) / len(common)
            pairwise[f"{a}|{b}"] = k
            print(f"  {a:10s} vs {b:10s}  kappa={k:+.4f}   raw_agree={agree:.1%}")
    out["pairwise_cohen_kappa"] = pairwise

    split = [
        (common[i], Counter(rows[i]).most_common())
        for i in range(n_items)
        if len(set(rows[i])) > 1
    ]
    if split:
        print()
        print(f"split items ({len(split)}) — majority wins, blanks resolve to S3:")
        for iid, counts in split[:25]:
            shown = "  ".join(f"{lab}×{n}" for lab, n in counts)
            print(f"  {iid}   {shown}")
        if len(split) > 25:
            print(f"  ... and {len(split) - 25} more")
    out["split_item_ids"] = [iid for iid, _ in split]

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inter-annotator agreement (Cohen / Fleiss / Gwet AC1).",
    )
    ap.add_argument("--a", type=Path, help="two-rater mode: first CSV")
    ap.add_argument("--b", type=Path, help="two-rater mode: second CSV")
    ap.add_argument("--sheets", type=Path, nargs="+", help="N-rater mode: one CSV per rater")
    ap.add_argument("--master", type=Path, help="N-rater mode: master CSV with label_* columns")
    ap.add_argument("--label-col", default=None, help="override label column name")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 disables)")
    ap.add_argument("--seed", type=int, default=20260828, help="bootstrap seed, for reproducibility")
    ap.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = ap.parse_args()

    if args.master:
        result = report_many(load_master(args.master), args.boot, args.seed)
    elif args.sheets:
        if len(args.sheets) < 2:
            raise SystemExit("--sheets needs at least two files")
        raters = {}
        for p in args.sheets:
            name = re.sub(r"^sheet_", "", p.stem)
            raters[name] = load_labels(p, args.label_col)
        result = report_many(raters, args.boot, args.seed)
    elif args.a and args.b:
        result = report_two(
            load_labels(args.a, args.label_col), load_labels(args.b, args.label_col)
        )
    else:
        ap.error("give --master, or --sheets A B C, or --a A --b B")
        return 2

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
