#!/usr/bin/env python3
"""Cohen's κ (and per-label) from two filled annotation CSVs.

Expects columns: item_id, label
Labels should match docs/RUBRIC_v2.md (after freeze).

  python3 scripts/kappa_from_csv.py \\
      --a path/annotator_A.csv --b path/annotator_B.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def load_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = (row.get("item_id") or "").strip()
            lab = (row.get("label") or "").strip().lower()
            if not iid:
                continue
            if not lab:
                continue
            out[iid] = lab
    return out


def cohens_kappa(y1: list[str], y2: list[str]) -> float:
    assert len(y1) == len(y2) and y1
    labels = sorted(set(y1) | set(y2))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(y1)
    mat = [[0] * len(labels) for _ in labels]
    for a, b in zip(y1, y2):
        mat[idx[a]][idx[b]] += 1
    # observed agreement
    po = sum(mat[i][i] for i in range(len(labels))) / n
    # expected
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    args = ap.parse_args()

    la, lb = load_labels(args.a), load_labels(args.b)
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
        print(f"  {lab:20s}  κ={k:.4f}  items_touching={n_lab}")
    # confusion-ish
    print("agreement_rate=", sum(a == b for a, b in zip(y1, y2)) / len(common))
    disagree = Counter((a, b) for a, b in zip(y1, y2) if a != b)
    if disagree:
        print("top disagreements (a→b):")
        for (a, b), n in disagree.most_common(12):
            print(f"  {n:3d}  {a} → {b}")
    gate = 0.70
    print(f"gate per-category κ≥{gate}:", "PASS candidates only if each label meets it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
