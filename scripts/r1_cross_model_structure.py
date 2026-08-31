#!/usr/bin/env python3
"""Pre-registered cross-model structure test (writeup 24 §A, frozen 1d6694b).

    python3 -m scripts.r1_cross_model_structure --models qwen-1.8b yi-6b ...

For every model pair with R1 reports: Spearman between the 45 (or shared-subset)
upper-triangle entries of the two cross-category cosine matrices, with a
10,000-draw joint row/column category-label permutation null, seed 0,
one-sided. The 45 entries are never treated as independent observations: the
permutation is of CATEGORY LABELS, so the null respects the matrix dependence
structure. Pearson and mean |off-diagonal| are reported as secondaries.

Writes runs/_r1_audit/cross_model_structure.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

N_PERM = 10_000
SEED = 0


def load_matrix(model):
    rep = json.load(open(f"runs/r1_annotation_{model}/report_annotation_contrast.json"))
    cc = rep["cross_category"]
    return cc["names"], np.array(cc["matrix"], dtype=np.float64)


def upper(m, idx=None):
    n = m.shape[0]
    iu = np.triu_indices(n, k=1)
    return m[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    mats = {}
    for m in args.models:
        try:
            mats[m] = load_matrix(m)
        except FileNotFoundError:
            print(f"{m}: no R1 report yet, skipped")

    out = {"prereg": "results/writeups/24 §A @ 1d6694b", "n_perm": N_PERM,
           "pairs": {}}
    rng = np.random.default_rng(SEED)
    for m1, m2 in itertools.combinations(sorted(mats), 2):
        n1, M1 = mats[m1]
        n2, M2 = mats[m2]
        shared = sorted(set(n1) & set(n2))
        i1 = [n1.index(c) for c in shared]
        i2 = [n2.index(c) for c in shared]
        A = M1[np.ix_(i1, i1)]
        B = M2[np.ix_(i2, i2)]
        a, b = upper(A), upper(B)
        rho = float(spearmanr(a, b).statistic)
        r = float(pearsonr(a, b).statistic)
        null = []
        k = len(shared)
        for _ in range(N_PERM):
            p = rng.permutation(k)
            null.append(spearmanr(upper(A[np.ix_(p, p)]), b).statistic)
        null = np.array(null, dtype=np.float64)
        pval = float((np.sum(null >= rho) + 1) / (N_PERM + 1))
        out["pairs"][f"{m1}|{m2}"] = {
            "shared_categories": len(shared),
            "spearman": rho, "pearson": r, "p_perm_onesided": pval,
            "null_median": float(np.median(null)),
            "null_q95": float(np.quantile(null, 0.95)),
            "mean_abs_offdiag": [float(np.mean(np.abs(a))),
                                 float(np.mean(np.abs(b)))],
        }
        print(f"{m1} vs {m2} (k={len(shared)}): spearman {rho:+.3f} "
              f"p={pval:.4f} (null med {np.median(null):+.3f})")

    path = Path("runs/_r1_audit/cross_model_structure.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
