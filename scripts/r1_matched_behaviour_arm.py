#!/usr/bin/env python3
"""P0: the behaviour-derived contrast on EXACTLY the R1 universe (17 §5.4).

    python3 -m scripts.r1_matched_behaviour_arm --model qwen-1.8b

Everything below was fixed before this script first ran (this docstring is the
freeze). CPU-only; consumes only committed/checkpointed artifacts:

  items       : the R1 ambiguous arm, verbatim — runs/r1_annotation_<m>/
                residuals/<cat>__a.npy + sidecar ids (200 items/category)
  margins     : runs/_r1_audit/<m>_margins.json — the same items, scored on
                this machine with the run-1 scoring code and system prompt
  splits      : THE SAME 400 split assignments as the annotation floor
                (random.Random(0+k) index shuffle, k = 0..399), so each
                half's ambiguous items are identical between arms
  aggregation : norm-weighted per-layer mean cosine (weights from the first
                half's direction), as in the annotation arm
  statistic   : mean over 400 splits + percentile-bootstrap 95% CI (2000
                draws, seed 0), as in the annotation arm

  ESTIMATORS (both reported, primary declared first):
    quintile     : within each half, rank items by margin; direction =
                   mean(top 20%) − mean(bottom 20%). This is run 1's rule
                   (k = max(1, int(0.20 · n_half)) → 20 v 20 at n_half=100).
    median_split : top 50% vs bottom 50% — the maximally-powered behaviour
                   split, reported to show the result is not about throwing
                   away 60% of the items.

  CONTROLS (quintile estimator, both reported):
    fixed        : margins permuted once per category (seed 7), shared by
                   both halves and all splits — design-symmetric with the
                   annotation arm's conservative fixed-shuffle control.
    independent  : margins permuted freshly per half per split
                   (seeds 60000+2k / 60000+2k+1) — the true zero reference
                   established by the audit (writeup 25 §2).

Output: runs/_r1_audit/<m>_matched_behaviour_arm.json with the paired
per-category table (annotation floor from the committed R1 report vs the
behaviour floors above) and the residual-differences enumeration.

WHAT REMAINS DIFFERENT AFTER THIS MATCHING (enumerated, not claimed away):
  1. The behaviour contrast uses only the ambiguous arm (200 rows/category);
     the annotation contrast uses both arms (400 rows). This is inherent to
     the contrast definitions, not a nuisance we failed to control.
  2. Effective items per half-direction: annotation 100 pairs (200 rows);
     behaviour quintile 40 rows, median_split 100 rows. The median_split
     variant closes most of this gap by construction.
  3. Margins were scored on this machine (MPS fp16, parity-gated) in 2026-08;
     run 1's original margins for this model were scored on CUDA in a
     different session and were never cached, so instrument identity across
     hardware is asserted by the parity gate, not by file identity.
  4. The margin itself is a model behaviour measured once per item; its
     test-retest reliability is unknown, attenuating any behaviour-derived
     contrast (this is part of what the comparison measures, not a bug).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.pilot.analysis import (  # noqa: E402
    bootstrap_ci, norm_weighted_mean_cosine)
from src.bias_steer.bias_taxonomy import per_layer_cosine  # noqa: E402

N_SPLITS = 400
QUINTILE = 0.20


def nw_cos(dA, dB):
    return norm_weighted_mean_cosine(per_layer_cosine(dA, dB), dA)


def summ(vals):
    v = [x for x in vals if np.isfinite(x)]
    lo, hi = bootstrap_ci(v, n_boot=2000, seed=0)
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
            "ci_lo": lo, "ci_hi": hi, "n_splits": len(v)}


def split_idx(n, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    mid = n // 2
    return idx[:mid], idx[mid:]


def extreme_dir(resid, marg, idx, frac):
    order = sorted(idx, key=lambda i: marg[i])
    k = max(1, int(len(order) * frac))
    top, bot = order[-k:], order[:k]
    return resid[top].mean(axis=0) - resid[bot].mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b")
    args = ap.parse_args()

    run_dir = f"runs/r1_annotation_{args.model}"
    marg_all = json.load(open(f"runs/_r1_audit/{args.model}_margins.json"))["items"]
    r1 = json.load(open(f"{run_dir}/report_annotation_contrast.json"))

    cats = sorted({f.split("__")[0]
                   for f in os.listdir(os.path.join(run_dir, "residuals"))
                   if f.endswith("__a.npy")})
    out = {"model": args.model, "n_splits": N_SPLITS, "quintile": QUINTILE,
           "split_assignments": "identical to the annotation floor "
                                "(random.Random(0+k), k=0..399)",
           "per_category": {}}

    print(f"{'category':<22}{'annot':>8}{'quint':>8}{'medspl':>8}"
          f"{'ctrl-fix':>9}{'ctrl-ind':>9}")
    for c in cats:
        resid = np.load(os.path.join(run_dir, "residuals", f"{c}__a.npy")
                        ).astype(np.float64)
        ids = json.load(open(os.path.join(run_dir, "residuals", f"{c}__a.json"))
                        )["item_ids"]
        if any(i not in marg_all for i in ids):
            missing = sum(i not in marg_all for i in ids)
            print(f"{c}: {missing} items missing margins — abort this category")
            continue
        marg = np.array([marg_all[i]["margin"] for i in ids])
        n = resid.shape[0]

        rng_fix = np.random.default_rng(7)
        marg_fixed = marg[rng_fix.permutation(n)]

        quint, medspl, cfix, cind = [], [], [], []
        for k in range(N_SPLITS):
            A, B = split_idx(n, 0 + k)
            dA = extreme_dir(resid, marg, A, QUINTILE)
            dB = extreme_dir(resid, marg, B, QUINTILE)
            quint.append(nw_cos(dA, dB))

            dA2 = extreme_dir(resid, marg, A, 0.50)
            dB2 = extreme_dir(resid, marg, B, 0.50)
            medspl.append(nw_cos(dA2, dB2))

            dAf = extreme_dir(resid, marg_fixed, A, QUINTILE)
            dBf = extreme_dir(resid, marg_fixed, B, QUINTILE)
            cfix.append(nw_cos(dAf, dBf))

            rA = np.random.default_rng(60_000 + 2 * k)
            rB = np.random.default_rng(60_000 + 2 * k + 1)
            mA, mB = marg.copy(), marg.copy()
            permA, permB = rA.permutation(n), rB.permutation(n)
            dAi = extreme_dir(resid, mA[permA], A, QUINTILE)
            dBi = extreme_dir(resid, mB[permB], B, QUINTILE)
            cind.append(nw_cos(dAi, dBi))

        annot = r1["observed_floor"][c]["mean"]
        row = {
            "annotation_floor_mean": annot,
            "behaviour_quintile": summ(quint),
            "behaviour_median_split": summ(medspl),
            "behaviour_control_fixed": summ(cfix),
            "behaviour_control_independent": summ(cind),
        }
        out["per_category"][c] = row
        print(f"{c:<22}{annot:>8.3f}{row['behaviour_quintile']['mean']:>8.3f}"
              f"{row['behaviour_median_split']['mean']:>8.3f}"
              f"{row['behaviour_control_fixed']['mean']:>9.3f}"
              f"{row['behaviour_control_independent']['mean']:>9.3f}")

    path = Path(f"runs/_r1_audit/{args.model}_matched_behaviour_arm.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwritten {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
