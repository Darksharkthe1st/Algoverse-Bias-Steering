#!/usr/bin/env python3
"""Audit battery for an R1 annotation-contrast run. CPU-only, from residuals.

    python3 -m scripts.r1_audit --model qwen-1.8b

Writes runs/_r1_audit/<model>.json with five sections. Every statistic below
was fixed BEFORE any of them was computed (this docstring is the freeze):

PROVENANCE — assembled from committed artifacts plus code facts: item universe
per arm, pairing integrity, capture site, and the explicit inventory of what
differs between run 1 and R1 beyond the contrast labels. The paper's current
sentence "only the labelling of the contrast changed" is graded against it.

LEAKAGE —
  (a) split disjointness: recompute all 400 split seeds, assert no scenario
      pair is in both halves (structural, asserted anyway);
  (b) template overlap: mean number of question_index templates shared by the
      two halves under the standard pair-level split (they are expected to
      share nearly all templates; this quantifies it);
  (c) template-disjoint floor: split at TEMPLATE level (all pairs of a
      question_index in one half), 200 splits, same estimator and layer
      summary. If the floor survives, scenario-template leakage does not
      explain it; if it collapses, the floor is template-bound.

NEGATIVE-CONTROL MECHANISM — why shuffled-label controls sit far above the
1/sqrt(d) chance line. Three floors, 200 splits each:
  fixed_shuffle      : the run's own control design (one shuffle, then splits)
  per_half_reshuffle : each half gets an INDEPENDENT fresh shuffle, breaking
                       any correlation carried by a shared labelling
  label_free         : no labels at all — each half's "direction" is the mean
                       difference of two random disjoint item groups; measures
                       pure residual-population anisotropy
Reading (fixed): control level explained by anisotropy iff label_free is
within the fixed_shuffle CI; any excess of fixed_shuffle over per_half_reshuffle
measures shared-labelling correlation.

LOCO TRANSFER — for each held-out category C: shared direction = per-layer
mean of the unit-normalised (per layer) full-data directions of the other 9
categories. Per layer and per the norm-weighted summary (weights = shared
direction's per-layer norms): AUC(ambig vs disambig projections on C), and the
within-pair statistic P(proj(a) > proj(b)) with its binomial 95% CI. AUC and
paired sign rate are threshold-free and were chosen before computation.

DECOMPOSITION — per category C, 200 splits, split unit = scenario pair:
  original_floor : split-half cosine of d_C (norm-weighted summary), mean+CI
  shared_selffloor: same for shared_-C (the 9-category mean direction)
  residual_floor : same for d_C with shared_-C,h projected out per half h
                   (the shared estimate never uses C or the other half)
  cos_shared     : norm-weighted |cos(d_C, shared_-C)| on full data
No thresholds; numbers only. Interpretation lives in the memo, not here.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.pilot import pairing  # noqa: E402
from scripts.pilot.analysis import (  # noqa: E402
    bootstrap_ci, norm_weighted_mean_cosine)
from src.bias_steer.bias_taxonomy import per_layer_cosine  # noqa: E402

N_SPLITS = 200
SEED = 0


def load_run(run_dir: str, cats: list[str]):
    """(a_resid, b_resid, ids_a, ids_b) per category, alignment asserted."""
    out = {}
    for c in cats:
        a = np.load(os.path.join(run_dir, "residuals", f"{c}__a.npy"))
        b = np.load(os.path.join(run_dir, "residuals", f"{c}__b.npy"))
        sa = json.load(open(os.path.join(run_dir, "residuals", f"{c}__a.json")))
        sb = json.load(open(os.path.join(run_dir, "residuals", f"{c}__b.json")))
        out[c] = (a.astype(np.float64), b.astype(np.float64),
                  sa["item_ids"], sb["item_ids"])
    return out


def direction(a, b, idx):
    return a[idx].mean(axis=0) - b[idx].mean(axis=0)


def nw_cos(dA, dB):
    return norm_weighted_mean_cosine(per_layer_cosine(dA, dB), dA)


def summ(vals, n_boot=2000):
    v = [x for x in vals if np.isfinite(x)]
    lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=SEED)
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
            "ci_lo": lo, "ci_hi": hi, "n_splits": len(v)}


def split_idx(n, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    mid = n // 2
    return idx[:mid], idx[mid:]


def unitize(d):
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(n > 0, d / np.where(n > 0, n, 1.0), 0.0)


# --------------------------------------------------------------------------- #
def provenance(run_dir, data, pairs_by_cat):
    per_cat = {}
    for c, (a, b, ida, idb) in data.items():
        pairs = pairs_by_cat[c]
        want_a = [pairing.item_key(p.a) for p in pairs]
        want_b = [pairing.item_key(p.b) for p in pairs]
        per_cat[c] = {
            "n_pairs": len(pairs),
            "arm_a_rows": int(a.shape[0]), "arm_b_rows": int(b.shape[0]),
            "sidecar_ids_match_fresh_pairing": bool(ida == want_a and idb == want_b),
            "templates_in_sample": len({p.key[0] for p in pairs}),
            "arm_a_is_ambig": all(p.a["context_condition"] == "ambig" for p in pairs),
            "arm_b_is_disambig": all(p.b["context_condition"] == "disambig" for p in pairs),
        }
    cap = json.load(open(os.path.join(run_dir, "capture_site.json")))
    return {
        "per_category": per_cat,
        "capture_index": cap["capture_index"],
        "capture_hook": "resid_pre",
        "estimator": "difference of arm means, no hyperparameter",
        "floor_procedure": "400 scenario-pair splits, norm-weighted per-layer "
                           "mean cosine, mean over splits with bootstrap CI",
        "run1_vs_r1_differences_beyond_labels": [
            "item universe: run 1 used 600 seeded ambiguous items (extremes "
            "kept 240); R1 uses 200 evenly-spaced scenario pairs = 200 ambig "
            "+ 200 disambig rows; the ambiguous sets overlap but are not "
            "identical, and run 1 saved no item ids for this model, so the "
            "overlap is not verifiable for qwen-1.8b",
            "floor statistic: run 1 = q05 of 10 item-level splits, unweighted "
            "median across layers; R1 = mean of 400 pair-level splits, "
            "norm-weighted mean across layers, bootstrap CI",
            "negative control: run 1 had none at category level; R1 shuffles "
            "arm labels within scenario",
            "identical: model, dataset files, bare-prompt scoring surface, "
            "chat template and system prompt, capture hook and index (-1), "
            "difference-of-means estimator family",
        ],
    }


# --------------------------------------------------------------------------- #
def leakage(data, pairs_by_cat):
    out = {}
    for c, (a, b, *_ ) in data.items():
        pairs = pairs_by_cat[c]
        n = len(pairs)
        tmpl = [p.key[0] for p in pairs]
        by_tmpl = collections.defaultdict(list)
        for i, t in enumerate(tmpl):
            by_tmpl[t].append(i)
        tkeys = sorted(by_tmpl)

        disjoint_ok, shared_templates = True, []
        std, tdisj = [], []
        fixed_shuffle, per_half_reshuffle, label_free = [], [], []

        # the run's own fixed shuffle (analysis.py: seed+10_000), swap flags
        rngf = random.Random(SEED + 10_000)
        fixed_swap = [rngf.random() < 0.5 for _ in range(n)]

        pooled = np.concatenate([a, b], axis=0)          # (2n, L, d)

        for k in range(N_SPLITS):
            A, B = split_idx(n, SEED + k)
            if set(A) & set(B):
                disjoint_ok = False
            shared_templates.append(
                len({tmpl[i] for i in A} & {tmpl[i] for i in B}))

            dA, dB = direction(a, b, A), direction(a, b, B)
            std.append(nw_cos(dA, dB))

            # template-disjoint split
            tA, tB = split_idx(len(tkeys), SEED + 1000 + k)
            iA = [i for t in tA for i in by_tmpl[tkeys[t]]]
            iB = [i for t in tB for i in by_tmpl[tkeys[t]]]
            if iA and iB:
                tdisj.append(nw_cos(direction(a, b, iA), direction(a, b, iB)))

            # fixed shuffle: same swap pattern both halves (the run's design)
            sgnA = np.array([-1.0 if fixed_swap[i] else 1.0 for i in A])
            sgnB = np.array([-1.0 if fixed_swap[i] else 1.0 for i in B])
            dAs = ((a[A] - b[A]) * sgnA[:, None, None]).mean(axis=0)
            dBs = ((a[B] - b[B]) * sgnB[:, None, None]).mean(axis=0)
            fixed_shuffle.append(nw_cos(dAs, dBs))

            # per-half independent reshuffle
            rA = random.Random(30_000 + 2 * k)
            rB = random.Random(30_000 + 2 * k + 1)
            sgnA2 = np.array([-1.0 if rA.random() < 0.5 else 1.0 for _ in A])
            sgnB2 = np.array([-1.0 if rB.random() < 0.5 else 1.0 for _ in B])
            dAr = ((a[A] - b[A]) * sgnA2[:, None, None]).mean(axis=0)
            dBr = ((a[B] - b[B]) * sgnB2[:, None, None]).mean(axis=0)
            per_half_reshuffle.append(nw_cos(dAr, dBr))

            # label-free anisotropy: random item bipartition inside each half
            items_A = [i for i in A] + [n + i for i in A]
            items_B = [i for i in B] + [n + i for i in B]
            rf = random.Random(50_000 + k)
            rf.shuffle(items_A); rf.shuffle(items_B)
            hA = len(items_A) // 2
            hB = len(items_B) // 2
            dAf = pooled[items_A[:hA]].mean(axis=0) - pooled[items_A[hA:]].mean(axis=0)
            dBf = pooled[items_B[:hB]].mean(axis=0) - pooled[items_B[hB:]].mean(axis=0)
            label_free.append(nw_cos(dAf, dBf))

        out[c] = {
            "split_disjointness_ok": disjoint_ok,
            "templates_total": len(tkeys),
            "templates_shared_between_halves_mean":
                float(np.mean(shared_templates)),
            "floor_standard_split": summ(std),
            "floor_template_disjoint_split": summ(tdisj),
            "control_fixed_shuffle": summ(fixed_shuffle),
            "control_per_half_reshuffle": summ(per_half_reshuffle),
            "control_label_free_anisotropy": summ(label_free),
        }
    return out


# --------------------------------------------------------------------------- #
def loco(data):
    dirs = {c: direction(a, b, list(range(a.shape[0])))
            for c, (a, b, *_ ) in data.items()}
    out = {}
    for c, (a, b, *_ ) in data.items():
        others = [unitize(dirs[o]) for o in dirs if o != c]
        shared = np.mean(np.stack(others, axis=0), axis=0)     # (L, d)
        w = np.linalg.norm(shared, axis=1)

        pa = np.einsum("nld,ld->nl", a, shared)                # (n, L)
        pb = np.einsum("nld,ld->nl", b, shared)

        n_items = a.shape[0]
        per_layer_auc, per_layer_pairwin = [], []
        for layer in range(a.shape[1]):
            x, y = pa[:, layer], pb[:, layer]
            ranks = np.argsort(np.argsort(np.concatenate([x, y])))
            auc = (ranks[:n_items].sum() - n_items * (n_items - 1) / 2) / (n_items * n_items)
            per_layer_auc.append(float(auc))
            per_layer_pairwin.append(float((x > y).mean()))

        wl = w / w.sum()
        pw = float(np.dot(per_layer_pairwin, wl))
        se = float(np.sqrt(pw * (1 - pw) / n_items))
        out[c] = {
            "n_pairs": n_items,
            "auc_per_layer": per_layer_auc,
            "pairwin_per_layer": per_layer_pairwin,
            "auc_norm_weighted": float(np.dot(per_layer_auc, wl)),
            "pairwin_norm_weighted": pw,
            "pairwin_ci95": [max(0.0, pw - 1.96 * se), min(1.0, pw + 1.96 * se)],
        }
    return out


# --------------------------------------------------------------------------- #
def project_out_dir(d, axis):
    u = unitize(axis)
    coef = np.einsum("ld,ld->l", d, u)
    return d - coef[:, None] * u


def decompose(data):
    cats = sorted(data)
    out = {}
    full_dirs = {c: direction(*data[c][:2], list(range(data[c][0].shape[0])))
                 for c in cats}
    for c in cats:
        a, b, *_ = data[c]
        n = a.shape[0]
        shared_full = np.mean(np.stack(
            [unitize(full_dirs[o]) for o in cats if o != c], axis=0), axis=0)
        cos_shared = abs(nw_cos(full_dirs[c], shared_full))

        orig, resid, shared_sf = [], [], []
        for k in range(N_SPLITS):
            A, B = split_idx(n, SEED + k)
            dA, dB = direction(a, b, A), direction(a, b, B)
            orig.append(nw_cos(dA, dB))

            shA = np.mean(np.stack(
                [unitize(direction(*data[o][:2], A)) for o in cats if o != c],
                axis=0), axis=0)
            shB = np.mean(np.stack(
                [unitize(direction(*data[o][:2], B)) for o in cats if o != c],
                axis=0), axis=0)
            shared_sf.append(nw_cos(shA, shB))
            rA = project_out_dir(dA, shA)
            rB = project_out_dir(dB, shB)
            resid.append(nw_cos(rA, rB))

        out[c] = {
            "original_floor": summ(orig),
            "residual_floor": summ(resid),
            "shared_axis_selffloor": summ(shared_sf),
            "cos_with_shared_loco": cos_shared,
        }
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-per-arm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir or f"runs/r1_annotation_{args.model}"
    cats = sorted({f.split("__")[0]
                   for f in os.listdir(os.path.join(run_dir, "residuals"))
                   if f.endswith(".npy")})
    print(f"{args.model}: {len(cats)} categories from {run_dir}")

    data = load_run(run_dir, cats)
    cd = pairing.load_pilot_categories(cats, limit_pairs=args.n_per_arm)
    pairs_by_cat = {x.category: x.pairs for x in cd}

    report = {"model": args.model, "run_dir": run_dir,
              "n_splits": N_SPLITS, "seed": SEED}
    print("provenance ...");  report["provenance"] = provenance(run_dir, data, pairs_by_cat)
    print("leakage + controls ...");  report["leakage"] = leakage(data, pairs_by_cat)
    print("loco transfer ...");  report["loco_transfer"] = loco(data)
    print("decomposition ...");  report["decomposition"] = decompose(data)

    # Residual RDM: full-data leave-C-out-shared-removed directions, pairwise
    # norm-weighted cosines. Statistic frozen 2026-08-31 AFTER qwen-1.8b and
    # BEFORE any second model's R1 existed (writeup 24 amendment A1); its
    # cross-model comparison is post-prereg relative to 24 §A and is labelled
    # so wherever it is reported.
    cats_sorted = sorted(data)
    full_dirs = {c: direction(*data[c][:2], list(range(data[c][0].shape[0])))
                 for c in cats_sorted}
    rdirs = {}
    for c in cats_sorted:
        sh = np.mean(np.stack([unitize(full_dirs[o]) for o in cats_sorted
                               if o != c], axis=0), axis=0)
        rdirs[c] = project_out_dir(full_dirs[c], sh)
    M = [[1.0 if i == j else nw_cos(rdirs[a], rdirs[b])
          for j, b in enumerate(cats_sorted)] for i, a in enumerate(cats_sorted)]
    report["residual_rdm"] = {"names": cats_sorted, "matrix": M,
                              "frozen": "2026-08-31 post-qwen-1.8b pre-model-2"}

    out = Path(args.out or f"runs/_r1_audit/{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"written to {out}")

    # console digest
    for c in cats:
        L = report["leakage"][c]; D = report["decomposition"][c]; T = report["loco_transfer"][c]
        print(f"  {c:<22} std {L['floor_standard_split']['mean']:+.3f}"
              f"  tmpl-disj {L['floor_template_disjoint_split']['mean']:+.3f}"
              f"  ctrl(fix/rehalf/free) {L['control_fixed_shuffle']['mean']:+.2f}"
              f"/{L['control_per_half_reshuffle']['mean']:+.2f}"
              f"/{L['control_label_free_anisotropy']['mean']:+.2f}"
              f"  resid-floor {D['residual_floor']['mean']:+.3f}"
              f"  loco-AUC {T['auc_norm_weighted']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
