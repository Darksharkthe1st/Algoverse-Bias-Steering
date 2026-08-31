#!/usr/bin/env python3
"""Pre-registered shared-axis behavioural test (writeup 24 §B) + residual test.

    # GPU: score margins for exactly the R1 ambiguous items
    python3 -m scripts.r1_axis_alignment score --model qwen-1.8b --device mps

    # CPU: the alignment statistics, from committed artifacts only
    python3 -m scripts.r1_axis_alignment test --model qwen-1.8b

Statistics are FROZEN by results/writeups/24 §B (committed 1d6694b, before any
second-model matrix and before this script ran):

  A. shared axis = per-layer mean of the ten unit-normalised category
     directions; item projection summarised across layers by the MEDIAN;
     primary statistic = within-category Spearman(projection, abstention
     margin), summarised as the median across categories, bootstrap CI over
     items. Alignment iff median |rho| >= 0.30 with CI excluding zero.

  B. (added here, frozen before running, same conventions): category-specific
     residual direction r_C = d_C with the leave-C-out shared axis projected
     out; within-category Spearman(projection on r_C, stereotype margin);
     median across categories, bootstrap CI. Same 0.30 reading.

Both numbers are reported whatever they are. The scorer records provenance
(model, ids, system prompt) and the test consumes only committed artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

RHO_BAR = 0.30
N_BOOT = 2000


def unitize(d):
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(n > 0, d / np.where(n > 0, n, 1.0), 0.0)


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def cmd_score(args):
    from src.bias_steer import bbq_score as bs
    from src.bias_steer import models as M
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    run_dir = f"runs/r1_annotation_{args.model}"
    loaded = M.load_model(MODELS[args.model], device=args.device)

    out = {"model": args.model, "system_prompt": DEFAULT_SYS,
           "definition": {"margin": "logP(biased) - logP(nonstereo)",
                          "abstention": "logP(unknown) - max(named)"},
           "items": {}}
    cats = sorted({f.split("__")[0]
                   for f in os.listdir(os.path.join(run_dir, "residuals"))
                   if f.endswith("__a.npy")})
    for cat in cats:
        side = json.load(open(os.path.join(run_dir, "residuals", f"{cat}__a.json")))
        want = side["item_ids"]                     # "Category:example_id"
        pool = bs.load_scoreable(cat, "ambig", 100_000, 0)
        # Example.id is "bbq-<category>-<example_id>"; sidecars key on
        # "<category>:<example_id>" (pairing.item_key). Map via the tail.
        by_id = {f"{cat}:{e.id.rsplit('-', 1)[-1]}": (e, r) for e, r in pool}
        missing = [w for w in want if w not in by_id]
        print(f"{cat}: {len(want)} wanted, {len(missing)} unmatched", flush=True)
        for w in want:
            if w not in by_id:
                continue
            e, r = by_id[w]
            a = e.metadata["answers"]
            s = bs.score_answers(loaded, bs.bare_prompt(e),
                                 [a[r.biased], a[r.nonstereo], a[r.unknown]],
                                 DEFAULT_SYS)
            out["items"][w] = {"category": cat,
                               "margin": float(s[0] - s[1]),
                               "abstention": float(s[2] - max(s[0], s[1]))}
    path = Path(f"runs/_r1_audit/{args.model}_margins.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"written {path} ({len(out['items'])} items)")
    return 0


def cmd_test(args):
    run_dir = f"runs/r1_annotation_{args.model}"
    marg = json.load(open(f"runs/_r1_audit/{args.model}_margins.json"))["items"]

    cats = sorted({f.split("__")[0]
                   for f in os.listdir(os.path.join(run_dir, "residuals"))
                   if f.endswith("__a.npy")})
    resid, ids, dirs = {}, {}, {}
    for c in cats:
        a = np.load(os.path.join(run_dir, "residuals", f"{c}__a.npy")).astype(np.float64)
        b = np.load(os.path.join(run_dir, "residuals", f"{c}__b.npy")).astype(np.float64)
        side = json.load(open(os.path.join(run_dir, "residuals", f"{c}__a.json")))
        resid[c], ids[c] = a, side["item_ids"]
        dirs[c] = a.mean(axis=0) - b.mean(axis=0)

    shared_all = np.mean(np.stack([unitize(dirs[c]) for c in cats], axis=0), axis=0)
    u_shared = unitize(shared_all)

    def proj_median(items_resid, axis_unit):
        p = np.einsum("nld,ld->nl", items_resid, axis_unit)
        return np.median(p, axis=1)

    report = {"model": args.model, "prereg": "results/writeups/24 §B @ 1d6694b",
              "rho_bar": RHO_BAR, "per_category": {}, }
    rhos_abst, rhos_resid, per_cat_pairs = {}, {}, {}
    for c in cats:
        keep = [i for i, iid in enumerate(ids[c]) if iid in marg]
        if len(keep) < 10:
            continue
        m = np.array([marg[ids[c][i]]["margin"] for i in keep])
        ab = np.array([marg[ids[c][i]]["abstention"] for i in keep])
        r_items = resid[c][keep]

        p_shared = proj_median(r_items, u_shared)
        shared_loco = np.mean(np.stack(
            [unitize(dirs[o]) for o in cats if o != c], axis=0), axis=0)
        u_loco = unitize(shared_loco)
        coef = np.einsum("ld,ld->l", dirs[c], u_loco)
        r_C = dirs[c] - coef[:, None] * u_loco
        p_resid = proj_median(r_items, unitize(r_C))

        rho_a = spearman(p_shared, ab)
        rho_r = spearman(p_resid, m)
        rhos_abst[c], rhos_resid[c] = rho_a, rho_r
        per_cat_pairs[c] = (p_shared, ab, p_resid, m)
        report["per_category"][c] = {
            "n": len(keep),
            "rho_sharedproj_abstention": rho_a,
            "rho_residproj_stereomargin": rho_r,
        }

    def med_ci(store, which):
        obs = float(np.median([abs(v) for v in store.values()]))
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(N_BOOT):
            meds = []
            for c, (ps, ab, pr, m) in per_cat_pairs.items():
                n = len(ab)
                idx = rng.integers(0, n, size=n)
                if which == "abst":
                    meds.append(abs(spearman(ps[idx], ab[idx])))
                else:
                    meds.append(abs(spearman(pr[idx], m[idx])))
            boots.append(np.median(meds))
        return obs, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    for label, store, which in (("shared_axis_vs_abstention", rhos_abst, "abst"),
                                ("residual_vs_stereotype_margin", rhos_resid, "resid")):
        obs, lo, hi = med_ci(store, which)
        report[label] = {
            "median_abs_rho": obs, "ci_lo": lo, "ci_hi": hi,
            "aligned_per_prereg": bool(obs >= RHO_BAR and lo > 0.0),
        }
        print(f"{label}: median |rho| = {obs:.3f} [{lo:.3f}, {hi:.3f}] "
              f"-> {'ALIGNED' if obs >= RHO_BAR and lo > 0 else 'not aligned'}")

    path = Path(f"runs/_r1_audit/{args.model}_axis_alignment.json")
    path.write_text(json.dumps(report, indent=1))
    print(f"written {path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--model", default="qwen-1.8b")
    s.add_argument("--device", default="mps"); s.set_defaults(fn=cmd_score)
    t = sub.add_parser("test"); t.add_argument("--model", default="qwen-1.8b")
    t.set_defaults(fn=cmd_test)
    c = sub.add_parser("calibrate"); c.add_argument("--model", default="qwen-1.8b")
    c.set_defaults(fn=cmd_calibrate)
    args = ap.parse_args()
    return args.fn(args)




def cmd_calibrate(args):
    """Permutation calibration of the two median-|rho| statistics.

    Added after the prereg tests ran, with the procedure fixed before running
    it: |rho| medians are positively biased under the null (E|rho_hat| ~ 0.8/
    sqrt(n) at rho=0), so 'not aligned' is only a licensed reading against a
    calibrated zero-reference. 2000 within-category permutations of the
    behavioural variable, seed 0, one-sided p for observed >= null. Applied
    to BOTH tests symmetrically, whatever it shows.
    """
    import numpy as np
    run_dir = f"runs/r1_annotation_{args.model}"
    marg = json.load(open(f"runs/_r1_audit/{args.model}_margins.json"))["items"]
    rep = json.load(open(f"runs/_r1_audit/{args.model}_axis_alignment.json"))

    cats = sorted({f.split("__")[0]
                   for f in os.listdir(os.path.join(run_dir, "residuals"))
                   if f.endswith("__a.npy")})
    resid, ids, dirs = {}, {}, {}
    for c in cats:
        a = np.load(os.path.join(run_dir, "residuals", f"{c}__a.npy")).astype(np.float64)
        b = np.load(os.path.join(run_dir, "residuals", f"{c}__b.npy")).astype(np.float64)
        side = json.load(open(os.path.join(run_dir, "residuals", f"{c}__a.json")))
        resid[c], ids[c] = a, side["item_ids"]
        dirs[c] = a.mean(axis=0) - b.mean(axis=0)
    shared_all = np.mean(np.stack([unitize(dirs[c]) for c in cats], axis=0), axis=0)
    u_shared = unitize(shared_all)

    per = {}
    for c in cats:
        keep = [i for i, iid in enumerate(ids[c]) if iid in marg]
        m = np.array([marg[ids[c][i]]["margin"] for i in keep])
        ab = np.array([marg[ids[c][i]]["abstention"] for i in keep])
        r_items = resid[c][keep]
        ps = np.median(np.einsum("nld,ld->nl", r_items, u_shared), axis=1)
        loco = np.mean(np.stack([unitize(dirs[o]) for o in cats if o != c],
                                axis=0), axis=0)
        u_l = unitize(loco)
        coef = np.einsum("ld,ld->l", dirs[c], u_l)
        r_C = dirs[c] - coef[:, None] * u_l
        pr = np.median(np.einsum("nld,ld->nl", r_items, unitize(r_C)), axis=1)
        per[c] = (ps, ab, pr, m)

    rng = np.random.default_rng(0)
    N_PERM = 2000
    out = {}
    for label, which, obs in (
            ("shared_axis_vs_abstention", "abst",
             rep["shared_axis_vs_abstention"]["median_abs_rho"]),
            ("residual_vs_stereotype_margin", "resid",
             rep["residual_vs_stereotype_margin"]["median_abs_rho"])):
        null = []
        for _ in range(N_PERM):
            meds = []
            for c, (ps, ab, pr, m) in per.items():
                y = ab if which == "abst" else m
                yp = y[rng.permutation(len(y))]
                x = ps if which == "abst" else pr
                meds.append(abs(spearman(x, yp)))
            null.append(float(np.median(meds)))
        null = np.array(null)
        p = float((1 + np.sum(null >= obs)) / (N_PERM + 1))
        out[label] = {"observed": obs, "null_median": float(np.median(null)),
                      "null_q95": float(np.quantile(null, 0.95)),
                      "p_perm_onesided": p, "n_perm": N_PERM}
        print(f"{label}: obs {obs:.3f} vs null med {np.median(null):.3f} "
              f"q95 {np.quantile(null, 0.95):.3f} -> p = {p:.4f}")

    rep["null_calibration"] = out
    Path(f"runs/_r1_audit/{args.model}_axis_alignment.json").write_text(
        json.dumps(rep, indent=1))
    print("artifact updated")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
