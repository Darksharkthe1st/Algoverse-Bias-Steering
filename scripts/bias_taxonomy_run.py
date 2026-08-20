"""Experiment 1, end to end: do bias topics have different directions?

    python scripts/bias_taxonomy_run.py --model qwen-1.8b

Stages, in the order they gate each other:

  1. POSITIVE CONTROL per category. Disambiguated items, no option list, can the
     model pick the person the context names? Chance is 1/3. A category that
     fails is dropped: its margins are not measuring anything.

  2. MARGINS per category on ambiguous items.
         margin = logP(stereotyped person) - logP(other person)
     No option list — an earlier design scored options the prompt had just
     listed, and list position accounted for the entire margin.

  3. EXTRACT a direction per category from the margin extremes: the mean
     prompt-token residual of the top quintile minus that of the bottom
     quintile, per layer -> (n_layers, d_model). Top leans stereotyped, bottom
     leans anti-stereotyped, so the contrast is stereotype-vs-anti rather than
     commit-vs-abstain.

  4. EXTRACTION FLOOR per category: re-extract from random halves of the SAME
     category and take the cosine. This is how much a direction moves when the
     topic did not change, and nothing downstream is interpretable without it.
     Reported with its n, always.

  5. COSINE MATRIX between categories, read against each pair's own floors, then
     hierarchical clustering and a PERMUTATION NULL. Random vectors make
     convincing dendrograms; the null is what separates structure from decoration.

  6. FLOOR vs N on the largest category, to show how much of the floor is sample
     size rather than direction stability.

Everything is written to a run directory. Interpretation happens after, never
during.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.bias_steer import bbq_score as bs  # noqa: E402
from src.bias_steer import bias_taxonomy as bt  # noqa: E402


def say(msg):
    print(f"\n=== {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--categories", nargs="+", default=bs.CATEGORIES)
    ap.add_argument("--ambig-limit", type=int, default=400)
    ap.add_argument("--control-limit", type=int, default=150)
    ap.add_argument("--quintile", type=float, default=0.20)
    ap.add_argument("--floor-splits", type=int, default=10)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"runs/{stamp}_bias-taxonomy_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out_dir}")

    spec = MODELS[args.model]
    print(f"model  : {args.model}  ({spec.hf_id})")
    loaded = models.load_model(spec)
    n_layers = loaded.model.cfg.n_layers
    d_model = loaded.model.cfg.d_model
    print(f"         n_layers={n_layers} d_model={d_model}")

    report = {
        "model": args.model, "hf_id": spec.hf_id,
        "n_layers": n_layers, "d_model": d_model,
        "random_floor": bt.random_floor(d_model),
        "thresholds": {"pc_min_accuracy": bs.PC_MIN_ACCURACY,
                       "pc_min_z": bs.PC_MIN_Z,
                       "quintile": args.quintile,
                       "distinguishable_margin": bt.DEFAULT_MARGIN},
        "categories": {},
    }

    # ---------------- stage 1: positive control ----------------
    say("STAGE 1 - positive control (disambiguated, no option list)")
    print(f"{'category':<22}{'n':>6}{'acc':>9}{'z':>8}   gate")
    print("-" * 52)
    usable = []
    for cat in args.categories:
        pc = bs.positive_control(loaded, cat, DEFAULT_SYS,
                                 limit=args.control_limit, seed=args.seed)
        report["categories"][cat] = {"positive_control": pc}
        acc_s = f"{pc['accuracy']:.1%}" if pc["accuracy"] is not None else "-"
        z_s = f"{pc['z_vs_chance']:+.1f}" if pc["z_vs_chance"] is not None else "-"
        print(f"{cat:<22}{pc['n']:>6}{acc_s:>9}{z_s:>8}   "
              f"{'PASS' if pc['passes'] else 'FAIL - dropped'}")
        if pc["passes"]:
            usable.append(cat)

    print(f"\nusable categories: {len(usable)}/{len(args.categories)}")
    if len(usable) < 3:
        print("\n*** Fewer than 3 categories pass the positive control. There is")
        print("*** nothing to cluster. Stopping before extraction.")
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        return 1

    # ---------------- stage 2: margins ----------------
    say("STAGE 2 - stereotype margins on ambiguous items")
    print(f"{'category':<22}{'n':>6}{'mean':>9}{'sd':>8}{'top_q':>9}{'bot_q':>9}   sign split")
    print("-" * 74)
    msets = {}
    for cat in usable:
        ms = bs.margins(loaded, cat, DEFAULT_SYS, limit=args.ambig_limit, seed=args.seed)
        msets[cat] = ms
        m = np.asarray(ms.margins)
        top_i, bot_i = ms.extremes(args.quintile)
        tq, bq = float(m[top_i].mean()), float(m[bot_i].mean())
        ok_sign = tq > 0 > bq
        report["categories"][cat]["margins"] = {
            "n": len(m), "mean": float(m.mean()), "sd": float(m.std(ddof=1)),
            "top_quintile_mean": tq, "bottom_quintile_mean": bq,
            "quintile_n": len(top_i), "sign_separation": bool(ok_sign),
            "median_abstention_margin": float(np.median(ms.abstention)),
        }
        print(f"{cat:<22}{len(m):>6}{m.mean():>+9.3f}{m.std(ddof=1):>8.3f}"
              f"{tq:>+9.3f}{bq:>+9.3f}   {'opposite signs' if ok_sign else 'SAME SIGN (weak)'}")

    # ---------------- stage 3: extract ----------------
    say("STAGE 3 - extract a direction per category from the margin extremes")
    directions, bucket_resid = {}, {}
    for cat in usable:
        ms = msets[cat]
        top_i, bot_i = ms.extremes(args.quintile)
        prompts_top = [bs.bare_prompt(ms.items[i][0]) for i in top_i]
        prompts_bot = [bs.bare_prompt(ms.items[i][0]) for i in bot_i]
        r_top = bs.capture_prompt_residuals(loaded, prompts_top, DEFAULT_SYS)
        r_bot = bs.capture_prompt_residuals(loaded, prompts_bot, DEFAULT_SYS)
        bucket_resid[cat] = (r_top, r_bot)
        d = bt.assert_direction(r_top.mean(axis=0) - r_bot.mean(axis=0), name=cat)
        directions[cat] = d
        print(f"  {cat:<22} top n={len(top_i):<4} bottom n={len(bot_i):<4} "
              f"-> direction {d.shape}")
        np.save(out_dir / f"direction_{cat}.npy", d)

    # ---------------- stage 4: extraction floor ----------------
    say("STAGE 4 - extraction floor (re-extract the SAME category from halves)")

    def make_extractor(cat):
        r_top, r_bot = bucket_resid[cat]

        def extract(idx_pairs):
            ti = [i for kind, i in idx_pairs if kind == "t"]
            bi = [i for kind, i in idx_pairs if kind == "b"]
            if not ti or not bi:
                return np.zeros((n_layers, d_model))
            return r_top[ti].mean(axis=0) - r_bot[bi].mean(axis=0)
        return extract

    floors = {}
    for cat in usable:
        r_top, r_bot = bucket_resid[cat]
        pool = [("t", i) for i in range(len(r_top))] + [("b", i) for i in range(len(r_bot))]
        floors[cat] = bt.extraction_floor(pool, make_extractor(cat),
                                          n_splits=args.floor_splits, seed=args.seed)
        report["categories"][cat]["extraction_floor"] = floors[cat]

    rep = bt.TaxonomyReport(topics=usable, floors=floors,
                            random_floor=bt.random_floor(d_model))
    print(rep.floor_table())
    spread = rep.n_spread()
    if spread:
        print(f"\n  n spread across categories: {spread:.1f}x")

    # ---------------- stage 5: cosines, clustering, null ----------------
    say("STAGE 5 - cosine matrix, clustering, permutation null")
    names, M = bt.cosine_matrix(directions)
    report["cosine_matrix"] = {"names": names, "matrix": M.tolist()}

    print("      " + "".join(f"{n[:8]:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:20]:<20}" + "".join(f"{M[i][j]:>10.3f}" for j in range(len(names))))

    print(f"\n  random-direction floor (1/sqrt(d)) = {bt.random_floor(d_model):.4f}")
    print("\n  pair verdicts against each pair's own extraction floor:")
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            f = min(floors[names[i]]["q05"], floors[names[j]]["q05"])
            dist = bt.distinguishable(M[i][j], f)
            pair_rows.append({"a": names[i], "b": names[j], "cosine": float(M[i][j]),
                              "floor_q05": f, "distinguishable": dist})
            print(f"    {names[i][:18]:<19}{names[j][:18]:<19} cos={M[i][j]:+.3f} "
                  f"floor={f:.3f}  {'DISTINCT' if dist else 'not distinguishable'}")
    report["pairs"] = pair_rows

    Z = bt.cluster_topics(names, M)
    observed = bt.cluster_strength(Z)
    report["cluster_strength"] = observed
    report["linkage"] = np.asarray(Z).tolist()

    pool_by_topic = {}
    for cat in usable:
        r_top, r_bot = bucket_resid[cat]
        pool_by_topic[cat] = ([("t", i) for i in range(len(r_top))]
                              + [("b", i) for i in range(len(r_bot))])

    # The null must reshuffle ITEMS across topics, so it needs one extractor over
    # a shared pool rather than per-category arrays.
    all_top = np.concatenate([bucket_resid[c][0] for c in usable], axis=0)
    all_bot = np.concatenate([bucket_resid[c][1] for c in usable], axis=0)

    def global_extract(idx_pairs):
        ti = [i for kind, i in idx_pairs if kind == "t"]
        bi = [i for kind, i in idx_pairs if kind == "b"]
        if not ti or not bi:
            return np.zeros((n_layers, d_model))
        return all_top[ti].mean(axis=0) - all_bot[bi].mean(axis=0)

    off_t = off_b = 0
    global_pools = {}
    for c in usable:
        nt, nb = len(bucket_resid[c][0]), len(bucket_resid[c][1])
        global_pools[c] = ([("t", off_t + i) for i in range(nt)]
                           + [("b", off_b + i) for i in range(nb)])
        off_t += nt
        off_b += nb

    null = bt.permutation_null(global_pools, global_extract,
                               n_permutations=args.permutations, seed=args.seed)
    p = bt.null_p_value(observed, null)
    report["permutation_null"] = {"median": null["median"], "q95": null["q95"],
                                 "max": null["max"], "n": null["n_permutations"]}
    report["p_value"] = p
    print(f"\n  cluster strength observed : {observed:.4f}")
    print(f"  permutation null median   : {null['median']:.4f}  q95 {null['q95']:.4f}")
    print(f"  p                          : {p:.4f}")

    # ---------------- stage 6: floor vs n ----------------
    say("STAGE 6 - how much of the floor is sample size?")
    biggest = max(usable, key=lambda c: floors[c]["n_items"])
    smallest_n = min(floors[c]["n_items"] for c in usable)
    try:
        fvn = bt.floor_vs_n(pool_by_topic[biggest], make_extractor(biggest),
                            [floors[biggest]["n_items"], smallest_n],
                            n_splits=args.floor_splits, seed=args.seed)
        report["floor_vs_n"] = {"category": biggest,
                                "result": {str(k): v for k, v in fvn.items()}}
        print(f"  on {biggest}:")
        print(bt.summarize_floor_vs_n(fvn))
    except ValueError as e:
        print(f"  skipped: {e}")

    # ---------------- verdict ----------------
    rep.matrix = M
    rep.observed_strength = observed
    rep.null = null
    rep.p_value = p
    say("VERDICT")
    print("  " + rep.verdict())
    report["verdict"] = rep.verdict()

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    np.save(out_dir / "cosine_matrix.npy", M)
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
