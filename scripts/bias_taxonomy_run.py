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


def _git_sha():
    """Commit this run executed from, or None outside a repo."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _extract(args, resid, marg, n_layers, d_model, cat):
    """One direction from residuals + margins, by whichever method is selected."""
    if args.method == "probe":
        if len(marg) < 3:
            return np.zeros((n_layers, d_model))
        return bt.probe_direction(resid, marg, alpha=args.alpha)

    if args.tail_trim > 0:
        top, bot = bt.trimmed_extremes(list(marg), quintile=args.quintile,
                                       trim=args.tail_trim)
    else:
        # historical behaviour, byte-for-byte: argsort then quintile
        order = np.argsort(marg)
        k = max(1, int(len(order) * args.quintile))
        top, bot = order[-k:], order[:k]
    if len(top) == 0 or len(bot) == 0:
        return np.zeros((n_layers, d_model))
    return bt.assert_direction(
        resid[list(top)].mean(axis=0) - resid[list(bot)].mean(axis=0), name=cat)


def _p3_manifest_sha():
    """Hash of the frozen P3 subset list, or None if this is not a subset run.

    A subset floor is only interpretable against the subset list that was fixed
    before any floor was seen. Recording the hash is what lets a reader check
    that the split was not chosen after the fact.
    """
    import hashlib
    import pathlib
    p = pathlib.Path("runs/_p3_manifest.json")
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--categories", nargs="+", default=bs.CATEGORIES)
    ap.add_argument("--ambig-limit", type=int, default=400)
    ap.add_argument("--control-limit", type=int, default=150)
    ap.add_argument("--quintile", type=float, default=0.20)
    ap.add_argument("--method", choices=["extremes", "probe"], default="extremes",
                    help="extremes = difference of means over the top/bottom "
                         "quintile of the margin. probe = per-layer ridge "
                         "regression of the margin onto the residuals, which "
                         "uses every item and every gradation instead of "
                         "discarding 60%% of them and binarising the rest.")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--tail-trim", type=float, default=0.0,
                    help="extremes only: drop this fraction of items at EACH "
                         "margin extreme before taking quintiles (heavy-tail "
                         "robustness check; 0 = historical behaviour)")
    ap.add_argument("--winsorise", type=float, default=0.0,
                    help="probe only: clip margin targets at the q / 1-q "
                         "quantiles before fitting (0 = off). Deliberately a "
                         "no-op for extremes, whose selection is rank-based - "
                         "use --tail-trim there.")
    ap.add_argument("--save-residuals", action="store_true",
                    help="persist each category's captured residual tensor as "
                         "fp16 npz in the run dir (~250 MB/category at 14B "
                         "scale). Never commit these; checkpoint them off-box "
                         "so follow-up analyses stay CPU-only.")
    ap.add_argument("--stereotyped-group", default=None,
                    help="WP-43 P3: restrict to items whose BBQ-annotated "
                         "stereotyped_groups contains this label (e.g. 'black'). "
                         "Applied AFTER sampling, so the subset is a strict "
                         "subset of the pooled run at the same --ambig-limit "
                         "and --seed, and reuses its cached margins. Subsets "
                         "must come from runs/_p3_manifest.json, which fixes "
                         "the list before any floor is computed.")
    ap.add_argument("--margins-cache", default=None,
                    help="dir for cached per-item margins (default runs/_margins_cache)")
    ap.add_argument("--refresh-margins", action="store_true",
                    help="recompute margins even if a cache entry exists")
    ap.add_argument("--cluster-usable-only", action="store_true",
                    help="cluster only categories whose direction reproduces "
                         "(floor q05 >= MIN_USABLE_FLOOR). Directions that do "
                         "not reproduce contribute noise to the matrix and to "
                         "the null, so including them makes the p-value a test "
                         "of the wrong thing.")
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
    # Margins are the expensive stage (three forward passes per item) and they do
    # not depend on the extraction method, so they are cached per
    # (model, category, limit, seed). That is what makes it affordable to re-run
    # the analysis with a different estimator instead of re-scoring for hours.
    cache_dir = Path(args.margins_cache or "runs/_margins_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"margin cache: {cache_dir}")
    print(f"{'category':<22}{'n':>6}{'mean':>9}{'sd':>8}{'top_q':>9}{'bot_q':>9}   sign split")
    print("-" * 74)
    msets = {}
    for cat in usable:
        key = (f"{args.model}_{cat}_{args.ambig_limit}_{args.seed}.json"
               if not args.stereotyped_group else
               f"{args.model}_{cat}_{args.ambig_limit}_{args.seed}"
               f"_grp-{args.stereotyped_group.replace(' ', '-')}.json")
        cpath = cache_dir / key
        ms = None
        if cpath.exists() and not args.refresh_margins:
            try:
                blob = json.loads(cpath.read_text())
                items = bs.load_scoreable(cat, "ambig", args.ambig_limit, args.seed,
                                          stereotyped_group=args.stereotyped_group)
                if len(items) == len(blob["margins"]) and \
                        [e.id for e, _ in items] == blob["ids"]:
                    ms = bs.MarginSet(category=cat, items=items,
                                      margins=blob["margins"],
                                      abstention=blob["abstention"])
                    print(f"  (cached) ", end="")
            except Exception:
                ms = None
        # A P3 subset is a strict subset of the pooled run at the same
        # (limit, seed), so the pooled run's cached margins already contain
        # every item this subset needs. Slice them rather than paying three
        # forward passes per item again -- this is what makes P3 cost no GPU
        # time on top of the pooled run.
        if ms is None and args.stereotyped_group:
            pooled = cache_dir / f"{args.model}_{cat}_{args.ambig_limit}_{args.seed}.json"
            if pooled.exists():
                blob = json.loads(pooled.read_text())
                pos = {i: k for k, i in enumerate(blob["ids"])}
                items = bs.load_scoreable(cat, "ambig", args.ambig_limit, args.seed,
                                          stereotyped_group=args.stereotyped_group)
                missing = [e.id for e, _ in items if e.id not in pos]
                if missing:
                    print(f"  (pooled cache missing {len(missing)} subset items; rescoring)")
                else:
                    idx = [pos[e.id] for e, _ in items]
                    ms = bs.MarginSet(
                        category=cat, items=items,
                        margins=[blob["margins"][i] for i in idx],
                        abstention=[blob["abstention"][i] for i in idx])
                    cpath.write_text(json.dumps({
                        "ids": [e.id for e, _ in items],
                        "margins": ms.margins, "abstention": ms.abstention,
                        "sliced_from": pooled.name}))
                    print(f"  (sliced {len(idx)} of {len(blob['ids'])} from pooled cache) ",
                          end="")

        if ms is None:
            ms = bs.margins(loaded, cat, DEFAULT_SYS,
                            limit=args.ambig_limit, seed=args.seed,
                            stereotyped_group=args.stereotyped_group)
            cpath.write_text(json.dumps({
                "ids": [e.id for e, _ in ms.items],
                "margins": ms.margins, "abstention": ms.abstention}))
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
    say(f"STAGE 3 - extract a direction per category  (method: {args.method})")
    report["method"] = args.method
    # Provenance: every choice that produced these numbers, recorded at run
    # time so no future audit has to reconstruct it from folder names and logs
    # (that reconstruction is exactly what 09-open-questions Q2-Q6 had to do).
    report["estimator_params"] = {
        "method": args.method,
        "alpha": args.alpha if args.method == "probe" else None,
        "quintile": args.quintile if args.method == "extremes" else None,
        "tail_trim": args.tail_trim if args.method == "extremes" else None,
        "winsorise": args.winsorise if args.method == "probe" else None,
    }
    report["sampling"] = {
        "ambig_limit": args.ambig_limit, "control_limit": args.control_limit,
        "seed": args.seed, "floor_splits": args.floor_splits,
        "permutations": args.permutations,
        "stereotyped_group": args.stereotyped_group,
        "p3_manifest_sha256": _p3_manifest_sha(),
    }
    report["code_version"] = _git_sha()
    directions, cat_resid, cat_margin = {}, {}, {}
    for cat in usable:
        ms = msets[cat]
        if args.method == "probe":
            # every item, so the probe sees the full gradation
            idx = list(range(len(ms.items)))
        elif args.tail_trim > 0:
            top_i, bot_i = bt.trimmed_extremes(ms.margins,
                                               quintile=args.quintile,
                                               trim=args.tail_trim)
            idx = list(top_i) + list(bot_i)
        else:
            top_i, bot_i = ms.extremes(args.quintile)
            idx = list(top_i) + list(bot_i)
        prompts = [bs.bare_prompt(ms.items[i][0]) for i in idx]
        resid = bs.capture_prompt_residuals(loaded, prompts, DEFAULT_SYS)
        marg = np.asarray([ms.margins[i] for i in idx])
        if args.method == "probe" and args.winsorise > 0:
            marg = bt.winsorise(marg, args.winsorise)
        cat_resid[cat], cat_margin[cat] = resid, marg
        if args.save_residuals:
            man = bt.save_residuals(out_dir / f"residuals_{cat}.npz", resid,
                                    [ms.items[i][0].id for i in idx], marg)
            report["categories"][cat]["residuals"] = man
            print(f"  {cat:<22} residuals -> {man['path']} "
                  f"({man['bytes'] / 1e6:.0f} MB)")

        d = _extract(args, resid, marg, n_layers, d_model, cat)
        directions[cat] = d
        print(f"  {cat:<22} n={len(idx):<5} -> direction {d.shape}")
        np.save(out_dir / f"direction_{cat}.npy", d)

    # ---------------- stage 4: extraction floor ----------------
    say("STAGE 4 - extraction floor (re-extract the SAME category from halves)")

    def make_extractor(cat):
        resid, marg = cat_resid[cat], cat_margin[cat]

        def extract(idx):
            idx = list(idx)
            return _extract(args, resid[idx], marg[idx], n_layers, d_model, cat)
        return extract

    floors = {}
    for cat in usable:
        pool = list(range(len(cat_margin[cat])))
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

    cluster_cats = usable
    if args.cluster_usable_only:
        cluster_cats = [c for c in usable if bt.floor_is_usable(floors[c])]
        dropped = [c for c in usable if c not in cluster_cats]
        print(f"  clustering only the {len(cluster_cats)} reproducible categories")
        if dropped:
            print(f"  excluded (floor below {bt.MIN_USABLE_FLOOR}): {dropped}")
        if len(cluster_cats) < 3:
            # Do NOT hand this to TaxonomyReport with a placeholder p-value: its
            # verdict would read "clustering is within the permutation null
            # (p=1.000)", which asserts that a null was run and not beaten. No
            # clustering happened at all. Say what is true.
            report["verdict"] = (
                f"NOT CLUSTERABLE: only {len(cluster_cats)} of {len(usable)} "
                f"categories produce a direction that reproduces against itself "
                f"(floor q05 >= {bt.MIN_USABLE_FLOOR}) — {cluster_cats}. Three are "
                f"needed to cluster, so no similarity structure was computed and "
                f"no permutation null was run. This is neither evidence for nor "
                f"against separable subtypes.")
            report["p_value"] = None
            report["cluster_strength"] = None
            print("\n  fewer than 3 reproducible directions — nothing to cluster.")
            print("\n=== VERDICT\n  " + report["verdict"])
            (out_dir / "report.json").write_text(json.dumps(report, indent=2))
            return 0
        directions = {c: directions[c] for c in cluster_cats}

    names, M = bt.cosine_matrix(directions)
    report["cosine_matrix"] = {"names": names, "matrix": M.tolist()}

    print("      " + "".join(f"{n[:8]:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:20]:<20}" + "".join(f"{M[i][j]:>10.3f}" for j in range(len(names))))

    print(f"\n  random-direction floor (1/sqrt(d)) = {bt.random_floor(d_model):.4f}")
    print("\n  pair verdicts. 'indeterminate' means at least one of the two")
    print("  directions does not reproduce against itself, so the cosine between")
    print(f"  them carries no information (usable floor >= {bt.MIN_USABLE_FLOOR}):")
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fa, fb = floors[names[i]], floors[names[j]]
            v = bt.pair_verdict(M[i][j], fa, fb)
            pair_rows.append({"a": names[i], "b": names[j], "cosine": float(M[i][j]),
                              "floor_a": fa["q05"], "floor_b": fb["q05"], "verdict": v})
            print(f"    {names[i][:18]:<19}{names[j][:18]:<19} cos={M[i][j]:+.3f} "
                  f"floors={fa['q05']:+.3f}/{fb['q05']:+.3f}  {v}")
    report["pairs"] = pair_rows

    Z = bt.cluster_topics(names, M)
    observed = bt.cluster_strength(Z)
    report["cluster_strength"] = observed
    report["linkage"] = np.asarray(Z).tolist()

    # The null reshuffles ITEMS across topics, so it needs one extractor over a
    # shared pool. Margins travel with their items: a shuffled group is a real
    # mix of items carrying their own real margins, which is exactly the point —
    # it asks whether the TOPIC LABELS are doing any work.
    all_resid = np.concatenate([cat_resid[c] for c in cluster_cats], axis=0)
    all_marg = np.concatenate([cat_margin[c] for c in cluster_cats], axis=0)

    def global_extract(idx):
        idx = list(idx)
        if len(idx) < 3:
            return np.zeros((n_layers, d_model))
        return _extract(args, all_resid[idx], all_marg[idx], n_layers, d_model, "null")

    global_pools, off = {}, 0
    for c in cluster_cats:
        k = len(cat_margin[c])
        global_pools[c] = list(range(off, off + k))
        off += k

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
        fvn = bt.floor_vs_n(list(range(len(cat_margin[biggest]))),
                            make_extractor(biggest),
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
