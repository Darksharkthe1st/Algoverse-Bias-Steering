"""Was the probe's failure real, or just an untuned ridge penalty?

    python scripts/probe_alpha_sweep.py --model qwen-14b

The ridge probe produced 0/10 reproducible directions on qwen-14b against 4/10
for the extremes contrast. That was read as "the probe is the worse estimator" —
but `alpha=1.0` was never tuned, and against d_model=5120 with n=600 that is
essentially unregularised, so the probe was almost certainly overfitting. An
overfitted probe cannot reproduce across a split-half by construction.

Until the penalty is swept, "the probe is worse" and "the probe was
misconfigured" are indistinguishable, and the conclusion that some categories
have no recoverable direction rests partly on that ambiguity.

Residuals are captured ONCE and reused for every alpha, so the sweep costs one
capture pass rather than one per setting. Margins come from the cache written by
bias_taxonomy_run.py.

Reports the extraction floor per (category, alpha), and whether any alpha lifts
a category over the usability threshold that the extremes contrast missed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.bias_steer import bbq_score as bs  # noqa: E402
from src.bias_steer import bias_taxonomy as bt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-14b")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--ambig-limit", type=int, default=600)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1.0, 1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--floor-splits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margins-cache", default="runs/_margins_cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    cats = args.categories or bs.CATEGORIES
    cache_dir = Path(args.margins_cache)

    loaded = models.load_model(MODELS[args.model])
    n_layers, d_model = loaded.model.cfg.n_layers, loaded.model.cfg.d_model
    print(f"{args.model}  n_layers={n_layers} d_model={d_model}")
    print(f"alphas: {args.alphas}\n")

    report = {"model": args.model, "alphas": args.alphas, "categories": {}}
    header = f"{'category':<24}" + "".join(f"{a:>11.0e}" for a in args.alphas)
    print(header)
    print("-" * len(header))

    for cat in cats:
        cpath = cache_dir / f"{args.model}_{cat}_{args.ambig_limit}_{args.seed}.json"
        if not cpath.exists():
            print(f"{cat:<24}  (no cached margins - skipped)")
            continue
        blob = json.loads(cpath.read_text())
        items = bs.load_scoreable(cat, "ambig", args.ambig_limit, args.seed)
        if [e.id for e, _ in items] != blob["ids"]:
            print(f"{cat:<24}  (cache mismatch - skipped)")
            continue
        marg = np.asarray(blob["margins"])

        resid = bs.capture_prompt_residuals(
            loaded, [bs.bare_prompt(e) for e, _ in items], DEFAULT_SYS)

        row, per_alpha = "", {}
        for a in args.alphas:
            def extract(idx, _r=resid, _m=marg, _a=a):
                idx = list(idx)
                if len(idx) < 3:
                    return np.zeros((n_layers, d_model))
                return bt.probe_direction(_r[idx], _m[idx], alpha=_a)

            floor = bt.extraction_floor(list(range(len(marg))), extract,
                                        n_splits=args.floor_splits, seed=args.seed)
            per_alpha[str(a)] = floor
            mark = "*" if bt.floor_is_usable(floor) else " "
            row += f"{floor['q05']:>10.3f}{mark}"

        report["categories"][cat] = per_alpha
        print(f"{cat:<24}{row}")

    print("\n* clears the usability threshold "
          f"({bt.MIN_USABLE_FLOOR})")

    # did any alpha rescue a category?
    best = {}
    for cat, per in report["categories"].items():
        b = max(per.items(), key=lambda kv: kv[1]["q05"])
        best[cat] = {"alpha": b[0], "q05": b[1]["q05"],
                     "usable": bt.floor_is_usable(b[1])}
    rescued = [c for c, v in best.items() if v["usable"]]
    print(f"\nbest alpha per category:")
    for c, v in sorted(best.items(), key=lambda kv: -kv[1]["q05"]):
        print(f"  {c:<24} alpha={float(v['alpha']):>9.0e}  q05={v['q05']:+.3f}"
              f"  {'USABLE' if v['usable'] else ''}")
    print(f"\ncategories reproducible under SOME alpha: {len(rescued)} -> {rescued}")
    print("Compare against the extremes contrast, which reproduced 4/10 on")
    print("qwen-14b (Disability_status, Age, Religion, Physical_appearance).")
    print("\nIf a tuned probe lifts categories the extremes contrast missed, then")
    print("'no recoverable direction' was an estimator limit for those categories,")
    print("not a fact about the model.")

    report["best_alpha"] = best
    out = Path(args.out or f"runs/_probe_alpha_sweep_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
