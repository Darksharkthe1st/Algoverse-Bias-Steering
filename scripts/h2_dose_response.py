"""Direct test of H2: does the floor track the model's tilt, WITHIN a category?

    python scripts/h2_dose_response.py --model qwen-14b --category Age

H2 says a category has an extractable bias direction when the model leans the
same way across its items — mean margin predicts the extraction floor at
pearson +0.726 (14B) and +0.689 (1.8B), and mean/sd at +0.835.

That correlation is over ten points, one per category, and the categories differ
in many ways besides their tilt. This tests the same claim WITHIN a single
category, where topic, vocabulary, prompt format and item count are all held
fixed and only the tilt varies.

METHOD
    Take one category whose direction reproduces. Sort its items by margin. Build
    several sub-datasets of the SAME SIZE at different effect sizes:

      - "extremes"  : the most positive and most negative items (large |mean|)
      - "moderate"  : items from the middle of each half
      - "null-tilt" : items whose margins sit closest to zero (|mean| ~ 0)

    Extract a direction from each and measure its split-half floor. Same n, same
    category, same everything else.

PREDICTION, fixed before running
    floor(extremes) > floor(moderate) > floor(null-tilt), and the null-tilt
    subset should fail to reproduce (floor < 0.50) even though it has exactly as
    many items as the extremes subset.

    If instead all three reproduce equally well, H2 is wrong and the
    between-category correlation was driven by something else.

This is also the control against the mechanical worry: if the floor only tracked
sample size, all three subsets would score the same, because n is identical.
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
    ap.add_argument("--category", default="Age")
    ap.add_argument("--ambig-limit", type=int, default=600)
    ap.add_argument("--bucket", type=int, default=100,
                    help="items per pole in each sub-dataset (n held constant)")
    ap.add_argument("--floor-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margins-cache", default="runs/_margins_cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    loaded = models.load_model(MODELS[args.model])
    n_layers, d_model = loaded.model.cfg.n_layers, loaded.model.cfg.d_model
    print(f"{args.model}  n_layers={n_layers} d_model={d_model}")
    print(f"category: {args.category}\n")

    # margins (cached by bias_taxonomy_run.py if it has been run)
    cache = Path(args.margins_cache) / \
        f"{args.model}_{args.category}_{args.ambig_limit}_{args.seed}.json"
    items = bs.load_scoreable(args.category, "ambig", args.ambig_limit, args.seed)
    if cache.exists():
        blob = json.loads(cache.read_text())
        if [e.id for e, _ in items] == blob["ids"]:
            margins = np.asarray(blob["margins"])
            print(f"margins: loaded from cache ({len(margins)} items)")
        else:
            cache = None
    else:
        cache = None
    if cache is None:
        ms = bs.margins(loaded, args.category, DEFAULT_SYS,
                        limit=args.ambig_limit, seed=args.seed)
        items, margins = ms.items, np.asarray(ms.margins)
        print(f"margins: computed ({len(margins)} items)")

    order = np.argsort(margins)          # ascending
    n = len(order)
    b = args.bucket
    if n < 6 * b:
        print(f"need at least {6 * b} items for three disjoint pairs of buckets; "
              f"have {n}. Reduce --bucket.")
        return 1

    mid = n // 2
    subsets = {
        # most negative vs most positive
        "extremes": (order[:b], order[-b:]),
        # a step in from each end
        "moderate": (order[b:2 * b], order[-2 * b:-b]),
        # the b items either side of the median — margins closest to zero
        "null-tilt": (order[mid - b:mid], order[mid:mid + b]),
    }

    print(f"\n{'subset':<12}{'n/pole':>8}{'mean neg':>11}{'mean pos':>11}"
          f"{'separation':>12}{'floor q05':>11}{'median':>9}   reproduces?")
    print("-" * 88)

    report = {"model": args.model, "category": args.category,
              "bucket": b, "subsets": {}}
    for name, (lo, hi) in subsets.items():
        idx = list(lo) + list(hi)
        prompts = [bs.bare_prompt(items[i][0]) for i in idx]
        resid = bs.capture_prompt_residuals(loaded, prompts, DEFAULT_SYS)
        half = len(lo)

        def extract(sub_idx, _r=resid, _h=half):
            lo_i = [i for i in sub_idx if i < _h]
            hi_i = [i for i in sub_idx if i >= _h]
            if not lo_i or not hi_i:
                return np.zeros((n_layers, d_model))
            return _r[hi_i].mean(axis=0) - _r[lo_i].mean(axis=0)

        floor = bt.extraction_floor(list(range(len(idx))), extract,
                                    n_splits=args.floor_splits, seed=args.seed)
        mneg, mpos = float(margins[lo].mean()), float(margins[hi].mean())
        sep = mpos - mneg
        ok = bt.floor_is_usable(floor)
        print(f"{name:<12}{half:>8}{mneg:>+11.3f}{mpos:>+11.3f}{sep:>12.3f}"
              f"{floor['q05']:>11.3f}{floor['median']:>9.3f}   {'YES' if ok else 'no'}")
        report["subsets"][name] = {
            "n_per_pole": half, "mean_neg": mneg, "mean_pos": mpos,
            "separation": sep, "floor": floor, "reproduces": bool(ok)}

    e = report["subsets"]["extremes"]["floor"]["q05"]
    m = report["subsets"]["moderate"]["floor"]["q05"]
    z = report["subsets"]["null-tilt"]["floor"]["q05"]
    monotone = e > m > z
    print(f"\nprediction was floor(extremes) > floor(moderate) > floor(null-tilt)")
    print(f"observed: {e:.3f} > {m:.3f} > {z:.3f}   -> "
          f"{'SUPPORTED' if monotone else 'NOT supported'}")
    print(f"null-tilt subset reproduces: "
          f"{report['subsets']['null-tilt']['reproduces']} "
          f"(H2 predicts False, at identical n)")
    report["monotone"] = bool(monotone)

    out = Path(args.out or f"runs/_h2_dose_{args.model}_{args.category}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    print("\nn is identical across the three subsets, so a monotone result cannot")
    print("be explained by sample size — which is the mechanical worry about the")
    print("between-category correlation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
