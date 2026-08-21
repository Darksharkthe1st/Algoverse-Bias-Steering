"""Is the EXTRACTION broken, or is the bias margin genuinely not represented?

The full run returned a clean null: per-category extraction floors of -0.07 to
0.48, i.e. re-extracting the same category from random halves mostly gives
unrelated directions. Two very different explanations produce that:

  (a) the extraction machinery cannot recover ANY direction from these prompts —
      wrong capture site, too few items, something broken; or
  (b) extraction works fine, and the stereotype-margin contrast simply has no
      stable neural correlate at this model scale.

Those call for opposite responses, so guessing between them is not acceptable.

THE CONTROL
    Extract a direction for something that MUST be represented if anything is:
    topic identity. Race_ethnicity prompts versus Gender_identity prompts differ
    in vocabulary, entities and subject matter. If the residual stream at the
    final prompt token carries anything at all, it carries that.

    Run the identical pipeline on it — same capture site, same mean-difference,
    same split-half floor, comparable n.

READING IT
    topic floor HIGH, bias floor ~0  -> extraction works; the bias contrast is
                                        the thing with no stable correlate.
                                        The null is real and reportable.
    topic floor ALSO ~0             -> the machinery is broken. Fix that before
                                        believing any null.

This is the same discipline as the project's G1 gate, applied one level down:
prove the instrument can measure something before reporting that it measured
nothing.

    python scripts/extraction_positive_control.py --model qwen-1.8b
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
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--pairs", nargs="+",
                    default=["Race_ethnicity:Gender_identity",
                             "Religion:Age",
                             "Nationality:Sexual_orientation"])
    ap.add_argument("--items", type=int, default=160)
    ap.add_argument("--floor-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_extraction_positive_control.json")
    args = ap.parse_args()

    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    loaded = models.load_model(MODELS[args.model])
    n_layers = loaded.model.cfg.n_layers
    d_model = loaded.model.cfg.d_model
    print(f"model {args.model}  n_layers={n_layers} d_model={d_model}")
    print(f"random-direction floor (1/sqrt d) = {bt.random_floor(d_model):.4f}\n")

    report = {"model": args.model, "n_layers": n_layers, "d_model": d_model,
              "random_floor": bt.random_floor(d_model), "pairs": {}}

    print(f"{'topic contrast':<42}{'n':>6}{'floor q05':>11}{'median':>9}   verdict")
    print("-" * 82)

    for pair in args.pairs:
        a_cat, b_cat = pair.split(":")
        a_items = bs.load_scoreable(a_cat, "ambig", args.items, args.seed)
        b_items = bs.load_scoreable(b_cat, "ambig", args.items, args.seed)
        if not a_items or not b_items:
            print(f"{pair:<42}  — missing items —")
            continue

        r_a = bs.capture_prompt_residuals(
            loaded, [bs.bare_prompt(e) for e, _ in a_items], DEFAULT_SYS)
        r_b = bs.capture_prompt_residuals(
            loaded, [bs.bare_prompt(e) for e, _ in b_items], DEFAULT_SYS)

        def extract(idx_pairs, _ra=r_a, _rb=r_b):
            ai = [i for k, i in idx_pairs if k == "a"]
            bi = [i for k, i in idx_pairs if k == "b"]
            if not ai or not bi:
                return np.zeros((n_layers, d_model))
            return _ra[ai].mean(axis=0) - _rb[bi].mean(axis=0)

        pool = ([("a", i) for i in range(len(r_a))]
                + [("b", i) for i in range(len(r_b))])
        floor = bt.extraction_floor(pool, extract, n_splits=args.floor_splits,
                                    seed=args.seed)

        # A topic direction should be strongly reproducible. If this is near the
        # random floor the machinery, not the hypothesis, is at fault.
        ok = floor["q05"] >= 0.50
        print(f"{a_cat + ' vs ' + b_cat:<42}{floor['n_items']:>6}"
              f"{floor['q05']:>11.3f}{floor['median']:>9.3f}   "
              f"{'REPRODUCIBLE' if ok else 'NOT reproducible — machinery suspect'}")
        report["pairs"][pair] = {"floor": floor, "reproducible": bool(ok)}

    print("\nCompare against the bias-margin floors from the full run:")
    print("  Age 0.423 | Race_x_gender 0.138 | Nationality 0.057 | Race_x_SES 0.017")
    print("  Religion -0.008 | Gender_identity -0.072 | Race_ethnicity -0.115 |"
          " Sexual_orientation -0.202   (q05)")
    print("\nIf the topic floors are high and those are ~0, extraction works and")
    print("the stereotype-margin contrast is what has no stable correlate.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
