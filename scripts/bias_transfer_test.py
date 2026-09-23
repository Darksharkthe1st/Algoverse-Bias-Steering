"""Step 6: are the bias directions FUNCTIONALLY different, not just geometrically?

    python scripts/bias_transfer_test.py --run-dir runs/full_qwen14b --model qwen-14b

The cosine matrix says whether two directions point different ways. It does not
say whether that difference does any work. This does: steer one category's
prompts with ANOTHER category's direction and measure what happens to the
stereotype margin.

    effect(v, C) = mean margin on category C's items when steered with v,
                   minus the unsteered mean margin

Read the resulting matrix by rows:

    strong on the diagonal, weak off it  -> the directions are specific; a
                                            taxonomy is doing real work
    uniform everywhere                   -> one shared knob; the geometric
                                            differences do not matter
    nothing anywhere                     -> no detectable effect at this dose

Two properties make this cheap and clean. Because scoring is by likelihood,
steering is measured as a shift in a continuous margin - no generation, no
parser, no judge. And because the margin is signed, we can check the SIGN of the
effect, not just its size.

DOSE NORMALISATION - added after the first run was found to be confounded.
A mean-difference direction's norm depends on how far apart its two pole means
landed in activation space, and across qwen-14b's ten categories that varies by
5x (Frobenius 100 to 314). It also correlates with the extraction floor at
pearson +0.91. `apply_resid_pre_add` multiplies by a fixed coefficient, so the
first run delivered a 5x stronger perturbation to exactly the categories whose
directions reproduce - and those were precisely the ones that then failed the
sign-flip control, looking like generic damage. Each layer is now unit-normalised
by default so the coefficient means the same thing everywhere, and the dose lives
entirely in the coefficient. `--no-normalize` reproduces the old behaviour.

Controls, all required before any of this is reportable (AGENTS.md section 5):

  - a norm-matched random direction per category. Per-layer norms span
    600-1391x within a model, so an isotropic random vector mostly tests
    magnitude. Matching the norm profile makes the control test direction.
  - both signs. If +v and -v move the margin the same way, we are measuring
    generic damage from perturbing the residual stream, not a direction.

Still missing, and so this remains UNCONTROLLED: the coefficient is chosen rather
than derived, the random control is norm-matched rather than covariance-matched,
there is no coherence check on the generations, and there is no system-prompt
baseline. Do not describe any result here as causal.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.bias_steer import bbq_score as bs  # noqa: E402
from src.bias_steer import bias_taxonomy as bt  # noqa: E402


def mean_margin(loaded, items, system_prompt, hooks=None, limit=None) -> float:
    """Mean stereotype margin over `items`, optionally under steering."""
    vals = []
    for e, r in items[:limit]:
        a = e.metadata["answers"]
        s = bs.score_answers(loaded, bs.bare_prompt(e),
                             [a[r.biased], a[r.nonstereo]], system_prompt,
                             fwd_hooks=hooks)
        vals.append(s[0] - s[1])
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir holding direction_*.npy")
    ap.add_argument("--model", default="qwen-1.8b")
    ap.add_argument("--items", type=int, default=120, help="items per category")
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-normalize", action="store_true",
                    help="steer with the raw direction instead of unit-normalising "
                         "each layer. Reproduces the confounded first run, in which "
                         "direction norms varied 5x across categories and a fixed "
                         "coefficient therefore delivered a different dose to each.")
    ap.add_argument("--only-reproducible", action="store_true",
                    help="restrict to categories whose direction reproduces "
                         "(floor q05 >= MIN_USABLE_FLOOR), read from report.json")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()
    normalize = not args.no_normalize

    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS

    run_dir = Path(args.run_dir)
    dirs = {}
    for p in sorted(run_dir.glob("direction_*.npy")):
        cat = p.stem.replace("direction_", "")
        dirs[cat] = bt.assert_direction(np.load(p), name=cat)

    if args.only_reproducible:
        rep = json.loads((run_dir / "report.json").read_text())
        keep = {c for c, v in rep.get("categories", {}).items()
                if "extraction_floor" in v
                and bt.floor_is_usable(v["extraction_floor"])}
        dropped = sorted(set(dirs) - keep)
        dirs = {c: d for c, d in dirs.items() if c in keep}
        print(f"restricted to reproducible directions; dropped {dropped}")

    if len(dirs) < 2:
        print(f"need at least 2 directions in {run_dir}; found {len(dirs)}")
        return 1
    cats = sorted(dirs)
    print(f"directions: {cats}")
    print(f"dose normalisation: {'ON (unit per layer)' if normalize else 'OFF (raw)'}")

    loaded = models.load_model(MODELS[args.model])

    print("\nloading items per category ...")
    items = {c: bs.load_scoreable(c, "ambig", args.items, args.seed) for c in cats}
    for c in cats:
        print(f"  {c:<22} {len(items[c])} items")

    print("\nbaseline (unsteered) mean margins:")
    base = {}
    for c in cats:
        base[c] = mean_margin(loaded, items[c], DEFAULT_SYS, limit=args.items)
        print(f"  {c:<22}{base[c]:>+9.4f}")

    report = {"model": args.model, "coeff": args.coeff, "normalized": normalize,
              "categories": cats, "baseline_margin": base,
              "effects": {}, "controls": {}}

    print(f"\ntransfer matrix: change in mean margin when steering with each "
          f"direction (coeff={args.coeff:+.1f})")
    print("rows = direction used, columns = category steered\n")
    print(f"{'direction':<22}" + "".join(f"{c[:9]:>11}" for c in cats))
    print("-" * (22 + 11 * len(cats)))

    eff = np.zeros((len(cats), len(cats)))
    for i, dc in enumerate(cats):
        hooks = bs.steering_hooks(loaded, dirs[dc], args.coeff, normalize=normalize)
        row = []
        for j, tc in enumerate(cats):
            m = mean_margin(loaded, items[tc], DEFAULT_SYS, hooks=hooks, limit=args.items)
            eff[i][j] = m - base[tc]
            row.append(eff[i][j])
        report["effects"][dc] = {c: float(v) for c, v in zip(cats, row)}
        print(f"{dc:<22}" + "".join(f"{v:>+11.4f}" for v in row))

    # ---- control 1: norm-matched random directions -----------------------
    print("\ncontrol - norm-matched random direction (should be ~0 everywhere):")
    print(f"{'random(for)':<22}" + "".join(f"{c[:9]:>11}" for c in cats))
    print("-" * (22 + 11 * len(cats)))
    rnd_eff = np.zeros((len(cats), len(cats)))
    for i, dc in enumerate(cats):
        rv = bs.norm_matched_random(dirs[dc], seed=args.seed + i)
        hooks = bs.steering_hooks(loaded, rv, args.coeff, normalize=normalize)
        row = []
        for j, tc in enumerate(cats):
            m = mean_margin(loaded, items[tc], DEFAULT_SYS, hooks=hooks, limit=args.items)
            rnd_eff[i][j] = m - base[tc]
            row.append(rnd_eff[i][j])
        report["controls"][f"random_{dc}"] = {c: float(v) for c, v in zip(cats, row)}
        print(f"{'rand~' + dc[:16]:<22}" + "".join(f"{v:>+11.4f}" for v in row))

    # ---- control 2: sign flip on the diagonal ----------------------------
    print(f"\ncontrol - sign flip on own category (coeff={-args.coeff:+.1f}).")
    print("A real direction should move the margin the OTHER way; generic damage")
    print("from perturbing the residual stream would move it the same way.\n")
    print(f"{'category':<22}{'+coeff':>11}{'-coeff':>11}   verdict")
    print("-" * 58)
    flips = {}
    for i, c in enumerate(cats):
        hooks = bs.steering_hooks(loaded, dirs[c], -args.coeff, normalize=normalize)
        m = mean_margin(loaded, items[c], DEFAULT_SYS, hooks=hooks, limit=args.items)
        neg = m - base[c]
        pos = eff[i][i]
        opposite = (pos > 0) != (neg > 0)
        flips[c] = {"pos": float(pos), "neg": float(neg), "opposite_signs": bool(opposite)}
        print(f"{c:<22}{pos:>+11.4f}{neg:>+11.4f}   "
              f"{'sign flips (good)' if opposite else 'SAME SIGN - generic damage?'}")
    report["sign_flip"] = flips

    # ---- summary ----------------------------------------------------------
    diag = np.array([eff[i][i] for i in range(len(cats))])
    off = np.array([eff[i][j] for i in range(len(cats))
                    for j in range(len(cats)) if i != j])
    rnd = rnd_eff.flatten()
    summary = {
        "mean_own_effect": float(diag.mean()),
        "mean_cross_effect": float(off.mean()) if off.size else None,
        "mean_random_effect": float(np.abs(rnd).mean()),
        "specificity_own_minus_cross": float(diag.mean() - off.mean()) if off.size else None,
        "n_sign_flips_good": int(sum(1 for v in flips.values() if v["opposite_signs"])),
        "n_categories": len(cats),
    }
    report["summary"] = summary

    print("\nsummary")
    print(f"  mean effect, own direction        : {summary['mean_own_effect']:+.4f}")
    print(f"  mean effect, cross direction      : {summary['mean_cross_effect']:+.4f}")
    print(f"  mean |effect|, random control     : {summary['mean_random_effect']:+.4f}")
    print(f"  specificity (own - cross)         : {summary['specificity_own_minus_cross']:+.4f}")
    print(f"  sign-flip control passed          : "
          f"{summary['n_sign_flips_good']}/{summary['n_categories']}")
    print("\n  own >> cross  -> directions are specific; the taxonomy does work")
    print("  own ~= cross  -> one shared knob; geometry differs but function does not")
    print("  own ~= random -> no detectable effect at this dose")
    print("\n  UNCONTROLLED: coefficient chosen not derived, random control is")
    print("  norm-matched not covariance-matched, no coherence check, no")
    print("  system-prompt baseline. Do not call any of this causal.")
    print("  Say 'a direction', never 'the direction' (arXiv:2602.06801).")

    suffix = args.tag or ("" if normalize else "_raw")
    out = run_dir / f"transfer_test{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
