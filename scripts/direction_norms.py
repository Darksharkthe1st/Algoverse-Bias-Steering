"""Are the reproducible directions simply BIGGER? If so the transfer test is confounded.

    python scripts/direction_norms.py --run-dir runs/full_qwen14b

In both qwen-14b and gemma-2b, the categories whose directions reproduce best are
the ones that FAIL the sign-flip control — the margin moves the same way under
+coeff and -coeff, which reads as generic damage rather than steering.

There is an obvious mechanical explanation. A mean-difference direction's norm
scales with the separation between its two poles. Categories that reproduce are
exactly the ones with large separation, so their direction vectors are LARGER.
`steering_hooks` applies `(coeff / n_layers) * direction[layer]` with coeff fixed
across categories, so a larger direction injects a larger perturbation. Past some
magnitude the model degrades, and degradation is not sign-sensitive.

If direction norm correlates with the extraction floor, the transfer test as run
compared interventions of different strengths and its specificity numbers are not
interpretable. The fix is to unit-normalise each direction and put the dose in the
coefficient, which is what the Arditi ablation operator already does and what
`apply_resid_pre_add` does NOT.

CPU only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.bias_steer import bias_taxonomy as bt  # noqa: E402


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = p
        return r
    return pearson(ranks(xs), ranks(ys)) if len(xs) >= 3 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    rep = json.loads((run / "report.json").read_text())
    floors = {c: v["extraction_floor"]["q05"]
              for c, v in rep.get("categories", {}).items()
              if "extraction_floor" in v}
    seps = {}
    for c, v in rep.get("categories", {}).items():
        m = v.get("margins")
        if m and m.get("top_quintile_mean") is not None:
            seps[c] = m["top_quintile_mean"] - m["bottom_quintile_mean"]

    rows = []
    for p in sorted(run.glob("direction_*.npy")):
        c = p.stem.replace("direction_", "")
        d = bt.assert_direction(np.load(p), name=c)
        rows.append({
            "category": c,
            "frobenius_norm": float(np.linalg.norm(d)),
            "max_layer_norm": float(np.linalg.norm(d, axis=1).max()),
            "mean_layer_norm": float(np.linalg.norm(d, axis=1).mean()),
            "floor": floors.get(c),
            "quintile_separation": seps.get(c),
        })

    rows.sort(key=lambda r: -(r["floor"] if r["floor"] is not None else -99))
    print(f"run: {run.name}   model: {rep.get('model')}\n")
    print(f"{'category':<24}{'floor':>8}{'‖d‖_F':>12}{'max layer':>12}"
          f"{'separation':>12}")
    print("-" * 70)
    for r in rows:
        fl = f"{r['floor']:+.3f}" if r["floor"] is not None else "-"
        sp = f"{r['quintile_separation']:.2f}" if r["quintile_separation"] else "-"
        print(f"{r['category']:<24}{fl:>8}{r['frobenius_norm']:>12.2f}"
              f"{r['max_layer_norm']:>12.2f}{sp:>12}")

    have = [r for r in rows if r["floor"] is not None]
    if len(have) >= 3:
        f = [r["floor"] for r in have]
        nF = [r["frobenius_norm"] for r in have]
        print(f"\nfloor vs direction norm : pearson {pearson(f, nF):+.3f}"
              f"   spearman {spearman(f, nF):+.3f}")
        s = [r["quintile_separation"] for r in have if r["quintile_separation"]]
        if len(s) == len(have):
            print(f"norm vs separation      : pearson {pearson(nF, s):+.3f}"
                  f"   spearman {spearman(nF, s):+.3f}")

    ratio = (max(r["frobenius_norm"] for r in rows)
             / min(r["frobenius_norm"] for r in rows)) if rows else None
    print(f"\nlargest / smallest direction norm: {ratio:.1f}x")
    print("\nIf that ratio is well above 1, a FIXED steering coefficient applied")
    print("a different dose to each category, and the transfer test compared")
    print("interventions of different strengths. Unit-normalise the direction and")
    print("carry the dose in the coefficient before reading specificity.")

    out = Path(args.out or (run / "direction_norms.json"))
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
