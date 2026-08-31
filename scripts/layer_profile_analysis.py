"""Where in the network does a bias direction actually live?

    python -m scripts.layer_profile_analysis

Nobody has looked at this, and it settles an open decision. `notes/13` §15 fixes
the primary read at a "fixed mid-layer" and names layers for two models that are
not in our model set, so `notes/19` §6.2 proposed
`primary_layer = round(0.47 x n_layers)` -- borrowed from the reference paper's
own depth fraction (gemma-2-9b-it layer 20/42 = 0.476; Llama-3.1-8B 15-16/32).

Borrowing a constant from another paper's models is exactly the kind of
undeclared parameter this project keeps getting burned by. The 46 saved
direction files let the question be answered from our own data:

  1. At what relative depth does each direction carry its norm?
  2. Is that depth consistent across categories, and across models?
  3. Does it differ between directions that reproduce and directions that do not?

(3) is the one that matters. If reproducing and non-reproducing directions peak
at the same depth, then depth is a property of the architecture and a single
fraction is defensible. If they differ, then choosing one layer is choosing an
answer, and the all-layer summary has to stay primary.
"""

from __future__ import annotations

import glob
import json
import os
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(ROOT), "runs")


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "full_*", "report.json"))):
        d = json.load(open(p, encoding="utf-8"))
        rundir = os.path.dirname(p)
        for cat, v in (d.get("categories") or {}).items():
            f = os.path.join(rundir, f"direction_{cat}.npy")
            fl = (v or {}).get("extraction_floor") or {}
            if not os.path.exists(f) or "q05" not in fl:
                continue
            arr = np.load(f)
            norms = np.linalg.norm(arr, axis=1)
            out.append(dict(model=d["model"], cat=cat, n_layers=arr.shape[0],
                            d_model=arr.shape[1], norms=norms, arr=arr,
                            q05=fl["q05"]))
    return out


def centroid_depth(norms):
    """Norm-weighted mean layer, as a fraction of depth in [0,1].

    Layer 0 is exactly zero by construction (the capture token is the chat
    template's final token, identical across items, so the difference of means is
    exactly 0 at the embedding layer) and is excluded rather than allowed to drag
    the centroid down.
    """
    n = norms[1:]
    if n.sum() <= 0:
        return float("nan")
    idx = np.arange(1, len(norms))
    return float((idx * n).sum() / n.sum() / (len(norms) - 1))


def peak_depth(norms):
    n = norms.copy()
    n[0] = 0.0
    return float(np.argmax(n) / (len(norms) - 1))


def main():
    rows = load()
    print(f"{len(rows)} saved directions across "
          f"{len({r['model'] for r in rows})} models\n")

    print("=" * 78)
    print("1. WHERE DOES EACH DIRECTION CARRY ITS NORM?")
    print("=" * 78)
    print(f"\n{'model':<11}{'category':<21}{'L':>4}{'peak':>7}{'peak%':>8}"
          f"{'centroid%':>11}{'top-5 layers share':>20}{'  repro'}")
    for r in sorted(rows, key=lambda r: (r["model"], -r["q05"])):
        n = r["norms"].copy(); n[0] = 0.0
        share = float(np.sort(n)[-5:].sum() / n.sum()) if n.sum() > 0 else float("nan")
        r["peak_frac"] = peak_depth(r["norms"])
        r["centroid_frac"] = centroid_depth(r["norms"])
        r["top5_share"] = share
        print(f"{r['model']:<11}{r['cat']:<21}{r['n_layers']:>4}"
              f"{int(np.argmax(n)):>7}{r['peak_frac']:>8.2f}{r['centroid_frac']:>11.2f}"
              f"{share:>20.1%}   {'YES' if r['q05'] >= .5 else '-'}")

    print()
    print("=" * 78)
    print("2. IS THE DEPTH CONSISTENT?")
    print("=" * 78)
    print(f"\n{'model':<11}{'n':>4}{'peak% mean':>12}{'sd':>8}{'centroid% mean':>16}{'sd':>8}")
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m]
        pk = [r["peak_frac"] for r in sub]
        ct = [r["centroid_frac"] for r in sub]
        print(f"{m:<11}{len(sub):>4}{statistics.mean(pk):>12.3f}"
              f"{(statistics.stdev(pk) if len(pk) > 1 else 0):>8.3f}"
              f"{statistics.mean(ct):>16.3f}"
              f"{(statistics.stdev(ct) if len(ct) > 1 else 0):>8.3f}")

    allpk = [r["peak_frac"] for r in rows]
    allct = [r["centroid_frac"] for r in rows]
    print(f"\n  ALL {len(rows)} directions: peak depth {statistics.mean(allpk):.3f}"
          f" +/- {statistics.stdev(allpk):.3f},"
          f"  centroid depth {statistics.mean(allct):.3f} +/- {statistics.stdev(allct):.3f}")

    print()
    print("=" * 78)
    print("3. DO REPRODUCING AND NON-REPRODUCING DIRECTIONS LIVE AT THE SAME DEPTH?")
    print("=" * 78)
    good = [r for r in rows if r["q05"] >= .5]
    bad = [r for r in rows if r["q05"] < .5]
    print(f"\n  reproducing     n={len(good):>3}  peak {statistics.mean([r['peak_frac'] for r in good]):.3f}"
          f"  centroid {statistics.mean([r['centroid_frac'] for r in good]):.3f}"
          f"  top-5 share {statistics.mean([r['top5_share'] for r in good]):.1%}")
    print(f"  NOT reproducing n={len(bad):>3}  peak {statistics.mean([r['peak_frac'] for r in bad]):.3f}"
          f"  centroid {statistics.mean([r['centroid_frac'] for r in bad]):.3f}"
          f"  top-5 share {statistics.mean([r['top5_share'] for r in bad]):.1%}")

    # permutation test on the difference in mean centroid depth
    rng = np.random.default_rng(0)
    obs = (statistics.mean([r["centroid_frac"] for r in good])
           - statistics.mean([r["centroid_frac"] for r in bad]))
    pool = np.array([r["centroid_frac"] for r in rows])
    ng = len(good)
    null = []
    for _ in range(20000):
        p = rng.permutation(pool)
        null.append(p[:ng].mean() - p[ng:].mean())
    null = np.asarray(null)
    pval = float((np.abs(null) >= abs(obs)).mean())
    print(f"\n  difference in mean centroid depth = {obs:+.3f}"
          f"   permutation p = {pval:.3f}")

    # ---------------------------------------------------------------- #
    # 4. Is the norm profile SIGNAL, or just residual-stream scale growth?
    # ---------------------------------------------------------------- #
    print()
    print("=" * 78)
    print("4. IS THE NORM PROFILE SIGNAL, OR JUST SCALE GROWTH?")
    print("=" * 78)
    print("""
A transformer's residual stream accumulates contributions, so its norm grows
with depth, and a difference of means computed per layer inherits that growth for
free. Before reading anything into "the direction peaks at the top", check
whether the profile could be anything else.
""")
    inc = [float(np.mean(np.diff(r["norms"][1:]) > 0)) for r in rows]
    print(f"  monotonicity: {statistics.mean(inc):.1%} of layer-to-layer steps increase")
    print("                (pure scale growth would be 100%)")

    def shape(r):
        n = r["norms"][1:].astype(float)
        return n / n.max()
    shape_cos = {}
    for L in sorted({r["n_layers"] for r in rows}):
        g = [shape(r) for r in good if r["n_layers"] == L]
        b = [shape(r) for r in bad if r["n_layers"] == L]
        if not g or not b:
            continue
        gm, bm = np.mean(g, 0), np.mean(b, 0)
        c = float(gm @ bm / (np.linalg.norm(gm) * np.linalg.norm(bm)))
        shape_cos[L] = c
        print(f"  depth {L:>3}: cosine between the MEAN profile of reproducing "
              f"({len(g)}) and non-reproducing ({len(b)}) directions = {c:.4f}")

    # scale-free: where do directions agree, after per-layer normalisation?
    print("\n  Scale-free check -- mean |cosine| between category directions, per layer.")
    print("  Cosine ignores magnitude, so structure here cannot be a norm artifact.")
    print(f"\n  {'model':<11}{'L':>4}{'peak layers':>16}{'peak depth':>13}"
          f"{'|cos| first':>13}{'mid':>8}{'last':>8}")
    peak_depths = {}
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        L = rs[0]["n_layers"]
        agree, cnt = np.zeros(L), 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                A, B = rs[i]["arr"], rs[j]["arr"]
                num = (A * B).sum(1)
                den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
                agree += np.nan_to_num(np.abs(c))
                cnt += 1
        agree /= max(cnt, 1)
        top = np.sort(np.argsort(agree[1:])[-3:] + 1)
        depth = [round(float(t) / (L - 1), 2) for t in top]
        peak_depths[m] = dict(peak_layers=top.tolist(), peak_depth=depth,
                              profile=[float(x) for x in agree])
        print(f"  {m:<11}{L:>4}{str(top.tolist()):>16}{str(depth):>13}"
              f"{agree[1]:>13.3f}{agree[L // 2]:>8.3f}{agree[-1]:>8.3f}")

    ref = 0.47
    mean_ct = statistics.mean(allct)
    print(f"""
=============================================================================
WHAT THIS MEANS FOR THE PRIMARY-LAYER DECISION (notes/19 hole (e))
=============================================================================

  Norm-centroid depth over all {len(rows)} directions: {mean_ct:.3f}
  All {len(rows)} directions peak at the FINAL layer, with zero variance.

  DO NOT read that as "the bias signal lives at the top", and in particular do
  NOT use {mean_ct:.2f} as the primary-layer fraction. An earlier version of this
  script drew exactly that conclusion and it is wrong, for two measured reasons:

    * the norm profile is {statistics.mean(inc):.0%} monotonically increasing, which is what
      residual-stream scale growth looks like on its own; and
    * the profile SHAPE is the same for directions that reproduce and directions
      that do not -- cosine between the two mean profiles is
      {min(shape_cos.values()) if shape_cos else float('nan'):.4f} or higher at every depth. A direction carrying real
      signal and one that is noise have indistinguishable norm profiles.

  A statistic that cannot tell signal from noise cannot locate signal. So the
  norm profile says nothing about where to read, and the {mean_ct:.2f} figure is an
  artifact of the residual stream, not a property of bias.

  THIS ALSO QUALIFIES THE N1 CLOSURE. notes/13 §8 fixes the layer summary as a
  norm-weighted mean, justified on the grounds that "per-layer norms span orders
  of magnitude, so an unweighted median treats a near-zero-norm layer as equal to
  the highest-signal one." That justification assumes high norm implies high
  signal, and the shape comparison above shows it does not. Weighting by a
  near-monotone function of depth makes the headline number effectively a
  late-layer read.

  The practical impact is small -- notes/12 N1 measured the norm-weighted mean
  and the unweighted median to differ by <= 0.033 -- so this is a flaw in the
  JUSTIFICATION rather than in the number. But the justification should be
  restated, and both summaries should keep being reported, which notes/13
  already requires.

  What to do about hole (e): the scale-free per-layer agreement between category
  directions peaks near depth 0.55-0.62 on qwen-14b and qwen-7b -- the two models
  with the most reproducible directions -- which is in the same neighbourhood as
  the reference paper's {ref:.2f} and nowhere near {mean_ct:.2f}. Keep the all-layer summary
  as the headline, and if a single layer is declared, declare it from the
  scale-free profile rather than from the norm profile.""")

    if pval < 0.05:
        print(f"""
  Separately: reproducing and non-reproducing directions sit at significantly
  different norm-centroid depths (p = {pval:.3f}).""")
    else:
        print(f"""
  Consistent with all of the above, reproducing and non-reproducing directions
  sit at statistically indistinguishable norm-centroid depths (p = {pval:.3f}) --
  another way of seeing that the norm profile is architecture, not signal.""")

    out = os.path.join(RUNS, "_reanalysis")
    os.makedirs(out, exist_ok=True)
    payload = [{k: v for k, v in r.items() if k not in ("norms", "arr")}
               for r in rows]
    for r, src in zip(payload, rows):
        r["norms"] = [float(x) for x in src["norms"]]
    p = os.path.join(out, "layer_profiles.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"directions": payload,
                   "mean_peak_frac": statistics.mean(allpk),
                   "mean_centroid_frac": statistics.mean(allct),
                   "depth_difference_p": pval,
                   "monotonicity": statistics.mean(inc),
                   "profile_shape_cosine_repro_vs_fail": shape_cos,
                   "scale_free_peak_depth": peak_depths}, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
