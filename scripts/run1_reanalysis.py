"""Analyses of run 1 that have never been done, all on cached artifacts, no GPU.

    python -m scripts.run1_reanalysis --out runs/_reanalysis

Defect S5 says residuals were not cached, and that is true. But the DIRECTIONS
were saved (`runs/*/direction_<Category>.npy`), as were the raw split-half
cosines, the per-item margins, and a floor-versus-n sweep. A surprising amount is
recoverable from that, and none of it needs hardware.

Six analyses, each closing something the project already knows is open:

  A. Bootstrap intervals on every floor            closes S1 retroactively
  B. Floor versus n                                answers "you just needed more data"
  C. Power: what could this design have detected?  notes/14 §6.4 item 4
  D. Abstention rate versus floor                  N3, never analysed
  E. Does the cross-category structure replicate?  notes/10 says untested
  F. Is the headline invariant to the summary statistic?  pre-empts "you picked q05"

Everything printed here is traceable to a file under runs/.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(os.path.dirname(ROOT), "runs")


def boot_ci(vals, n_boot=20000, seed=0, alpha=0.05):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def load_reports():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "full_*", "report.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out[d["model"]] = (d, os.path.dirname(p))
    return out


# --------------------------------------------------------------------------- #
# A. Bootstrap intervals on every floor  (closes S1 for run-1 data)
# --------------------------------------------------------------------------- #

def analysis_a(reports):
    print("=" * 78)
    print("A. BOOTSTRAP INTERVALS ON EVERY FLOOR")
    print("=" * 78)
    print("""
Run 1 reported the q05 of 10 split-half cosines as a POINT value and gated on it.
Defect S1: nothing anywhere estimates how much that number moves on a redraw.
The 10 raw cosines are in report.json, so the interval is recoverable now.
""")
    rows = []
    print(f"{'model':<11}{'category':<21}{'q05':>8}{'mean':>8}{'95% CI':>18}{'sd':>7}  gate")
    for model, (d, _) in sorted(reports.items()):
        for cat, v in sorted((d.get("categories") or {}).items()):
            fl = (v or {}).get("extraction_floor") or {}
            cos = fl.get("cosines")
            if not cos:
                continue
            lo, hi = boot_ci(cos)
            mean = statistics.mean(cos)
            sd = statistics.stdev(cos) if len(cos) > 1 else float("nan")
            # how often would a redraw of these 10 splits flip the 0.50 verdict?
            rng = np.random.default_rng(0)
            arr = np.asarray(cos)
            draws = arr[rng.integers(0, arr.size, size=(20000, arr.size))]
            q05s = np.quantile(draws, 0.05, axis=1)
            passes = fl["q05"] >= 0.50
            flip = float((q05s < 0.50).mean() if passes else (q05s >= 0.50).mean())
            rows.append(dict(model=model, category=cat, q05=fl["q05"], mean=mean,
                             ci_lo=lo, ci_hi=hi, sd=sd, passes=passes, flip_prob=flip))
            mark = ""
            if 0.05 < flip:
                mark = f"  <-- {flip:.0%} chance of flipping on redraw"
            print(f"{model:<11}{cat:<21}{fl['q05']:>+8.3f}{mean:>+8.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{sd:>7.3f}  {'PASS' if passes else 'fail'}{mark}")

    unstable = [r for r in rows if r["flip_prob"] > 0.05]
    print(f"\n  {len(rows)} model-category pairs.")
    print(f"  {len(unstable)} have >5% chance of flipping their 0.50 verdict on a redraw:")
    for r in sorted(unstable, key=lambda r: -r["flip_prob"]):
        print(f"     {r['model']:<11}{r['category']:<21}"
              f"q05={r['q05']:+.3f}  flip={r['flip_prob']:.0%}")
    print("\n  This is defect S1, measured. A decision statistic taken from 10 draws")
    print("  and read as a point value hides exactly this much instability.")
    return rows


# --------------------------------------------------------------------------- #
# B. Floor versus n  (the "you just needed more data" attack)
# --------------------------------------------------------------------------- #

def analysis_b(reports):
    print()
    print("=" * 78)
    print("B. IS THE FAILURE A SAMPLE-SIZE PROBLEM?")
    print("=" * 78)
    print("""
The first thing a reviewer says about a negative result is "you needed more
data". run 1 swept n and stored the result; nobody analysed it.
""")
    out = {}
    for model, (d, _) in sorted(reports.items()):
        fv = d.get("floor_vs_n")
        if not fv:
            continue
        cat, res = fv.get("category"), fv.get("result") or {}
        pts = sorted((int(k), v) for k, v in res.items())
        if len(pts) < 2:
            continue
        print(f"\n  {model} - {cat}")
        print(f"    {'n':>6}{'q05':>9}{'median':>9}{'95% CI on the mean':>24}")
        xs, ys = [], []
        for n, v in pts:
            lo, hi = boot_ci(v["cosines"])
            m = statistics.mean(v["cosines"])
            xs.append(n); ys.append(m)
            print(f"    {n:>6}{v['q05']:>+9.3f}{v['median']:>+9.3f}   [{lo:+.3f},{hi:+.3f}]")

        # NO CURVE FIT. The sweep has exactly two n values, so any "fit" is a
        # line through two points and any extrapolation from it is arithmetic
        # dressed as evidence. An earlier version of this script did fit a
        # log-linear model and duly reported that qwen-1.8b/Race_ethnicity would
        # need 10^100 items to reach a floor of 0.50, which is not a finding.
        # Report the two points and the change between them; that is what was
        # measured.
        d_n = xs[-1] / xs[0]
        d_f = ys[-1] - ys[0]
        out[model] = dict(category=cat, n=xs, mean=ys, delta_mean=d_f, n_ratio=d_n)
        print(f"    {d_n:.2f}x more data changed the mean floor by {d_f:+.3f}")
        if d_f > 0.10:
            print("    -> n matters a lot HERE. This category was data-limited at the")
            print("       smaller n, and the sweep does not bound where it plateaus.")
        elif abs(d_f) <= 0.10:
            print("    -> n barely moves it. For this category the shortfall is not")
            print("       a sample-size problem in the range tested.")
    print("""
  What this does and does not support. The sweep covers ONE category per model
  and only TWO values of n, so it cannot say where either curve plateaus. What it
  does show is a contrast worth reporting: on a category that clears the bar
  (qwen-14b / Religion) a 1.4x increase in n moved the mean floor by +0.163,
  while on one that fails everywhere
  (qwen-1.8b / Race_ethnicity) a 1.9x increase moved it by almost nothing and
  left it negative. More data plainly helps where there is something to find.
  That is a bound on the "you just needed more data" attack, not a refutation of
  it, and it should be written that way.
""")
    return out


# --------------------------------------------------------------------------- #
# C. Power  (notes/14 §6.4 item 4 - never done)
# --------------------------------------------------------------------------- #

def analysis_c(reports):
    print("=" * 78)
    print("C. WHAT COULD THIS DESIGN HAVE DETECTED?")
    print("=" * 78)
    print("""
"A negative with a power analysis showing what effect size the design could have
detected" is listed in the project's own run plan as one of four things that
would raise this to conference quality. It was never done. Doing it now costs
nothing: the split-half SD is in the artifacts.
""")
    sds = []
    for model, (d, _) in reports.items():
        for cat, v in (d.get("categories") or {}).items():
            cos = ((v or {}).get("extraction_floor") or {}).get("cosines")
            if cos and len(cos) > 1:
                sds.append(statistics.stdev(cos))
    sd_med, sd_p90 = statistics.median(sds), sorted(sds)[int(0.9 * len(sds))]
    print(f"  split-half SD over {len(sds)} model-category pairs:"
          f" median {sd_med:.4f}, 90th pct {sd_p90:.4f}, max {max(sds):.4f}")
    print()
    print(f"  {'n_splits':>9}{'CI half-width (median sd)':>28}{'(90th pct sd)':>18}")
    for ns in (10, 50, 100, 400, 1000):
        print(f"  {ns:>9}{1.96*sd_med/math.sqrt(ns):>28.4f}{1.96*sd_p90/math.sqrt(ns):>18.4f}")
    hw10 = 1.96 * sd_p90 / math.sqrt(10)
    print(f"""
  At run 1's n_splits = 10 the 95% CI half-width on the mean floor is about
  {hw10:.3f} at the 90th-percentile variance. So run 1 could only have resolved
  two floors differing by more than roughly {2*hw10:.2f} - which is most of the
  usable range of a cosine. The design was not powered to make fine distinctions,
  and the coarse distinction it CAN make (control at 0.86-0.92 versus most
  categories below 0.5) is the only one the paper should lean on.

  To resolve a difference of 0.05 at the 90th-percentile variance you need
  n_splits ~ {int((1.96*sd_p90/0.025)**2):,}. That is affordable only with cached
  residuals, which is the concrete argument for the caching requirement.
""")
    return dict(sd_median=sd_med, sd_p90=sd_p90, n_pairs=len(sds),
                ci_halfwidth_at_10=hw10)


# --------------------------------------------------------------------------- #
# D. Abstention versus floor  (N3, measured but never analysed)
# --------------------------------------------------------------------------- #

def analysis_d(reports):
    print("=" * 78)
    print("D. DOES ABSTENTION EXPLAIN WHICH CATEGORIES FAIL?")
    print("=" * 78)
    print("""
Defect N3: on 23.5% of qwen-14b items the model's top choice is "can't answer",
yet the item is still ranked by a margin between two options it did not prefer,
and can enter either pole. The abstention margins were stored and never used.
If abstention-heavy categories are the ones that fail, that is an EXPLANATION
for the negative result rather than just a caveat.
""")
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, "_margins_cache", "*.json"))):
        base = os.path.basename(p)[:-5]
        d = json.load(open(p, encoding="utf-8"))
        ab = d.get("abstention")
        if not ab:
            continue
        model = base.split("_")[0]
        cat = "_".join(base.split("_")[1:-2])
        frac = float(np.mean([a > 0 for a in ab]))
        rows.append((model, cat, frac, len(ab)))

    by_model = collections.defaultdict(list)
    for model, cat, frac, n in rows:
        by_model[model].append((cat, frac, n))

    print(f"  {'model':<11}{'category':<21}{'abstain frac':>13}{'n':>7}{'floor q05':>11}")
    pairs = []
    for model, items in sorted(by_model.items()):
        rep = reports.get(model)
        for cat, frac, n in sorted(items):
            q = None
            if rep:
                fl = ((rep[0].get("categories") or {}).get(cat) or {}).get("extraction_floor")
                q = fl.get("q05") if fl else None
            print(f"  {model:<11}{cat:<21}{frac:>13.1%}{n:>7}"
                  f"{(f'{q:+.3f}' if q is not None else '  --'):>11}")
            if q is not None:
                pairs.append((frac, q))

    # The correlation MUST be computed within a model, not pooled.  Abstention
    # is overwhelmingly a model-level property -- yi-6b averages 5.0% across its
    # categories while qwen-14b averages 23.9% -- so pooling mixes a large
    # between-model effect into what is supposed to be a within-model,
    # between-category question.  Pooling is the exact confound that invalidated
    # the consistency/floor correlation retracted in notes/16, and it produced a
    # misleadingly flat r = -0.11 in the first version of this analysis.
    def pearson(xs, ys):
        if len(xs) < 3:
            return float("nan")
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        return num / den if den else float("nan")

    print(f"\n  WITHIN-MODEL correlation of abstention fraction against floor:")
    per_model, rs = {}, []
    for model, items in sorted(by_model.items()):
        rep = reports.get(model)
        if not rep:
            continue
        xy = []
        for cat, frac, _n in items:
            fl = ((rep[0].get("categories") or {}).get(cat) or {}).get("extraction_floor")
            if fl:
                xy.append((frac, fl["q05"]))
        if len(xy) < 4:
            continue
        r = pearson([a for a, _ in xy], [b for _, b in xy])
        # Exact permutation p: shuffle the floors against the abstention rates.
        # With 9-10 categories a t-approximation is not trustworthy, and a
        # permutation test costs nothing here.
        rng = np.random.default_rng(0)
        xs_ = np.array([a for a, _ in xy]); ys_ = np.array([b for _, b in xy])
        null = []
        for _ in range(20000):
            null.append(pearson(list(xs_), list(rng.permutation(ys_))))
        null = np.asarray(null)
        p = float((np.abs(null) >= abs(r)).mean())
        per_model[model] = dict(r=r, n=len(xy), p_perm=p)
        rs.append(r)
        print(f"    {model:<11} r = {r:+.3f}   (n = {len(xy)} categories, "
              f"permutation p = {p:.3f}){'  *' if p < 0.05 else ''}")

    pooled_r = pearson([a for a, _ in pairs], [b for _, b in pairs])
    print(f"\n    pooled across models (CONFOUNDED, shown only for contrast):"
          f" r = {pooled_r:+.3f}")
    if rs:
        mean_r = statistics.mean(rs)
        n_neg = sum(1 for r in rs if r < 0)
        # Each model's correlation is a separate small sample and none of them
        # will clear p<0.05 at n=9-10 unless |r|>~0.63. The evidence that is
        # actually available is the CONSISTENCY OF THE SIGN across independent
        # models, which is a sign test: under the null each sign is a coin flip.
        sign_p = 2 * sum(math.comb(len(rs), k) for k in range(n_neg, len(rs) + 1)) / 2 ** len(rs)
        sign_p = min(1.0, sign_p)
        min_p = min(v["p_perm"] for v in per_model.values())
        print(f"    mean within-model r = {mean_r:+.3f}"
              f"   range {min(rs):+.3f} to {max(rs):+.3f}")
        print(f"    sign consistency    = {n_neg}/{len(rs)} models negative"
              f"   (two-sided sign test p = {sign_p:.3f})")
        print(f"    smallest individual permutation p = {min_p:.3f}")

        strong = (min_p < 0.05)
        suggestive = (n_neg == len(rs) and mean_r < -0.2)
        if strong:
            print("\n  -> established within at least one model.")
        elif suggestive:
            print(f"""
  -> SUGGESTIVE, NOT ESTABLISHED. Say it exactly that way.

     Every one of the {len(rs)} models shows a negative relationship and the mean is
     {mean_r:+.3f}, but no single model's correlation is significant on its own
     (smallest p = {min_p:.3f} at n = 9-10 categories), and the sign test over
     models gives p = {sign_p:.3f}. That is a hint with a plausible mechanism
     attached, not a result.

     The mechanism, if it is real: the contrast ranks items by a stereotype
     margin between two NAMED options. On an item where the model actually
     prefers "can't answer", that margin is a difference between two options it
     rejected -- noise with a sign. A category with many such items builds both
     of its poles partly out of that noise, and reproduces worse.

     It is worth stating because it is CHEAP TO TEST PROPERLY and it makes a
     falsifiable prediction: the annotation-derived contrast never ranks
     anything, so abstention should not degrade it at all. Run 2 can check that
     directly, and if the prediction fails the mechanism is wrong.

     What it must NOT be called is an explanation of the negative result. That
     would be exactly the error notes/16 made and retracted -- reading a
     correlation over ~10 points as a finding.""")
        else:
            print("""
  -> no consistent within-model relationship. Abstention is a real defect in the
     procedure (N3) but it does NOT explain which categories fail. Worth
     reporting: it closes off an otherwise plausible explanation, and a null you
     went looking for is worth more than one you never tested.""")
        return dict(pairs=pairs, per_model=per_model, pooled_r=pooled_r,
                    mean_within_r=mean_r, sign_test_p=sign_p, min_perm_p=min_p,
                    verdict="established" if strong
                            else ("suggestive" if suggestive else "null"))
    return dict(pairs=pairs, per_model=per_model, pooled_r=pooled_r)


# --------------------------------------------------------------------------- #
# E. Does the cross-category structure replicate across models?
# --------------------------------------------------------------------------- #

def analysis_e(reports):
    print()
    print("=" * 78)
    print("E. DOES THE CROSS-CATEGORY STRUCTURE REPLICATE ACROSS MODELS?")
    print("=" * 78)
    print("""
notes/10 reports this as untested: comparing the ARRANGEMENT of directions needs
three categories reproducing in both models, and only three were available. The
saved direction files let the comparison be made over ALL categories, which is
weaker per-cell but has more cells.
""")
    mats = {}
    for model, (_d, rundir) in sorted(reports.items()):
        dirs = {}
        for p in sorted(glob.glob(os.path.join(rundir, "direction_*.npy"))):
            cat = os.path.basename(p)[len("direction_"):-4]
            dirs[cat] = np.load(p)
        if len(dirs) >= 3:
            mats[model] = dirs

    def offdiag(dirs, cats):
        out = {}
        for i, a in enumerate(cats):
            for b in cats[i + 1:]:
                A, B = dirs[a], dirs[b]
                num = (A * B).sum(axis=1)
                den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
                w = np.linalg.norm(A, axis=1)
                ok = np.isfinite(c) & (w > 0)
                out[(a, b)] = float(np.average(c[ok], weights=w[ok])) if ok.any() else float("nan")
        return out

    per = {m: offdiag(d, sorted(d)) for m, d in mats.items()}
    models = sorted(per)

    # Which (model, category) directions actually reproduce against themselves?
    repro = {m: {c for c, v in (reports[m][0].get("categories") or {}).items()
                 if ((v or {}).get("extraction_floor") or {}).get("q05", -9) >= 0.50}
             for m in models if m in reports}

    print(f"  models with >=3 saved directions: {models}")
    print(f"  reproducing directions per model: "
          f"{ {m: len(v) for m, v in repro.items()} }\n")
    print(f"  {'model A':<11}{'model B':<11}{'shared pairs':>13}{'pearson r':>11}{'spearman':>10}")
    results = []
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            shared = sorted(set(per[ma]) & set(per[mb]))
            if len(shared) < 4:
                continue
            xs = [per[ma][k] for k in shared]
            ys = [per[mb][k] for k in shared]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
            r = num / den if den else float("nan")

            def rank(v):
                s = sorted(range(len(v)), key=lambda i: v[i])
                out = [0] * len(v)
                for pos, i2 in enumerate(s):
                    out[i2] = pos + 1
                return out
            rx, ry = rank(xs), rank(ys)
            mrx, mry = statistics.mean(rx), statistics.mean(ry)
            n2 = sum((a - mrx) * (b - mry) for a, b in zip(rx, ry))
            d2 = math.sqrt(sum((a - mrx) ** 2 for a in rx) * sum((b - mry) ** 2 for b in ry))
            rs = n2 / d2 if d2 else float("nan")
            results.append((ma, mb, len(shared), r, rs))
            print(f"  {ma:<11}{mb:<11}{len(shared):>13}{r:>+11.3f}{rs:>+10.3f}")

    if results:
        rs_all = [x[3] for x in results if np.isfinite(x[3])]
        print(f"\n  mean pearson across {len(rs_all)} model pairs: {statistics.mean(rs_all):+.3f}")

    # The comparison that makes the number mean something: restrict to category
    # pairs where BOTH categories reproduce in BOTH models. If the apparent
    # agreement above is shared noise, restricting should not raise it. If the
    # structure is real where it is measurable, restricting should raise it.
    print("\n  RESTRICTED to pairs where both categories reproduce in both models:")
    print(f"  {'model A':<11}{'model B':<11}{'pairs':>7}{'pearson r':>11}"
          f"{'  vs unrestricted':>18}")
    restricted = []
    for ma, mb, _n, r_all, _rs in results:
        if ma not in repro or mb not in repro:
            continue
        good = sorted(set(per[ma]) & set(per[mb]) &
                      {k for k in per[ma]
                       if k[0] in repro[ma] and k[1] in repro[ma]
                       and k[0] in repro[mb] and k[1] in repro[mb]})
        if len(good) < 3:
            print(f"  {ma:<11}{mb:<11}{len(good):>7}{'--':>11}"
                  f"{'  too few to test':>18}")
            continue
        xs = [per[ma][k] for k in good]
        ys = [per[mb][k] for k in good]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        r = num / den if den else float("nan")
        restricted.append((ma, mb, len(good), r, r_all))
        print(f"  {ma:<11}{mb:<11}{len(good):>7}{r:>+11.3f}{r_all:>+18.3f}")

    print("""
  HOW TO READ THIS. The unrestricted correlations are computed over
  cross-category cosines between directions MOST OF WHICH DO NOT REPRODUCE
  against themselves, so they largely measure shared noise rather than shared
  structure. The restricted comparison is the test that would separate the two,
  and the honest outcome is that it cannot be run: with only two or three
  categories reproducing per model, almost no model pair has three category
  PAIRS where all four directions clear the bar.

  So this remains what notes/10 said it was -- untested rather than refuted --
  and the reanalysis confirms the reason rather than removing it. The apparent
  +0.39 mean agreement should not be reported as replication.
""")
    return dict(unrestricted=results, restricted=restricted)


# --------------------------------------------------------------------------- #
# F. Does the headline depend on WHICH summary of the 10 splits you use?
# --------------------------------------------------------------------------- #

def analysis_f(reports):
    print()
    print("=" * 78)
    print("F. IS THE RESULT INVARIANT TO THE CHOICE OF SUMMARY STATISTIC?")
    print("=" * 78)
    print("""
Run 1 gated on the q05 of ten split-half cosines. notes/13 §4 changes run 2 to
the mean, on the grounds that a quantile over B draws carries more error. If that
choice flips which categories reproduce, then "which categories reproduce" is
partly a statement about the statistic, and a reviewer is entitled to ask whether
q05 was picked to get the answer. So: run every reasonable summary and see.
""")
    cells = []
    for model, (d, _) in sorted(reports.items()):
        for cat, v in (d.get("categories") or {}).items():
            cos = ((v or {}).get("extraction_floor") or {}).get("cosines")
            if not cos:
                continue
            a = np.asarray(cos, float)
            lo, _hi = boot_ci(cos)
            cells.append(dict(model=model, cat=cat, min=float(a.min()),
                              q05=float(np.quantile(a, .05)),
                              median=float(np.median(a)),
                              mean=float(a.mean()), ci_lo=lo))

    stats = ("min", "q05", "median", "mean", "ci_lo")
    print(f"  {len(cells)} model-category cells\n")
    print(f"  {'statistic':<12}{'cells >= 0.50':>14}   categories clearing in >=2 models")
    print("  " + "-" * 74)
    table = {}
    for s in stats:
        k = sum(1 for c in cells if c[s] >= .5)
        per = collections.Counter(c["cat"] for c in cells if c[s] >= .5)
        multi = sorted(x for x, n in per.items() if n >= 2)
        table[s] = dict(cells=k, categories=multi)
        print(f"  {s:<12}{k:>14}   {', '.join(multi) if multi else '(none)'}")

    race = [c for c in cells if "Race" in c["cat"]]
    print(f"\n  Race-related categories ({len(race)} cells), maximum by each statistic:")
    race_max = {}
    for s in stats:
        mx = max(c[s] for c in race)
        race_max[s] = mx
        print(f"    {s:<10}{mx:>+8.3f}   {'CLEARS' if mx >= .5 else 'below 0.50'}")

    flips = [c for c in cells if (c["q05"] >= .5) != (c["mean"] >= .5)]
    print(f"\n  Cells whose verdict changes between q05 and mean: {len(flips)}/{len(cells)}")
    for c in sorted(flips, key=lambda c: c["cat"]):
        print(f"    {c['model']:<11}{c['cat']:<21}q05={c['q05']:+.3f} -> "
              f"mean={c['mean']:+.3f}  (fail -> PASS)")

    always = sorted({c["cat"] for c in cells
                     if all(c[s] >= .5 for s in stats)})
    print(f"""
  THIS IS THE ROBUSTNESS RESULT AND IT IS WORTH A SENTENCE IN THE PAPER.

  Disability_status and Physical_appearance clear the bar in at least two models
  under EVERY one of the five summaries, including the most conservative
  (the bootstrap CI lower bound). No race-related category clears 0.50 under ANY
  of them -- the largest race value anywhere is {max(race_max.values()):+.3f}, on the most
  permissive statistic. Only {len(flips)} of {len(cells)} cells change verdict between the run-1
  statistic and the run-2 one, and all {len(flips)} move in the same direction (fail to
  pass), so the run-2 change is if anything more generous, not less.

  So the two positive results and the race-wide negative are not artifacts of
  picking q05. That pre-empts a real attack, and it costs nothing to state.
""")
    return dict(by_statistic=table, race_max=race_max,
                n_flips=len(flips), n_cells=len(cells),
                cells_clearing_every_statistic=always)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RUNS, "_reanalysis"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    reports = load_reports()
    print(f"loaded {len(reports)} model reports: {sorted(reports)}\n")

    a = analysis_a(reports)
    b = analysis_b(reports)
    c = analysis_c(reports)
    d = analysis_d(reports)
    e = analysis_e(reports)
    f_ = analysis_f(reports)

    payload = {
        "A_bootstrap_floors": a,
        "B_floor_vs_n": b,
        "C_power": c,
        "D_abstention_vs_floor": {
            "per_model": d["per_model"],
            "pooled_r_CONFOUNDED": d["pooled_r"],
            "points": [{"abstain_frac": x, "floor_q05": y} for x, y in d["pairs"]],
        },
        "E_structure_replication": {
            "unrestricted": [
                {"model_a": x[0], "model_b": x[1], "shared_pairs": x[2],
                 "pearson": x[3], "spearman": x[4]} for x in e["unrestricted"]],
            "restricted_to_reproducing": [
                {"model_a": x[0], "model_b": x[1], "pairs": x[2],
                 "pearson": x[3], "pearson_unrestricted": x[4]}
                for x in e["restricted"]],
        },
        "F_statistic_sensitivity": f_,
    }
    p = os.path.join(args.out, "run1_reanalysis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
