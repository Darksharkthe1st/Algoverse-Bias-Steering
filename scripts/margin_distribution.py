"""Is a category's large margin sd broad inconsistency, or a few outliers?

    python scripts/margin_distribution.py --cache runs/_margins_cache --model qwen-14b

Gender_identity on qwen-14b has mean +1.076 (second highest of ten) but sd 5.497
(largest by far), and a floor of 0.081. That was read as "the model leans hard on
gender items and leans inconsistently" — but a large sd has two very different
causes and they are different findings:

  BROAD    the model genuinely disagrees with itself item to item; the tilt is
           real but unstable, and no single direction summarises it.
  OUTLIERS most items behave, a handful are extreme; the sd is a tail artifact
           and the bulk of the category may be perfectly consistent.

Reports, per category: mean, sd, median, IQR, robust sd (IQR/1.349), the
sd/robust-sd ratio, kurtosis, and what fraction of the total variance the top 5%
of items by |margin - median| account for.

CPU only — reads the cached per-item margins.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_margins_cache")
    ap.add_argument("--model", default="qwen-14b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = Path(args.cache)
    files = sorted(cache.glob(f"{args.model}_*.json"))
    if not files:
        print(f"no cached margins for {args.model} in {cache}")
        return 1

    rows = []
    for f in files:
        blob = json.loads(f.read_text())
        m = np.asarray(blob["margins"], dtype=np.float64)
        # filename: <model>_<Category>_<limit>_<seed>.json
        cat = f.stem[len(args.model) + 1:].rsplit("_", 2)[0]

        med = float(np.median(m))
        q1, q3 = np.percentile(m, [25, 75])
        iqr = float(q3 - q1)
        robust_sd = iqr / 1.349 if iqr > 0 else float("nan")
        sd = float(m.std(ddof=1))

        dev = np.abs(m - med)
        k = max(1, int(round(0.05 * len(m))))
        top_idx = np.argsort(dev)[-k:]
        ss_total = float(((m - m.mean()) ** 2).sum())
        ss_top = float(((m[top_idx] - m.mean()) ** 2).sum())

        mu = m.mean()
        kurt = float(((m - mu) ** 4).mean() / (m.var() ** 2)) if m.var() > 0 else float("nan")

        rows.append({
            "category": cat, "n": len(m), "mean": float(mu), "sd": sd,
            "median": med, "iqr": iqr, "robust_sd": robust_sd,
            "sd_over_robust_sd": sd / robust_sd if robust_sd == robust_sd and robust_sd > 0 else None,
            "excess_kurtosis": kurt - 3.0 if kurt == kurt else None,
            "top5pct_share_of_variance": ss_top / ss_total if ss_total > 0 else None,
            "frac_within_1_robust_sd": float(np.mean(dev <= robust_sd)) if robust_sd == robust_sd else None,
        })

    rows.sort(key=lambda r: -r["sd"])
    print(f"model: {args.model}\n")
    print(f"{'category':<22}{'n':>5}{'mean':>8}{'sd':>8}{'median':>8}{'robSD':>8}"
          f"{'sd/rob':>8}{'exKurt':>9}{'top5%var':>10}")
    print("-" * 86)
    for r in rows:
        sr = f"{r['sd_over_robust_sd']:.2f}" if r["sd_over_robust_sd"] else "-"
        ek = f"{r['excess_kurtosis']:+.1f}" if r["excess_kurtosis"] is not None else "-"
        tv = f"{r['top5pct_share_of_variance']:.0%}" if r["top5pct_share_of_variance"] else "-"
        print(f"{r['category']:<22}{r['n']:>5}{r['mean']:>+8.3f}{r['sd']:>8.3f}"
              f"{r['median']:>+8.3f}{r['robust_sd']:>8.3f}{sr:>8}{ek:>9}{tv:>10}")

    print("\nREADING IT")
    print("  sd/robSD ~ 1.0  : roughly normal spread -> BROAD inconsistency")
    print("  sd/robSD >> 1.5 : heavy tails -> the sd is driven by OUTLIERS")
    print("  excess kurtosis >> 0 and a large top-5% variance share point the")
    print("  same way: a few extreme items, not a broadly unstable category.")

    out = Path(args.out or f"runs/_margin_distribution_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
