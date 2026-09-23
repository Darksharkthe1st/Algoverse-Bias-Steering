"""Does a category's INTERNAL HETEROGENEITY explain whether it has a direction?

    python scripts/heterogeneity_vs_floor.py --run-dir runs/full_qwen14b

The 14B run split the ten BBQ categories cleanly in two, and the split is not the
one anybody would have predicted from the category names:

    reproducible : Disability_status .82, Age .75, Religion .69, Physical_app .65
    not          : Nationality .28, Sexual_orientation .16, Gender_identity .08,
                   Race_x_gender .07, Race_x_SES -.11, Race_ethnicity -.20

Every race-related category fails. Every category that succeeds has a narrow,
roughly binary contrast — old vs young, disabled vs not, one religion against
another. The failures each span many groups: Race_ethnicity covers a dozen racial
groups whose stereotypes differ from one another, Nationality covers dozens of
countries.

HYPOTHESIS
    A "bias direction" for a category only exists when the category names ONE
    contrast. When the label covers a mixture of unrelated stereotypes, the
    mean-difference over its items averages several different directions
    together and the result does not reproduce.

If that holds it is a real answer to the taxonomy question, and a more
interesting one than the surface grouping: what determines whether bias has a
direction is not the topic but whether the topic is internally coherent.

MEASURE
    Heterogeneity = the number of distinct stereotyped-group values the
    category's ambiguous items carry, and the entropy of their distribution.
    Correlate against the extraction floor. CPU only.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer.config import DatasetSpec, SampleSpec  # noqa: E402
from src.bias_steer.datasets import load_bbq, sample  # noqa: E402

BBQ_DIR = "datasets/BBQ_Prompt_Sets"


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
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return pearson(ranks(xs), ranks(ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = json.loads((Path(args.run_dir) / "report.json").read_text())
    floors = {c: v["extraction_floor"]["q05"]
              for c, v in report["categories"].items()
              if "extraction_floor" in v}

    margin_stats = {c: v.get("margins", {}) for c, v in report["categories"].items()}

    rows = []
    for cat in floors:
        exs = load_bbq(DatasetSpec(name="bbq", path=f"{BBQ_DIR}/{cat}.jsonl"))
        exs = sample(exs, SampleSpec(filter={"context_condition": ["ambig"]},
                                     limit=args.limit, seed=args.seed))
        groups = Counter()
        for e in exs:
            sg = e.metadata.get("stereotyped_groups") or []
            groups[tuple(sorted(str(g).lower() for g in sg))] += 1
        total = sum(groups.values()) or 1
        ent = -sum((c / total) * math.log2(c / total) for c in groups.values() if c)
        mstat = margin_stats.get(cat, {})
        mean_m = mstat.get("mean")
        sd_m = mstat.get("sd")
        rows.append({
            "category": cat, "floor": floors[cat],
            "n_distinct_stereotype_sets": len(groups),
            "entropy_bits": ent,
            "largest_share": max(groups.values()) / total,
            "mean_margin": mean_m,
            "sd_margin": sd_m,
            # How systematic is the tilt, relative to how much it varies? If the
            # model leans the same way on most items, there is one shared thing
            # to extract; if the mean is ~0 the lean is idiosyncratic per item.
            "margin_effect_size": (mean_m / sd_m) if (mean_m is not None and sd_m) else None,
        })

    rows.sort(key=lambda r: -r["floor"])
    print(f"{'category':<24}{'floor':>8}{'#grp':>6}{'entropy':>9}{'top%':>7}"
          f"{'mean_m':>9}{'sd_m':>8}{'mean/sd':>9}")
    print("-" * 80)
    for r in rows:
        mm = f"{r['mean_margin']:+.3f}" if r["mean_margin"] is not None else "-"
        sd = f"{r['sd_margin']:.3f}" if r["sd_margin"] else "-"
        es = f"{r['margin_effect_size']:+.3f}" if r["margin_effect_size"] is not None else "-"
        print(f"{r['category']:<24}{r['floor']:>8.3f}{r['n_distinct_stereotype_sets']:>6}"
              f"{r['entropy_bits']:>9.2f}{r['largest_share']:>6.0%}{mm:>9}{sd:>8}{es:>9}")

    f = [r["floor"] for r in rows]
    g = [r["n_distinct_stereotype_sets"] for r in rows]
    e = [r["entropy_bits"] for r in rows]
    s = [r["largest_share"] for r in rows]

    print(f"\ncorrelation of the extraction floor with, over {len(rows)} categories:")
    print(f"  H1  number of distinct stereotype sets : pearson {pearson(f, g):+.3f}"
          f"   spearman {spearman(f, g):+.3f}")
    print(f"  H1  entropy of that distribution       : pearson {pearson(f, e):+.3f}"
          f"   spearman {spearman(f, e):+.3f}")
    print(f"  H1  share of the most common set       : pearson {pearson(f, s):+.3f}"
          f"   spearman {spearman(f, s):+.3f}")

    have = [r for r in rows if r["mean_margin"] is not None and r["sd_margin"]]
    if len(have) >= 3:
        f2 = [r["floor"] for r in have]
        mm = [r["mean_margin"] for r in have]
        sd = [r["sd_margin"] for r in have]
        es = [r["margin_effect_size"] for r in have]
        print(f"\n  H2  MEAN margin (systematic tilt)      : pearson {pearson(f2, mm):+.3f}"
              f"   spearman {spearman(f2, mm):+.3f}")
        print(f"  H2  SD of the margin                   : pearson {pearson(f2, sd):+.3f}"
              f"   spearman {spearman(f2, sd):+.3f}")
        print(f"  H2  mean/sd (effect size)              : pearson {pearson(f2, es):+.3f}"
              f"   spearman {spearman(f2, es):+.3f}")

    print("\nH1 (heterogeneity): a category has a direction when it names ONE")
    print("   contrast rather than a mixture. Predicts negative r with #groups.")
    print("H2 (systematic tilt): a category has a direction when the model leans")
    print("   the SAME WAY across its items, so there is one shared thing to")
    print("   extract. A mean margin near zero means the lean is idiosyncratic")
    print("   per item and averaging finds nothing. Predicts positive r with mean.")
    print("\nn=10 categories either way, so this is suggestive, not confirmatory.")
    print("H2 is testable directly: split a HIGH-floor category into a subset with")
    print("a large mean margin and one with a near-zero mean, and check that only")
    print("the first reproduces.")

    out = Path(args.run_dir) / "heterogeneity.json"
    out.write_text(json.dumps({"rows": rows,
                               "pearson_floor_vs_ngroups": pearson(f, g),
                               "spearman_floor_vs_ngroups": spearman(f, g),
                               "pearson_floor_vs_entropy": pearson(f, e),
                               "spearman_floor_vs_entropy": spearman(f, e)}, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
