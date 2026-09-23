"""Does the taxonomy replicate across models?

    python scripts/cross_model_analysis.py --runs runs/full_qwen18 runs/full_qwen14b ...

One model's result is a finding about that model. The research question is about
bias, so what matters is which parts survive changing the model.

Four things are compared, in increasing order of how much they would mean:

1. **Which categories reproduce.** If the same categories clear the extraction
   floor in every model, that is a property of the categories rather than of one
   network.

2. **The ranking of floors.** Even where the absolute floor moves with scale, the
   ORDER may be stable. Spearman between each pair of models.

3. **H2 within each model.** Does mean margin predict the floor everywhere, or
   was that one model's accident?

4. **The shape of the cosine matrix.** Directions from different models live in
   different spaces and cannot be compared directly — d_model differs and there
   is no shared basis. But the PATTERN can: if Age and Disability are opposed in
   two models and Religion and Physical_appearance are close in both, the
   structure replicates. Compared by correlating the off-diagonal entries of the
   two matrices over their shared categories (a Mantel-style test).

CPU only — reads finished run directories.
"""

import argparse
import json
import math
import sys
from itertools import combinations
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
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def load_run(p: Path):
    rp = p / "report.json"
    if not rp.exists():
        return None
    rep = json.loads(rp.read_text())
    floors, means = {}, {}
    for c, v in rep.get("categories", {}).items():
        if "extraction_floor" in v:
            floors[c] = v["extraction_floor"]["q05"]
        if "margins" in v:
            means[c] = v["margins"].get("mean")
    dirs = {}
    for f in sorted(p.glob("direction_*.npy")):
        dirs[f.stem.replace("direction_", "")] = np.load(f)
    return {"path": p, "model": rep.get("model"), "hf_id": rep.get("hf_id"),
            "method": rep.get("method", "extremes"), "d_model": rep.get("d_model"),
            "floors": floors, "mean_margin": means, "dirs": dirs,
            "p_value": rep.get("p_value"), "verdict": rep.get("verdict")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/_cross_model.json")
    args = ap.parse_args()

    runs = []
    for r in args.runs:
        d = load_run(Path(r))
        if d:
            runs.append(d)
        else:
            print(f"skipping {r} (no report.json)")
    if len(runs) < 2:
        print("need at least two finished runs")
        return 1

    tag = {r["path"].name: r for r in runs}
    print(f"{'run':<24}{'model':<12}{'method':<10}{'d_model':>8}{'usable':>8}   verdict")
    print("-" * 100)
    for r in runs:
        u = [c for c, f in r["floors"].items() if f >= bt.MIN_USABLE_FLOOR]
        print(f"{r['path'].name:<24}{str(r['model']):<12}{r['method']:<10}"
              f"{str(r['d_model']):>8}{len(u):>4}/{len(r['floors']):<3}   "
              f"{(r['verdict'] or '')[:44]}")

    # ---- 1. which categories reproduce, per run -------------------------
    all_cats = sorted({c for r in runs for c in r["floors"]})
    print(f"\n\n1. EXTRACTION FLOOR per category (bold = clears {bt.MIN_USABLE_FLOOR})\n")
    hdr = f"{'category':<24}" + "".join(f"{r['path'].name[:13]:>15}" for r in runs)
    print(hdr)
    print("-" * len(hdr))
    for c in all_cats:
        row = f"{c:<24}"
        for r in runs:
            v = r["floors"].get(c)
            cell = "-" if v is None else (f"{v:+.3f}*" if v >= bt.MIN_USABLE_FLOOR
                                          else f"{v:+.3f} ")
            row += f"{cell:>15}"
        print(row)
    print("\n* clears the usability threshold")

    # how consistent is the SET of reproducible categories?
    sets = {r["path"].name: {c for c, f in r["floors"].items()
                             if f >= bt.MIN_USABLE_FLOOR} for r in runs}
    print("\nagreement on WHICH categories reproduce (Jaccard):")
    for a, b in combinations(sets, 2):
        A, B = sets[a], sets[b]
        j = len(A & B) / len(A | B) if (A | B) else float("nan")
        print(f"  {a:<24}{b:<24}  {j:.2f}   shared={sorted(A & B)}")

    # ---- 2. floor RANKING stability -------------------------------------
    print("\n\n2. FLOOR RANKING agreement (spearman over shared categories)\n")
    for a, b in combinations(runs, 2):
        shared = sorted(set(a["floors"]) & set(b["floors"]))
        if len(shared) < 3:
            continue
        rho = spearman([a["floors"][c] for c in shared],
                       [b["floors"][c] for c in shared])
        print(f"  {a['path'].name:<24}{b['path'].name:<24}"
              f"rho={rho:+.3f}  (n={len(shared)})")

    # ---- 3. H2 per run ---------------------------------------------------
    print("\n\n3. H2 — does the model's TILT predict the floor, in each run?\n")
    print(f"{'run':<24}{'n cats':>8}{'pearson':>10}{'spearman':>10}")
    print("-" * 52)
    h2 = {}
    for r in runs:
        shared = [c for c in r["floors"] if r["mean_margin"].get(c) is not None]
        if len(shared) < 3:
            continue
        x = [r["mean_margin"][c] for c in shared]
        y = [r["floors"][c] for c in shared]
        h2[r["path"].name] = {"pearson": pearson(x, y), "spearman": spearman(x, y),
                              "n": len(shared)}
        print(f"{r['path'].name:<24}{len(shared):>8}{pearson(x, y):>+10.3f}"
              f"{spearman(x, y):>+10.3f}")

    # ---- 4. does the SHAPE of the cosine matrix replicate? ---------------
    print("\n\n4. STRUCTURE replication — correlation of off-diagonal cosines\n")
    print("   Directions from different models are not comparable directly")
    print("   (different d_model, no shared basis). The PATTERN of which")
    print("   categories sit near which can still replicate.\n")
    struct = []
    for a, b in combinations(runs, 2):
        # Only categories whose direction reproduces in BOTH runs. A cosine
        # involving a non-reproducing direction is noise, so including it makes
        # this a correlation between two sets of noise.
        joint = sorted({c for c in set(a["dirs"]) & set(b["dirs"])
                        if a["floors"].get(c, -1) >= bt.MIN_USABLE_FLOOR
                        and b["floors"].get(c, -1) >= bt.MIN_USABLE_FLOOR})
        if len(joint) < 3:
            print(f"  {a['path'].name:<22}{b['path'].name:<22}"
                  f"only {len(joint)} jointly-reproducible categories "
                  f"{joint} — structure not comparable")
            struct.append({"a": a["path"].name, "b": b["path"].name,
                           "jointly_reproducible": joint,
                           "pearson": None, "spearman": None,
                           "note": "fewer than 3 jointly reproducible categories"})
            continue
        shared = joint
        na, Ma = bt.cosine_matrix({c: a["dirs"][c] for c in shared})
        nb, Mb = bt.cosine_matrix({c: b["dirs"][c] for c in shared})
        assert na == nb
        off_a = [Ma[i][j] for i in range(len(na)) for j in range(i + 1, len(na))]
        off_b = [Mb[i][j] for i in range(len(nb)) for j in range(i + 1, len(nb))]
        rp, rs = pearson(off_a, off_b), spearman(off_a, off_b)
        struct.append({"a": a["path"].name, "b": b["path"].name,
                       "n_categories": len(shared), "n_pairs": len(off_a),
                       "pearson": rp, "spearman": rs})
        print(f"  {a['path'].name:<22}{b['path'].name:<22}"
              f"pearson={rp:+.3f}  spearman={rs:+.3f}  "
              f"({len(shared)} cats, {len(off_a)} pairs)")

    print("\n   A high correlation means the two models agree about which bias")
    print("   categories are close to which — the taxonomy replicates. Near zero")
    print("   means each model has its own idiosyncratic arrangement.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "runs": [{"name": r["path"].name, "model": r["model"],
                  "method": r["method"], "d_model": r["d_model"],
                  "floors": r["floors"], "mean_margin": r["mean_margin"],
                  "p_value": r["p_value"], "verdict": r["verdict"]} for r in runs],
        "reproducible_sets": {k: sorted(v) for k, v in sets.items()},
        "h2": h2, "structure_replication": struct,
    }, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
