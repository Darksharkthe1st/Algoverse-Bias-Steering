"""Re-analyse a finished run using only the directions that reproduce.

    python scripts/reanalyze_run.py --run-dir runs/full_qwen14b

The original run clustered every category, including ones whose extraction floor
says the direction does not reproduce against itself. Those contribute noise to
the cosine matrix AND to the permutation null, so the p-value tests the wrong
thing. This re-reads the saved directions and floors and redoes the geometry on
the reproducible subset only.

CPU only — the directions are already on disk.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.bias_steer import bias_taxonomy as bt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--min-floor", type=float, default=bt.MIN_USABLE_FLOOR)
    args = ap.parse_args()

    run = Path(args.run_dir)
    report = json.loads((run / "report.json").read_text())
    d_model = report["d_model"]

    floors = {c: v["extraction_floor"]
              for c, v in report["categories"].items()
              if "extraction_floor" in v}
    dirs = {}
    for p in sorted(run.glob("direction_*.npy")):
        dirs[p.stem.replace("direction_", "")] = bt.assert_direction(np.load(p))

    print(f"run   : {run}")
    print(f"model : {report['model']}  ({report.get('hf_id')})")
    print(f"method: {report.get('method', 'extremes')}")
    print(f"random-direction floor 1/sqrt(d) = {bt.random_floor(d_model):.4f}\n")

    print(f"{'category':<24}{'n':>6}{'floor q05':>11}{'median':>9}   usable?")
    print("-" * 62)
    usable = []
    for c in sorted(floors, key=lambda k: -floors[k]["q05"]):
        f = floors[c]
        ok = f["q05"] >= args.min_floor
        if ok:
            usable.append(c)
        print(f"{c:<24}{f['n_items']:>6}{f['q05']:>11.3f}{f['median']:>9.3f}"
              f"   {'YES' if ok else 'no'}")

    print(f"\nreproducible: {len(usable)}/{len(floors)}  -> {usable}")
    if len(usable) < 2:
        print("\nfewer than 2 reproducible directions — no geometry is available.")
        return 0

    sub = {c: dirs[c] for c in usable if c in dirs}
    names, M = bt.cosine_matrix(sub)

    print("\ncosine matrix over the reproducible subset only:\n")
    print(" " * 24 + "".join(f"{n[:10]:>12}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<24}" + "".join(f"{M[i][j]:>12.3f}" for j in range(len(names))))

    print("\npair verdicts (both floors must clear the bar for a verdict):")
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = bt.pair_verdict(M[i][j], floors[names[i]], floors[names[j]])
            f = min(floors[names[i]]["q05"], floors[names[j]]["q05"])
            rows.append({"a": names[i], "b": names[j],
                         "cosine": float(M[i][j]), "floor": f, "verdict": v})
            print(f"  {names[i]:<22}{names[j]:<22} cos={M[i][j]:+.3f}  "
                  f"floor={f:.3f}  {v}")

    n_distinct = sum(1 for r in rows if r["verdict"] == "distinct")
    print(f"\n{n_distinct}/{len(rows)} pairs are distinguishable.")

    # How far below the floor does each pair sit? A cosine far below the floor is
    # a stronger separation claim than one that merely clears the margin.
    print("\nseparation = floor - cosine (larger means more clearly different):")
    for r in sorted(rows, key=lambda r: -(r["floor"] - r["cosine"])):
        print(f"  {r['a']:<22}{r['b']:<22}{r['floor'] - r['cosine']:>+8.3f}")

    Z = bt.cluster_topics(names, M)
    print(f"\ncluster strength (largest merge gap): {bt.cluster_strength(Z):.4f}")
    print("NOTE: no permutation null here — that needs the per-item residuals,")
    print("which are not saved. The run itself computes it.")

    out = run / "reanalysis.json"
    out.write_text(json.dumps({
        "min_floor": args.min_floor, "usable": usable,
        "names": names, "matrix": M.tolist(), "pairs": rows,
        "cluster_strength": bt.cluster_strength(Z),
    }, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
