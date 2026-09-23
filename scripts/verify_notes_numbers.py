"""Re-derive every number quoted in notes/22 and notes/23 from the artifacts.

    python -m scripts.verify_notes_numbers

Exits non-zero if any number in the notes no longer matches the file it came
from. The repo rule is that numbers trace to artifacts and a writeup is not an
artifact; this makes that rule executable, so a number cannot silently drift
away from its source as the notes are edited.

It has already earned its keep once: it caught notes/22 section B quoting the
median column while labelling it "mean floor".
"""
import collections
import glob
import json
import math
import os
import statistics

import numpy as np

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs")
BBQ = os.path.join(ROOT, "repo", "datasets", "BBQ_Prompt_Sets")
fails = []


def ck(label, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) and isinstance(got, (int, float)) else got == want
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}: got {got!r}, doc says {want!r}")
    if not ok:
        fails.append(label)


# ---- load floors + cosines -------------------------------------------------
cells = []
for p in sorted(glob.glob(os.path.join(RUNS, "full_*", "report.json"))):
    d = json.load(open(p, encoding="utf-8"))
    for cat, v in (d.get("categories") or {}).items():
        fl = (v or {}).get("extraction_floor") or {}
        if "cosines" in fl:
            cells.append((d["model"], cat, fl["q05"], fl["cosines"]))

print("=== notes/22 §A — bootstrap / flip probabilities ===")
flips = {}
for m, c, q05, cos in cells:
    a = np.asarray(cos)
    rng = np.random.default_rng(0)
    dr = a[rng.integers(0, a.size, size=(20000, a.size))]
    q = np.quantile(dr, .05, axis=1)
    p = float((q < .5).mean() if q05 >= .5 else (q >= .5).mean())
    flips[(m, c)] = p
ck("yi-6b Disability_status flip prob", round(flips[("yi-6b", "Disability_status")], 2), 0.27, 0.01)
ck("qwen-7b Religion flip prob", round(flips[("qwen-7b", "Religion")], 2), 0.11, 0.01)
ck("cells with flip>5%", sum(1 for v in flips.values() if v > .05), 3)

print("\n=== notes/22 §B — floor vs n ===")
for run, model, want_lo, want_hi, want_d in (
        ("full_qwen14b", "qwen-14b", 0.654, 0.817, 0.163),
        ("full_qwen18", "qwen-1.8b", -0.063, -0.061, 0.002)):
    d = json.load(open(os.path.join(RUNS, run, "report.json"), encoding="utf-8"))
    res = d["floor_vs_n"]["result"]
    pts = sorted((int(k), v) for k, v in res.items())
    lo = statistics.mean(pts[0][1]["cosines"]); hi = statistics.mean(pts[-1][1]["cosines"])
    ck(f"{model} floor at n={pts[0][0]}", round(lo, 3), want_lo, 0.001)
    ck(f"{model} floor at n={pts[-1][0]}", round(hi, 3), want_hi, 0.001)
    ck(f"{model} delta", round(hi - lo, 3), want_d, 0.001)

print("\n=== notes/22 §C — power ===")
sds = [statistics.stdev(c) for _, _, _, c in cells if len(c) > 1]
ck("split-half sd median", round(statistics.median(sds), 3), 0.145, 0.001)
ck("split-half sd p90", round(sorted(sds)[int(.9 * len(sds))], 3), 0.211, 0.001)
ck("CI halfwidth at n=10, p90 sd", round(1.96 * sorted(sds)[int(.9 * len(sds))] / math.sqrt(10), 3), 0.131, 0.001)

print("\n=== notes/22 §D — abstention, within model ===")
ab = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(RUNS, "_margins_cache", "*.json"))):
    b = os.path.basename(p)[:-5]
    d = json.load(open(p, encoding="utf-8"))
    if not d.get("abstention"):
        continue
    m = b.split("_")[0]; cat = "_".join(b.split("_")[1:-2])
    ab[m].append((cat, float(np.mean([x > 0 for x in d["abstention"]]))))
fl_by = {(m, c): q for m, c, q, _ in cells}


def pear(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    n = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dd = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return n / dd if dd else float("nan")


want = {"gemma-2b": -0.422, "qwen-14b": -0.101, "qwen-7b": -0.178, "yi-6b": -0.536}
rs = []
for m, items in sorted(ab.items()):
    xy = [(f, fl_by[(m, c)]) for c, f in items if (m, c) in fl_by]
    r = pear([a for a, _ in xy], [b for _, b in xy])
    rs.append(r)
    ck(f"  {m} within-model r", round(r, 3), want[m], 0.001)
ck("mean within-model r", round(statistics.mean(rs), 3), -0.309, 0.001)
ck("all 4 negative", sum(1 for r in rs if r < 0), 4)

print("\n=== notes/22 §F — statistic invariance ===")
stats = {}
for name, fn in (("min", min), ("mean", statistics.mean)):
    stats[name] = [(m, c, fn(cos)) for m, c, _, cos in cells]
stats["q05"] = [(m, c, q) for m, c, q, _ in cells]
stats["median"] = [(m, c, statistics.median(cos)) for m, c, _, cos in cells]
race_max = {}
for name, vals in stats.items():
    race = [v for m, c, v in vals if "Race" in c]
    race_max[name] = max(race)
ck("race cells", len([1 for m, c, _, _ in cells if "Race" in c]), 15)
ck("max race floor over ALL statistics", round(max(race_max.values()), 3), 0.384, 0.001)
ck("no race cell clears 0.50 anywhere", max(race_max.values()) < 0.5, True)
mm = [(m, c) for m, c, q, cos in cells if (q >= .5) != (statistics.mean(cos) >= .5)]
ck("cells flipping q05 vs mean", len(mm), 3)

print("\n=== notes/22 §G — layer profiles ===")
dirs = []
for p in sorted(glob.glob(os.path.join(RUNS, "full_*", "report.json"))):
    d = json.load(open(p, encoding="utf-8")); rd = os.path.dirname(p)
    for cat, v in (d.get("categories") or {}).items():
        f = os.path.join(rd, f"direction_{cat}.npy")
        fl = (v or {}).get("extraction_floor") or {}
        if os.path.exists(f) and "q05" in fl:
            a = np.load(f); n = np.linalg.norm(a, axis=1)
            dirs.append((d["model"], cat, a, n, fl["q05"]))
ck("saved directions", len(dirs), 46)
peaks = [int(np.argmax(np.concatenate(([0.0], n[1:])))) == len(n) - 1 for _, _, _, n, _ in dirs]
ck("all peak at FINAL layer", all(peaks), True)
inc = [float(np.mean(np.diff(n[1:]) > 0)) for _, _, _, n, _ in dirs]
ck("monotonicity", round(statistics.mean(inc), 3), 0.975, 0.002)
ct = []
for _, _, _, n, _ in dirs:
    nn = n[1:]; idx = np.arange(1, len(n))
    ct.append(float((idx * nn).sum() / nn.sum() / (len(n) - 1)))
ck("mean norm-centroid depth", round(statistics.mean(ct), 3), 0.829, 0.001)

print("\n=== notes/23 — sub-group feasibility ===")


def groups_of(r):
    md = r.get("additional_metadata") or {}
    return tuple(sorted(str(x).strip() for x in (md.get("stereotyped_groups") or []) if str(x).strip()))


want_sub = {"Race_ethnicity": (9, 6, 1400), "Race_x_gender": (7, 5, 3480),
            "Race_x_SES": (4, 4, 2160), "Gender_identity": (5, 3, 1672),
            "Sexual_orientation": (5, 0, 144)}
for cat, (ng, n172, largest) in want_sub.items():
    rows = [json.loads(l) for l in open(os.path.join(BBQ, cat + ".jsonl"), encoding="utf-8")]
    amb = [r for r in rows if r["context_condition"] == "ambig"]
    cnt = collections.Counter(groups_of(r) for r in amb); cnt.pop((), None)
    ck(f"  {cat} groups", len(cnt), ng)
    ck(f"  {cat} groups>=172", sum(1 for v in cnt.values() if v >= 172), n172)
    ck(f"  {cat} largest sub-group", max(cnt.values()), largest)

print("\n=== paper/REVIEW.md P0c — steering ratios ===")
d = json.load(open(os.path.join(RUNS, "full_gemma2b", "transfer_test_norm_c2.json"), encoding="utf-8"))
cs = d["categories"]; eff = d["effects"]; ctrl = d["controls"]
diag = [abs(eff[c][c]) for c in cs]
rnd_all = [abs(v) for _k, row in ctrl.items() for v in row.values()]
rnd_diag = [abs(ctrl[f"random_{c}"][c]) for c in cs]
ck("paper's 7.0x (diag / all-random)", round(statistics.mean(diag) / statistics.mean(rnd_all), 2), 7.01, 0.01)
ck("like-for-like (diag / diag-random)", round(statistics.mean(diag) / statistics.mean(rnd_diag), 2), 5.85, 0.01)
off = statistics.mean([abs(eff[a][b]) for a in cs for b in cs if a != b])
ck("own vs other (1.28x)", round(statistics.mean(diag) / off, 2), 1.28, 0.005)
per_t = {b: statistics.mean([abs(eff[a][b]) for a in cs]) for b in cs}
ck("which-items ratio (4.4x)", round(max(per_t.values()) / min(per_t.values()), 2), 4.36, 0.01)


print()
print("=== paper/HOSTILE-REVIEW.md A1 -- the two-tier control argument ===")
want_a1 = {"Disability_status": {"gemma-2b": .818, "yi-6b": .605,
                                 "qwen-7b": .700, "qwen-14b": .820},
           "Physical_appearance": {"gemma-2b": .511, "yi-6b": .614,
                                   "qwen-7b": .785, "qwen-14b": .648}}
got_a1 = {}
for m, c, q, _ in cells:
    got_a1.setdefault(c, {})[m] = q
for cat, row in want_a1.items():
    for m, w in row.items():
        ck(f"  {cat[:12]}/{m}", round(got_a1[cat][m], 3), w, 0.001)
    ck(f"  {cat[:12]} models with a direction", len(got_a1[cat]), 4)

print()
print("=== paper/REVIEW.md P0d -- clustering was never computed for 3 models ===")
cm = json.load(open(os.path.join(RUNS, "_cross_model_final.json"), encoding="utf-8"))
for r in cm["runs"]:
    m = r["model"]
    if m not in ("gemma-2b", "yi-6b", "qwen-7b"):
        continue
    nrep = sum(1 for v in (r.get("floors") or {}).values() if v >= 0.5)
    ck(f"  {m} reproducible categories", nrep, 2)
    ck(f"  {m} p_value is None (not computed)", r.get("p_value") is None, True)
    ck(f"  {m} cluster_strength is None", r.get("cluster_strength") is None, True)
stale = [r["model"] for r in cm["runs"]
         if r.get("p_value") is None and "p=1.000" in str(r.get("verdict"))]
ck("models whose verdict STRING claims p=1.000 with no p_value", stale, ["gemma-2b"])

print()
print("FAILURES:", fails if fails else "none - every number re-derives")
sys.exit(1 if fails else 0)
