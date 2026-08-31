#!/usr/bin/env python3
"""Recount every number in the InterpScience submission from committed artifacts.

The paper ("The Extraction Floor: Measurement Validity for Linear Social-Bias
Directions in Language Models") quotes no number that this script cannot
re-derive from `runs/` alone. Run it from the repo root on the branch that
carries the run artifacts (`jz/bias-taxonomy` lineage):

    python3 scripts/recount_taxonomy_paper.py

Sections mirror the paper: T1 topic-identity control, T2 extremes floors,
F1 alpha sweep, probe recovery + clustering, pair separations, predictors,
T3 transfer. Exit code is 0 only if every hard-coded paper value matches its
recount within tolerance, so CI can hold the paper to the artifacts.

CPU-only, numpy + stdlib. No torch, no model, no network.
"""
import glob
import json
import os
import sys

import numpy as np

FAILURES = []


def check(label, got, expected, tol=0.0015):
    ok = abs(got - expected) <= tol
    if not ok:
        FAILURES.append(f"{label}: recount {got:.4f} != paper {expected:.4f}")
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label}: {got:.4f} (paper {expected:.4f})")


def load(p):
    with open(p) as f:
        return json.load(f)


def floors(rep):
    out = {}
    for c, d in rep["categories"].items():
        ef = d.get("extraction_floor")
        out[c] = ef["q05"] if ef else None
    return out


EXTREMES_RUNS = {
    "qwen-1.8b": "full_qwen18",
    "qwen-7b": "full_qwen7b",
    "qwen-14b": "full_qwen14b",
    "gemma-2b": "full_gemma2b",
    "yi-6b": "full_yi6b",
}

# Table 2 of the paper, extremes floors (q05). None = dropped by task control.
PAPER_T2 = {
    "qwen-7b": {"Disability_status": 0.700, "Physical_appearance": 0.785,
                "Age": 0.194, "Religion": 0.316, "Nationality": -0.020,
                "Race_x_gender": 0.218, "Sexual_orientation": -0.104,
                "Gender_identity": -0.177, "Race_x_SES": -0.150,
                "Race_ethnicity": -0.214},
    "qwen-14b": {"Disability_status": 0.820, "Physical_appearance": 0.648,
                 "Age": 0.754, "Religion": 0.686, "Nationality": 0.279,
                 "Race_x_gender": 0.072, "Sexual_orientation": 0.161,
                 "Gender_identity": 0.081, "Race_x_SES": -0.105,
                 "Race_ethnicity": -0.204},
    "gemma-2b": {"Disability_status": 0.818, "Physical_appearance": 0.511,
                 "Age": 0.161, "Religion": -0.163, "Nationality": -0.038,
                 "Race_x_gender": -0.044, "Sexual_orientation": 0.056,
                 "Gender_identity": None, "Race_x_SES": -0.165,
                 "Race_ethnicity": -0.013},
    "yi-6b": {"Disability_status": 0.605, "Physical_appearance": 0.614,
              "Age": 0.177, "Religion": -0.454, "Nationality": 0.296,
              "Race_x_gender": 0.204, "Sexual_orientation": -0.304,
              "Gender_identity": None, "Race_x_SES": 0.089,
              "Race_ethnicity": -0.230},
    "qwen-1.8b": {"Disability_status": None, "Physical_appearance": None,
                  "Age": 0.423, "Religion": -0.008, "Nationality": 0.057,
                  "Race_x_gender": 0.138, "Sexual_orientation": -0.202,
                  "Gender_identity": -0.072, "Race_x_SES": 0.017,
                  "Race_ethnicity": -0.115},
}

print("== T2: extremes extraction floors ==")
for model, run in EXTREMES_RUNS.items():
    rep = load(f"runs/{run}/report.json")
    fl = floors(rep)
    for cat, exp in PAPER_T2[model].items():
        got = fl.get(cat)
        if exp is None or got is None:
            if (exp is None) != (got is None):
                FAILURES.append(f"{model}/{cat}: dropped-status disagrees")
            continue
        check(f"{model}/{cat}", got, exp)

print("\n== denominators (direction files per model) ==")
for model, run, n_exp in [("qwen-1.8b", "full_qwen18", 8), ("qwen-7b", "full_qwen7b", 10),
                          ("qwen-14b", "full_qwen14b", 10), ("gemma-2b", "full_gemma2b", 9),
                          ("yi-6b", "full_yi6b", 9)]:
    n = len(glob.glob(f"runs/{run}/direction_*.npy"))
    print(f"  [{'ok' if n == n_exp else 'MISMATCH'}] {model}: {n} (paper {n_exp})")
    if n != n_exp:
        FAILURES.append(f"{model}: {n} direction files, paper says {n_exp}")

print("\n== T1: topic-identity control ==")
CONTROLS = [("runs/_extraction_positive_control.json", "qwen-1.8b"),
            ("runs/_extraction_control_gemma2b.json", "gemma-2b"),
            ("runs/_extraction_control_yi6b.json", "yi-6b")]
all_med, all_q05 = [], []
for path, model in CONTROLS:
    d = load(path)
    for pair, v in d["pairs"].items():
        cos = v["floor"]["cosines"]
        med, q05 = float(np.median(cos)), float(np.quantile(cos, 0.05))
        all_med.append(med)
        all_q05.append(q05)
        print(f"  {model:10s} {pair:38s} median={med:.3f} q05={q05:.3f}")
check("control median low bound", min(all_med), 0.882, tol=0.002)
check("control median high bound", max(all_med), 0.926, tol=0.002)
check("control worst q05", min(all_q05), 0.860, tol=0.002)

print("\n== probe recovery + clustering ==")
rep14 = load("runs/probe_tuned_qwen14b/report.json")
rep7 = load("runs/probe_tuned_qwen7b_a1e6/report.json")
check("qwen-14b clustering p", rep14["p_value"], 0.030, tol=0.0005)
check("qwen-14b cluster strength", rep14["cluster_strength"], 0.096, tol=0.0005)
check("qwen-14b null median", rep14["permutation_null"]["median"], 0.047, tol=0.0005)
check("qwen-14b null q95", rep14["permutation_null"]["q95"], 0.085, tol=0.0005)
check("qwen-7b clustering p", rep7["p_value"], 0.005, tol=0.0005)
check("qwen-7b cluster strength", rep7["cluster_strength"], 0.335, tol=0.0005)
check("qwen-7b null median", rep7["permutation_null"]["median"], 0.057, tol=0.0005)
check("qwen-7b null q95", rep7["permutation_null"]["q95"], 0.138, tol=0.0005)
for rep, model, n_exp in [(rep14, "qwen-14b", 5), (rep7, "qwen-7b", 3)]:
    usable = [c for c, v in floors(rep).items() if v is not None and v >= 0.50]
    print(f"  [{'ok' if len(usable) == n_exp else 'MISMATCH'}] {model} probe usable: "
          f"{len(usable)} (paper {n_exp}): {sorted(usable)}")
    if len(usable) != n_exp:
        FAILURES.append(f"{model} probe usable count {len(usable)} != {n_exp}")

print("\n== extremes qwen-14b: pair separations among usable ==")
REPRO14 = {"Age", "Disability_status", "Physical_appearance", "Religion"}
rep = load("runs/full_qwen14b/report.json")
seps = [p["floor_q05"] - p["cosine"] for p in rep["pairs"]
        if p["a"] in REPRO14 and p["b"] in REPRO14]
dist = [p["distinguishable"] for p in rep["pairs"]
        if p["a"] in REPRO14 and p["b"] in REPRO14]
check("min pair separation", min(seps), 0.34, tol=0.005)
check("max pair separation", max(seps), 1.03, tol=0.005)
print(f"  [{'ok' if all(dist) and len(dist) == 6 else 'MISMATCH'}] all 6 pairs distinguishable")
if not (all(dist) and len(dist) == 6):
    FAILURES.append("qwen-14b usable pairs not all distinguishable")

print("\n== alpha sweep: race negative ==")
best = -9.0
for f in glob.glob("runs/_probe_alpha_sweep_*.json"):
    d = load(f)
    for a, v in d["categories"].get("Race_ethnicity", {}).items():
        if isinstance(v, dict):
            best = max(best, v.get("q05", -9.0))
check("Race_ethnicity best q05 anywhere", best, 0.146, tol=0.002)
d14 = load("runs/_probe_alpha_sweep_qwen-14b.json")
best14 = max(v["q05"] for v in d14["categories"]["Race_ethnicity"].values())
check("Race_ethnicity best q05 qwen-14b", best14, 0.009, tol=0.002)

print("\n== predictors: r(floor, direction norm), r(floor, |mean margin|) ==")
PAPER_R_NORM = {"qwen-1.8b": 0.81, "qwen-7b": 0.90, "qwen-14b": 0.91,
                "gemma-2b": 0.95, "yi-6b": 0.76}
r_tilts = []
for model, run in EXTREMES_RUNS.items():
    rep = load(f"runs/{run}/report.json")
    fs, ts, ns = [], [], []
    for c, d in rep["categories"].items():
        ef = d.get("extraction_floor")
        if not ef:
            continue
        npy = f"runs/{run}/direction_{c}.npy"
        if not os.path.exists(npy):
            continue
        fs.append(ef["q05"])
        ts.append(abs(d["margins"]["mean"]))
        ns.append(float(np.linalg.norm(np.load(npy))))
    r_norm = float(np.corrcoef(fs, ns)[0, 1])
    r_tilt = float(np.corrcoef(fs, ts)[0, 1])
    r_tilts.append(r_tilt)
    check(f"{model} r(floor,norm)", r_norm, PAPER_R_NORM[model], tol=0.005)
check("r(floor,|tilt|) low bound", min(r_tilts), 0.59, tol=0.005)
check("r(floor,|tilt|) high bound", max(r_tilts), 0.82, tol=0.005)

print("\n== T3: norm-matched transfer ==")
def transfer_row(run, reproducible, fname):
    d = load(f"runs/{run}/{fname}")
    own, cross = [], []
    for src in d["effects"]:
        if src not in reproducible:
            continue
        for tgt, eff in d["effects"][src].items():
            if tgt not in reproducible:
                continue
            (own if src == tgt else cross).append(eff)
    rands = []
    for k, v in d.get("controls", {}).items():
        if isinstance(v, dict):
            vals = [x for x in v.values() if isinstance(x, (int, float))]
            if vals:
                rands.append(float(np.mean(vals)))
    rand_extreme = max(rands, key=abs) if rands else float("nan")
    return float(np.mean(own)), float(np.mean(cross)), rand_extreme

PAPER_T3 = [
    ("full_qwen14b", REPRO14, "transfer_test_norm_c2.json", 0.000, 0.004, -0.005),
    ("full_qwen14b", REPRO14, "transfer_test_norm_c4.json", 0.002, 0.009, -0.010),
    ("full_qwen14b", REPRO14, "transfer_test_norm_c8.json", 0.004, 0.019, -0.019),
    ("full_qwen14b", REPRO14, "transfer_test_norm_c16.json", 0.007, 0.033, -0.036),
    ("full_gemma2b", {"Disability_status", "Physical_appearance"},
     "transfer_test_norm_c2.json", -0.081, -0.063, 0.013),
    ("full_gemma2b", {"Disability_status", "Physical_appearance"},
     "transfer_test_norm_c5.json", -0.221, -0.171, 0.032),
    ("full_gemma2b", {"Disability_status", "Physical_appearance"},
     "transfer_test_norm_c10.json", -0.433, -0.361, 0.063),
]
for run, repro, fname, e_own, e_cross, e_rand in PAPER_T3:
    own, cross, rand = transfer_row(run, repro, fname)
    tag = f"{run}/{fname}"
    check(f"{tag} own", own, e_own)
    check(f"{tag} cross", cross, e_cross)
    check(f"{tag} random", rand, e_rand)

print("\n== alpha instability (limitations section) ==")
sims = []
for c in sorted(load("runs/probe_qwen14b/report.json")["categories"]):
    a1 = f"runs/probe_qwen14b/direction_{c}.npy"
    a6 = f"runs/probe_tuned_qwen14b/direction_{c}.npy"
    if os.path.exists(a1) and os.path.exists(a6):
        A, B = np.load(a1), np.load(a6)
        num = (A * B).sum(axis=1)
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        sims.append(float(np.median(per[np.isfinite(per)])))
check("alpha self-similarity low", min(sims), 0.10, tol=0.02)
check("alpha self-similarity high", max(sims), 0.21, tol=0.02)

print()
if FAILURES:
    print(f"RECOUNT FAILED: {len(FAILURES)} mismatches")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("RECOUNT CLEAN: every checked paper number reproduces from runs/ artifacts.")

print("\n== S1/S2 disclosures (added 2026-08-31) ==")
rep14p = load("runs/probe_tuned_qwen14b/report.json")
cos = np.array(rep14p["categories"]["Nationality"]["extraction_floor"]["cosines"])
rng = np.random.default_rng(0)
fails = sum(np.quantile(rng.choice(cos, size=len(cos), replace=True), 0.05) < 0.50
            for _ in range(20000)) / 20000
print(f"  Nationality bootstrap P(fail): {fails:.3f} (paper: roughly 26%)")
if not 0.20 <= fails <= 0.33:
    FAILURES.append(f"Nationality bootstrap P(fail) {fails:.3f} outside 26%±7")
rep14e = load("runs/full_qwen14b/report.json")
n_ext = rep14e["categories"]["Nationality"]["margins"]["quintile_n"] * 2
n_prb = rep14p["categories"]["Nationality"]["margins"]["n"]
print(f"  estimator n confound: extremes {n_ext} vs probe {n_prb} (2.5x)")
if abs(n_prb / n_ext - 2.5) > 0.05:
    FAILURES.append("estimator n-ratio is not 2.5x")

if FAILURES:
    print(f"RECOUNT FAILED after addenda: {len(FAILURES)}")
    sys.exit(1)
print("ADDENDA CLEAN")

print("\n== reanalysis addenda (2026-08-31): F/A/B/C ==")
ra = load("runs/_reanalysis/run1_reanalysis.json")
F = ra["F_statistic_sensitivity"]
for stat in ("min", "q05", "median", "mean", "ci_lo"):
    cats = set(F["by_statistic"][stat]["categories"])
    if not {"Disability_status", "Physical_appearance"} <= cats:
        FAILURES.append(f"F: {stat} does not keep the two headline categories")
check("F race max (most permissive stat)", max(F["race_max"].values()), 0.384, tol=0.002)
yi = [r for r in ra["A_bootstrap_floors"]
      if r["model"] == "yi-6b" and r["category"] == "Disability_status"][0]
check("A yi-6b Disability flip prob", yi["flip_prob"], 0.27, tol=0.02)
B = ra["B_floor_vs_n"]
check("B Religion delta mean", B["qwen-14b"]["delta_mean"], 0.163, tol=0.002)
check("B Race_ethnicity delta mean", B["qwen-1.8b"]["delta_mean"], 0.002, tol=0.002)
check("C ci halfwidth at 10 splits (x2 ~ 0.26)", ra["C_power"]["ci_halfwidth_at_10"], 0.131, tol=0.003)

if FAILURES:
    print(f"RECOUNT FAILED after reanalysis addenda: {len(FAILURES)}")
    for f in FAILURES: print("  -", f)
    sys.exit(1)
print("REANALYSIS ADDENDA CLEAN")

print("\n== R1 annotation contrast, qwen-1.8b (2026-08-31) ==")
r1 = load("runs/r1_annotation_qwen-1.8b/report_annotation_contrast.json")
fl = {c: v["mean"] for c, v in r1["observed_floor"].items()}
nc = {c: v["mean"] for c, v in r1["negative_control_floor"].items()}
sc = r1["specificity_control"]["per_category"]
check("R1 floor min", min(fl.values()), 0.975, tol=0.002)
check("R1 floor max", max(fl.values()), 0.984, tol=0.002)
check("R1 control min", min(nc.values()), 0.194, tol=0.002)
check("R1 control max", max(nc.values()), 0.569, tol=0.002)
if not all(r1["reproduces"].values()) or len(r1["reproduces"]) != 10:
    FAILURES.append("R1: not 10/10 reproduces")
check("R1 length self-floor", r1["length_direction_selfcheck"]["mean"], 0.868, tol=0.002)
coss = [v["abs_cos_with_length_direction"] for v in sc.values()]
check("R1 |cos| w/ length low", min(coss), 0.532, tol=0.002)
check("R1 |cos| w/ length high", max(coss), 0.631, tol=0.002)
fps = [v["floor_after_projection"] for v in sc.values()]
check("R1 floor-after-projection low", min(fps), 0.959, tol=0.002)
check("R1 floor-after-projection high", max(fps), 0.978, tol=0.002)
if any(v["verdict"] != "BIAS-SPECIFIC" for v in sc.values()):
    FAILURES.append("R1: some category read as LENGTH")
M = np.array(r1["cross_category"]["matrix"])
off = np.abs(M[~np.eye(len(M), dtype=bool)])
check("R1 cross-cat |cos| min", float(off.min()), 0.72, tol=0.01)
check("R1 cross-cat |cos| max", float(off.max()), 0.93, tol=0.01)
check("R1 cross-cat |cos| median", float(np.median(off)), 0.81, tol=0.01)
# Table 4 cells, exact
T4 = {"Age": (0.983, 0.194, 0.563, 0.975),
      "Disability_status": (0.979, 0.326, 0.532, 0.971),
      "Gender_identity": (0.980, 0.569, 0.540, 0.972),
      "Nationality": (0.975, 0.309, 0.631, 0.959),
      "Physical_appearance": (0.983, 0.364, 0.612, 0.973),
      "Race_ethnicity": (0.976, 0.237, 0.555, 0.966),
      "Race_x_SES": (0.982, 0.412, 0.549, 0.974),
      "Race_x_gender": (0.978, 0.304, 0.540, 0.969),
      "Religion": (0.983, 0.263, 0.591, 0.975),
      "Sexual_orientation": (0.984, 0.396, 0.545, 0.978)}
for c, (ef, en, ec, ep) in T4.items():
    check(f"T4 {c} floor", fl[c], ef)
    check(f"T4 {c} control", nc[c], en)
    check(f"T4 {c} cos", sc[c]["abs_cos_with_length_direction"], ec)
    check(f"T4 {c} proj", sc[c]["floor_after_projection"], ep)

if FAILURES:
    print(f"RECOUNT FAILED after R1 addenda: {len(FAILURES)}")
    for f in FAILURES: print("  -", f)
    sys.exit(1)
print("R1 ADDENDA CLEAN")
