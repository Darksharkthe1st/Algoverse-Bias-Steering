"""Figure 1, upgraded: floors with bootstrap intervals rather than bare points.

    python paper/figures/make_floors_ci_figure.py

The first version of this figure plotted the q05 of 10 split-half cosines as a
bar. That is exactly what defect S1 objects to -- a decision statistic from ten
draws, drawn as though it were exact. The ten raw cosines are in report.json, so
the interval is recoverable and there is no reason to hide it.

Two panels:
  left   every model-category floor, mean with a bootstrap 95% CI, against the
         extraction control band
  right  the same data collapsed per category across models, which is what the
         paper's claim is actually about
"""

import collections
import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "runs")

MODELS = ["qwen-1.8b", "gemma-2b", "yi-6b", "qwen-7b", "qwen-14b"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def boot_ci(v, n_boot=20000, seed=0):
    a = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


# ---- data ------------------------------------------------------------------
floors = collections.defaultdict(dict)      # model -> cat -> (mean, lo, hi)
for p in sorted(glob.glob(os.path.join(RUNS, "full_*", "report.json"))):
    d = json.load(open(p, encoding="utf-8"))
    for cat, v in (d.get("categories") or {}).items():
        cos = ((v or {}).get("extraction_floor") or {}).get("cosines")
        if cos:
            lo, hi = boot_ci(cos)
            floors[d["model"]][cat] = (statistics.mean(cos), lo, hi)

ctrl = []
for p in glob.glob(os.path.join(RUNS, "*extraction*control*.json")):
    d = json.load(open(p, encoding="utf-8"))
    for _k, v in d.get("pairs", {}).items():
        ctrl.append(statistics.mean(v["floor"]["cosines"]))
c_lo, c_hi = min(ctrl), max(ctrl)

present = [m for m in MODELS if m in floors]
cats = sorted({c for m in present for c in floors[m]},
              key=lambda c: np.mean([floors[m][c][0] for m in present if c in floors[m]]))

fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [2.5, 1]})

# ---- left panel: every cell, with its interval ------------------------------
x = np.arange(len(cats)); w = 0.16
for i, m in enumerate(present):
    xs, ys, lo, hi = [], [], [], []
    for j, c in enumerate(cats):
        if c not in floors[m]:
            continue
        mu, l, h = floors[m][c]
        xs.append(j + (i - (len(present) - 1) / 2) * w)
        ys.append(mu); lo.append(mu - l); hi.append(h - mu)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=4.2, lw=0, elinewidth=1.3,
                capsize=2.2, color=COLORS[i % len(COLORS)], label=m, zorder=3)

ax.axhspan(c_lo, c_hi, color="#2E7D32", alpha=0.13, zorder=0)
ax.axhline(c_hi, color="#2E7D32", lw=1.0, zorder=1)
ax.axhline(c_lo, color="#2E7D32", lw=1.0, zorder=1)
ax.text(-0.45, c_hi + 0.025,
        f"extraction control (topic identity), mean {c_lo:.2f}-{c_hi:.2f}: "
        "the same pipeline recovers a direction that must exist",
        color="#1B5E20", fontsize=8.2, va="bottom", ha="left")
ax.axhline(0.50, color="#B0413E", lw=1.1, ls="--", zorder=2)
ax.text(-0.45, 0.515, "usability bar 0.50 (post-hoc)", color="#B0413E",
        fontsize=8.2, va="bottom", ha="left")
ax.axhline(0.0, color="0.35", lw=0.8, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=7.6)
ax.set_ylabel("split-half floor: mean cosine, 95% bootstrap CI", fontsize=9)
ax.set_ylim(-0.45, 1.16)
ax.set_xlim(-0.6, len(cats) - 0.4)
ax.legend(fontsize=8, ncol=len(present), frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, -0.34), columnspacing=1.5, handlelength=1.2)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
ax.set_title("Every model-category floor, with its interval",
             fontsize=10, loc="left", pad=8)

# ---- right panel: per category, pooled across models -------------------------
mu = [np.mean([floors[m][c][0] for m in present if c in floors[m]]) for c in cats]
lo = [min(floors[m][c][1] for m in present if c in floors[m]) for c in cats]
hi = [max(floors[m][c][2] for m in present if c in floors[m]) for c in cats]
y = np.arange(len(cats))
colors = ["#2E7D32" if v >= 0.5 else ("#B0413E" if v < 0.2 else "#8a8a8a") for v in mu]
ax2.barh(y, mu, color=colors, alpha=.85, height=.62, zorder=3)
ax2.hlines(y, lo, hi, color="0.25", lw=1.1, zorder=4)
ax2.axvspan(c_lo, c_hi, color="#2E7D32", alpha=0.13, zorder=0)
ax2.axvline(0.50, color="#B0413E", lw=1.1, ls="--", zorder=2)
ax2.axvline(0.0, color="0.35", lw=0.8, zorder=2)
ax2.set_yticks(y)
ax2.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=7.6)
ax2.set_xlabel("mean floor across models", fontsize=9)
ax2.set_xlim(-0.45, 1.0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelsize=8)
ax2.set_title("Pooled per category", fontsize=10, loc="left", pad=8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"floors_ci.{ext}"), dpi=200, bbox_inches="tight")

n = sum(len(floors[m]) for m in present)
print("wrote floors_ci.pdf / floors_ci.png")
print(f"  {n} model-category cells, each with a 20k bootstrap CI over its 10 splits")
print(f"  control band (means): {c_lo:.3f}-{c_hi:.3f}")
