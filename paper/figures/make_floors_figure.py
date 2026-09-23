"""Figure 1: per-category extraction floor by model, against the positive control.

Every number is read from runs/ at plot time -- nothing is hard-coded -- so the
figure cannot drift from the artifacts. Repo rule: numbers trace to artifacts.

    python paper/figures/make_floors_figure.py
"""

import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "runs")

# --- floors, per model, per category -------------------------------------- #
floors = collections.defaultdict(dict)
for path in sorted(glob.glob(os.path.join(RUNS, "full_*", "report.json"))):
    d = json.load(open(path, encoding="utf-8"))
    model = d.get("model")
    for cat, v in (d.get("categories") or {}).items():
        fl = (v or {}).get("extraction_floor") or {}
        if "q05" in fl:
            floors[model][cat] = fl["q05"]

# --- the extraction positive control -------------------------------------- #
ctrl = []
for path in glob.glob(os.path.join(RUNS, "*extraction*control*.json")):
    d = json.load(open(path, encoding="utf-8"))
    for _pair, v in d.get("pairs", {}).items():
        ctrl.append(v["floor"]["q05"])
c_lo, c_hi = min(ctrl), max(ctrl)

MODELS = [m for m in ("qwen-1.8b", "gemma-2b", "yi-6b", "qwen-7b", "qwen-14b")
          if m in floors]
cats = sorted({c for m in MODELS for c in floors[m]},
              key=lambda c: np.mean([floors[m][c] for m in MODELS if c in floors[m]]))

fig, ax = plt.subplots(figsize=(9.4, 4.9))
x = np.arange(len(cats))
w = 0.16
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

for i, m in enumerate(MODELS):
    ax.bar(x + (i - (len(MODELS) - 1) / 2) * w,
           [floors[m].get(c, np.nan) for c in cats],
           w, label=m, color=colors[i % len(colors)],
           edgecolor="white", linewidth=0.4, zorder=3)

# control band
ax.axhspan(c_lo, c_hi, color="#2E7D32", alpha=0.13, zorder=0)
ax.axhline(c_hi, color="#2E7D32", lw=1.0, zorder=1)
ax.axhline(c_lo, color="#2E7D32", lw=1.0, zorder=1)
ax.text(-0.42, c_hi + 0.03,
        "extraction control (topic identity), $q_{05}$ "
        f"{c_lo:.2f}-{c_hi:.2f}: the same pipeline recovers a direction that must exist",
        color="#1B5E20", fontsize=8.2, va="bottom", ha="left", zorder=4)

# usability bar
ax.axhline(0.50, color="#B0413E", lw=1.1, ls="--", zorder=2)
ax.text(-0.42, 0.515, "usability bar 0.50 (post-hoc; sensitivity in text)",
        color="#B0413E", fontsize=8.2, va="bottom", ha="left", zorder=4)

ax.axhline(0.0, color="0.35", lw=0.8, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
ax.set_ylabel("split-half floor  ($q_{05}$ of cosine)", fontsize=9)
ax.set_ylim(-0.55, 1.16)
ax.set_xlim(-0.55, len(cats) - 0.45)
ax.legend(fontsize=8, ncol=len(MODELS), frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, -0.31), columnspacing=1.6, handlelength=1.4)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
ax.set_title("A direction that does not reproduce against itself is not a direction",
             fontsize=10.5, loc="left", pad=8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"floors.{ext}"), dpi=200, bbox_inches="tight")

n_cells = sum(len(floors[m]) for m in MODELS)
print(f"wrote floors.pdf / floors.png")
print(f"  control band {c_lo:.3f}-{c_hi:.3f} from {len(ctrl)} control splits")
print(f"  {len(cats)} categories x {len(MODELS)} models = {n_cells} cells plotted")
