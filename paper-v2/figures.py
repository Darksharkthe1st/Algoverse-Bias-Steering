"""Regenerate every figure from paper-v2/numbers.json.

    py paper-v2/figures.py

Run collect.py first. Every figure is a pure function of numbers.json, so the
figures cannot disagree with the prose -- both come from the same file. Each
one is written as .pdf (for LaTeX) and .png (for reading on a phone at 3am).

Figures, in the order the argument needs them:
  fig1_floors    per-category split-half floor with CI, against the shuffled
                 control band. The headline: which cells reproduce at all.
  fig2_cosines   the full 400-draw split-half distribution per cell, real vs
                 shuffled. Shows the regime, not just its summary statistic.
  fig3_steering  the causal arm: bias/refusal rates at baseline, under the
                 system-prompt control, and at each dose.

Missing data is drawn as missing (an open marker, a gap) rather than skipped,
so a half-finished queue looks half-finished instead of looking complete.
"""

import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("matplotlib/numpy missing. Install with:\n"
             "    py -m pip install matplotlib numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers.json")))
V, RAW, META = N["values"], N["raw"], N["meta"]

MODELS = [r["model"] for r in META["runs"]]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
CATS = sorted({k.split(".")[2] for k in V if k.startswith("buckets.")})
USABILITY = 0.50


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

# ---- fig 1: floors with CI against the control band ------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
width = 0.8 / max(len(MODELS), 1)
ctrl = [V[k] for k in V if k.startswith("control.") and k.endswith(".mean")]
if ctrl:
    hi = max(ctrl)
    ax.axhspan(min(ctrl), hi, color="#2E7D32", alpha=0.13, zorder=0)
    ax.axhline(hi, color="#2E7D32", lw=1.0, zorder=1)
    ax.text(len(CATS) - 0.45, hi + 0.02, "shuffled control", color="#1B5E20",
            fontsize=8, ha="right", va="bottom")

for i, m in enumerate(MODELS):
    xs, ys, los, his, miss = [], [], [], [], []
    for j, c in enumerate(CATS):
        x = j + (i - (len(MODELS) - 1) / 2) * width
        mu = V.get(f"floor.{m}.{c}.mean")
        if mu is None:
            miss.append(x)
            continue
        xs.append(x); ys.append(mu)
        los.append(mu - V.get(f"floor.{m}.{c}.ci_lo", mu))
        his.append(V.get(f"floor.{m}.{c}.ci_hi", mu) - mu)
    if xs:
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", ms=4.5, lw=1.1,
                    capsize=2.2, color=COLORS[i % len(COLORS)], label=m, zorder=3)
    if miss:
        ax.plot(miss, [-0.06] * len(miss), marker="x", ls="none", ms=4.5,
                color=COLORS[i % len(COLORS)], alpha=.55, zorder=3)

ax.axhline(USABILITY, color="#B0413E", lw=1.1, ls="--", zorder=2)
ax.text(-0.45, USABILITY + 0.015, f"usability bar {USABILITY:.2f}",
        color="#B0413E", fontsize=8, va="bottom")
ax.axhline(0.0, color="0.35", lw=0.8, zorder=2)
ax.text(-0.45, -0.075, "x = untestable (no contrast to split on)",
        fontsize=7.5, color="0.35", va="top")
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels([c.replace("_", " ") for c in CATS], rotation=32,
                   ha="right", fontsize=8)
ax.set_ylabel("split-half cosine")
ax.set_ylim(-0.12, 1.05)
ax.set_xlim(-0.6, len(CATS) - 0.4)
ax.legend(fontsize=8, ncol=len(MODELS), frameon=False,
          loc="lower left", bbox_to_anchor=(0, 1.01))
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig1_floors")

# ---- fig 2: the split-half distributions themselves ------------------------
cells = sorted(k[len("floor."):-len(".cosines")]
               for k in RAW if k.startswith("floor.") and RAW[k])
if cells:
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.42 * len(cells) + 1.0)))
    for i, cell in enumerate(cells):
        real = np.asarray(RAW[f"floor.{cell}.cosines"], float)
        shuf = np.asarray(RAW.get(f"control.{cell}.cosines", []), float)
        ax.scatter(real, np.full(real.size, i) + np.random.default_rng(0)
                   .normal(0, .055, real.size), s=3.2, alpha=.30,
                   color="#4C72B0", lw=0, zorder=3)
        ax.plot([real.mean()], [i], marker="|", ms=15, mew=2.0,
                color="#1F3B63", zorder=5)
        if shuf.size:
            ax.scatter(shuf, np.full(shuf.size, i) + np.random.default_rng(1)
                       .normal(0, .055, shuf.size), s=3.2, alpha=.30,
                       color="#B0413E", lw=0, zorder=3)
    ax.axvline(0.0, color="0.35", lw=0.8)
    ax.axvline(USABILITY, color="#B0413E", lw=1.0, ls="--")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([c.replace("_", " ") for c in cells], fontsize=8)
    ax.set_xlabel("split-half cosine (blue = real contrast, red = shuffled control)")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.6, len(cells) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_cosines")
else:
    print("  skipped fig2: no testable cells yet")

# ---- fig 3: the causal arm -------------------------------------------------
steer = sorted({k.split(".")[1] + "|" + k.split(".")[2]
                for k in V if k.startswith("steer.") and ".baseline." in k})
if steer:
    fig, axes = plt.subplots(1, max(len(steer), 1), sharey=True,
                             figsize=(2.6 * len(steer) + 1.2, 3.0), squeeze=False)
    for ax, s in zip(axes[0], steer):
        m, cell = s.split("|")
        arms, bias, refus = [], [], []
        for label, pre in (("base", f"steer.{m}.{cell}.baseline"),
                           ("sys", f"steer.{m}.{cell}.system_prompt_baseline")):
            if V.get(f"{pre}.biased_rate") is not None:
                arms.append(label)
                bias.append(V[f"{pre}.biased_rate"])
                refus.append(V.get(f"{pre}.refusal_rate", 0))
        alphas = sorted({k.split(".")[3] for k in V
                         if k.startswith(f"steer.{m}.{cell}.a")})
        for a in alphas:
            for sign in ("plus", "minus"):
                pre = f"steer.{m}.{cell}.{a}.{sign}"
                if V.get(f"{pre}.biased_rate") is not None:
                    arms.append(f"{a[1:]}{'+' if sign == 'plus' else '-'}")
                    bias.append(V[f"{pre}.biased_rate"])
                    refus.append(V.get(f"{pre}.refusal_rate", 0))
        x = np.arange(len(arms))
        ax.bar(x - 0.19, bias, .38, color="#C44E52", label="biased")
        ax.bar(x + 0.19, refus, .38, color="#4C72B0", label="refusal")
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontsize=7.5, rotation=45, ha="right")
        ax.set_title(f"{m} {cell.replace('_to_', '→')}", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("rate")
    axes[0][0].legend(fontsize=7.5, frameon=False)
    save(fig, "fig3_steering")
else:
    print("  skipped fig3: causal arm has not produced cells yet")

print("figures done")
