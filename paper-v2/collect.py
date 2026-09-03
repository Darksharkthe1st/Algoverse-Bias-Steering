"""Collect every paper-facing number out of runs/ into numbers.json + numbers.tex.

    py paper-v2/collect.py

Re-run it whenever a run finishes. It is idempotent and reads whatever exists,
so it works fine while the queue is still going -- categories that are not
TESTABLE yet are simply reported as missing rather than crashing the build.

Why this exists: the submitted paper hard-codes its numbers in prose, so a
rerun of the experiment means hand-editing dozens of sentences and hoping none
were missed. Here the manuscript writes \rv{floor.qwen-14b.Age.mean} and this
script decides what that renders as. A number that the data does not support
renders as a loud ??key?? in the PDF instead of silently staying stale.

Reads:
  runs/r3_behavioural_*/report_behavioural.json   (the new experiment)
  runs/r3_behavioural_*/report_steering.json      (causal arm, optional)

Writes:
  paper-v2/numbers.json   exact values, full precision, plus raw cosine arrays
  paper-v2/numbers.tex    \rv{} macros for the manuscript
"""

import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")


def clean(o):
    """JSON round-trips NaN as a bare token that strict parsers reject."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def model_of(run_dir):
    return os.path.basename(run_dir).replace("r3_behavioural_", "")


# ---- gather ----------------------------------------------------------------
values = {}      # flat key -> exact value, this is what the paper cites
raw = {}         # flat key -> list, the distributions the figures need
meta = {"runs": [], "missing": [], "untestable": []}

for run_dir in sorted(glob.glob(os.path.join(RUNS, "r3_behavioural_*"))):
    model = model_of(run_dir)
    rb = os.path.join(run_dir, "report_behavioural.json")
    if not os.path.exists(rb):
        meta["missing"].append(f"{model}: no report_behavioural.json")
        continue
    d = json.load(open(rb))
    meta["runs"].append({
        "model": model,
        "contrast": d.get("contrast"),
        "n_splits": d.get("n_splits"),
        "min_bucket": d.get("min_bucket"),
    })
    values[f"design.{model}.n_splits"] = d.get("n_splits")
    values[f"design.{model}.min_bucket"] = d.get("min_bucket")

    testable = []
    for cat, c in d.get("per_category", {}).items():
        b = c.get("buckets", {})
        values[f"buckets.{model}.{cat}.n_biased"] = b.get("n_biased")
        values[f"buckets.{model}.{cat}.n_refusal"] = b.get("n_refusal")
        values[f"buckets.{model}.{cat}.n_total"] = b.get("n_total")
        values[f"buckets.{model}.{cat}.refusal_rate"] = b.get("refusal_rate")
        values[f"buckets.{model}.{cat}.unparsed_rate"] = b.get("unparsed_rate")
        values[f"buckets.{model}.{cat}.status"] = b.get("status")
        if c.get("judge_vs_heuristic_agreement") is not None:
            values[f"agree.{model}.{cat}"] = c["judge_vs_heuristic_agreement"]

        if b.get("status") != "TESTABLE" or "floor" not in c:
            meta["untestable"].append({
                "model": model, "category": cat,
                "status": b.get("status"),
                "reason": b.get("untestable_reason"),
            })
            continue

        f = c["floor"]
        testable.append(f["mean"])
        for k in ("mean", "sd", "ci_lo", "ci_hi", "n_biased", "n_refusal",
                  "sensitivity_unweighted_median_mean"):
            if k in f:
                values[f"floor.{model}.{cat}.{k}"] = f[k]
        raw[f"floor.{model}.{cat}.cosines"] = f.get("cosines", [])

        s = c.get("shuffled_control", {})
        for k in ("mean", "sd", "ci_lo", "ci_hi"):
            if k in s:
                values[f"control.{model}.{cat}.{k}"] = s[k]
        raw[f"control.{model}.{cat}.cosines"] = s.get("cosines", [])
        if c.get("reproduces") is not None:
            values[f"reproduces.{model}.{cat}"] = c["reproduces"]

    # per-model aggregates -- the sentences that say "0.93 to 0.99"
    values[f"agg.{model}.n_testable"] = len(testable)
    values[f"agg.{model}.n_categories"] = len(d.get("per_category", {}))
    if testable:
        values[f"agg.{model}.floor_min"] = min(testable)
        values[f"agg.{model}.floor_max"] = max(testable)

    # ---- causal arm, if it ran ----
    rs = os.path.join(run_dir, "report_steering.json")
    if os.path.exists(rs):
        st = clean(json.load(open(rs)))
        values[f"steer.{model}.judge_model"] = st.get("judge_model")
        values[f"steer.{model}.alphas"] = ", ".join(str(a) for a in st.get("alphas", []))
        for cell, cd in st.get("cells", {}).items():
            key = cell.replace("->", "_to_")
            for arm in ("baseline", "system_prompt_baseline"):
                for k, v in (cd.get(arm) or {}).items():
                    if isinstance(v, (int, float)) or v is None:
                        values[f"steer.{model}.{key}.{arm}.{k}"] = v
            for alpha, ad in (cd.get("doses") or {}).items():
                for sign, sd in ad.items():
                    if not isinstance(sd, dict):
                        continue
                    for k, v in sd.items():
                        if isinstance(v, (int, float)) or v is None:
                            values[f"steer.{model}.{key}.a{alpha}.{sign}.{k}"] = v

# cross-model aggregates -- the abstract's headline range
all_floors = [v for k, v in values.items()
              if k.startswith("floor.") and k.endswith(".mean") and v is not None]
if all_floors:
    values["agg.all.floor_min"] = min(all_floors)
    values["agg.all.floor_max"] = max(all_floors)
    values["agg.all.n_cells"] = len(all_floors)
all_ctrl = [v for k, v in values.items()
            if k.startswith("control.") and k.endswith(".mean") and v is not None]
if all_ctrl:
    values["agg.all.control_min"] = min(all_ctrl)
    values["agg.all.control_max"] = max(all_ctrl)
values["agg.all.n_models"] = len(meta["runs"])


# ---- write -----------------------------------------------------------------
with open(os.path.join(HERE, "numbers.json"), "w") as fh:
    json.dump(clean({"values": values, "raw": raw, "meta": meta}), fh, indent=1)


def fmt(key, v):
    if v is None:
        return r"\textbf{??" + key.replace("_", r"\_") + "??}"
    if isinstance(v, str):
        return v.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if key.endswith("_rate") or key.endswith("accuracy_vs_bbq_label"):
        return f"{100 * v:.1f}\\%"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


lines = [
    "% Generated by paper-v2/collect.py -- DO NOT EDIT BY HAND.",
    "% Re-run `py paper-v2/collect.py` after every run and recompile.",
    "% Cite a number in the manuscript as \\rv{floor.qwen-14b.Age.mean}.",
    "% An unknown key renders as a bold ??key?? so it cannot ship silently.",
    r"\makeatletter",
    r"\newcommand{\rv}[1]{\@ifundefined{rv@#1}"
    r"{\textbf{??#1??}}{\csname rv@#1\endcsname}}",
    r"\newcommand{\rvdef}[2]{\expandafter\gdef\csname rv@#1\endcsname{#2}}",
    r"\makeatother",
    "",
]
for k in sorted(values):
    lines.append(r"\rvdef{%s}{%s}" % (k, fmt(k, values[k])))
with open(os.path.join(HERE, "numbers.tex"), "w") as fh:
    fh.write("\n".join(lines) + "\n")


# ---- report ----------------------------------------------------------------
print(f"models found : {[r['model'] for r in meta['runs']]}")
print(f"numbers      : {len(values)} values, {len(raw)} distributions")
for r in meta["runs"]:
    m = r["model"]
    n, tot = values.get(f"agg.{m}.n_testable"), values.get(f"agg.{m}.n_categories")
    lo, hi = values.get(f"agg.{m}.floor_min"), values.get(f"agg.{m}.floor_max")
    span = f"{lo:.2f} to {hi:.2f}" if lo is not None else "none yet"
    print(f"  {m:<10} {n}/{tot} categories testable, floors {span}")
if meta["untestable"]:
    print(f"\nnot yet testable ({len(meta['untestable'])}):")
    for u in meta["untestable"][:12]:
        print(f"  {u['model']:<10} {u['category']:<20} {u['status']} {u['reason'] or ''}")
    if len(meta["untestable"]) > 12:
        print(f"  ... and {len(meta['untestable']) - 12} more")
print("\nwrote paper-v2/numbers.json and paper-v2/numbers.tex")
