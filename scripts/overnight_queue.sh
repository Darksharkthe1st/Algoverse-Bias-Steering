#!/usr/bin/env bash
# Overnight queue: REDO Experiment 1, then run the WP-43 diagnostics.
#
#   bash scripts/overnight_queue.sh 2>&1 | tee runs/_logs/overnight.log
#
# Start it, watch it clear preflight, go to bed.
#
# TIMING, measured rather than guessed:
#   GATE 1  smoke test, qwen-1.8b, 1 cat    ~2 min    <- proves capture runs
#   GATE 2  positive control, qwen-1.8b     ~10 min   <- proves it can find
#                                                        a direction at all
#   R1a  qwen-14b capture + fast read   25-70 min   <- the decisive answer
#   R1b  the other four models          1-2 h
#   R1c  n_splits=400 analysis, all 5   ~37 min CPU
#   P0-P3                               ~3 h
#   TOTAL                               6-9 h
#
# The capture rate is the uncertain part: run 1's 4.0 ops/sec anchor came
# from SCORING passes, and run_with_cache is slower. The range above assumes
# somewhere between 1x and 3x slower. Model downloads add 11-36 min on a
# fresh box (64 GB of weights across the five).
#
# ORDER, AND WHY
# --------------
# R1 runs FIRST because it is the actual redo. notes/17 concluded that run 1's
# floors collapsed because the CONTRAST was labelled by the model's own
# behaviour: items were ranked by their stereotype margin and the extremes taken
# as poles. Joad et al. get within-category floors of 0.95-0.99 from 32 items
# per class using DATASET ANNOTATIONS; run 1 used 240-600 and got -0.45 to +0.82.
# Sample size cannot explain that. The contrast can, and it is the only thing
# left that differs. R1 replaces it with `context_condition`, a label BBQ ships
# and the model never sees.
#
# P0-P3 run SECOND. They are not "hardening numbers we think are wrong" -- three
# of the four are tests that can kill a number, which is why they are still
# worth the machine time:
#   P0 tests whether the 0.50 bar transfers to the probe at all (defect S4). If
#      it does not, both clustering p-values are thresholded against the wrong
#      reference and the paper loses that section.
#   P1 tests whether alpha ever plateaus (defect S3, where the same category's
#      direction at alpha=1 and alpha=1e6 agree at only 0.10-0.21) and persists
#      the residuals that make every later analysis CPU-only (defect S5).
#   P2 tests whether the two heavy-tailed positives survive de-tailing (M2).
#   P3 asks whether the race negative is about the unit of analysis.
# What none of them do is touch the contrast, which is why they are second.
#
# Every run is independent and a failure does not stop the queue. Nothing starts
# at all if preflight fails, because spending a night on a broken environment is
# the one outcome worth preventing -- it is what incident I-10 was.

set -uo pipefail          # deliberately NOT -e: a failed run must not kill the queue

LOGDIR="runs/_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="$LOGDIR/overnight_${STAMP}_summary.txt"

say()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
note() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }

note "overnight queue started $(date -Is)"

# --------------------------------------------------------------------------- #
# Gate: preflight. Nothing runs if the environment is not proven.
# --------------------------------------------------------------------------- #
say "PREFLIGHT (nothing starts if this fails)"
if ! python3 -m scripts.preflight --load-model qwen-1.8b 2>&1 | tee "$LOGDIR/preflight_${STAMP}.log"; then
    note "PREFLIGHT FAILED — nothing was run. Read $LOGDIR/preflight_${STAMP}.log"
    note "Go to bed; this is a 10-minute fix in the morning, not a lost night."
    exit 1
fi
note "preflight: OK"

# --------------------------------------------------------------------------- #
# run <name> <command...>   — time it, log it, record the real exit code,
#                             and commit whatever landed.
# --------------------------------------------------------------------------- #
run () {
    local name="$1"; shift
    local log="$LOGDIR/${name}_${STAMP}.log"
    say "$name"
    local t0=$SECONDS
    "$@" > >(tee "$log") 2>&1
    local rc=${PIPESTATUS[0]}
    local mins=$(( (SECONDS - t0) / 60 ))
    if [ "$rc" -eq 0 ]; then
        note "[ok]   $name  (${mins} min)"
    else
        note "[FAIL] $name  (${mins} min, exit $rc) — see $log"
    fi
    # Checkpoint after every run: the box can vanish at any time.
    git add runs/ 2>/dev/null
    git commit -q -m "hardening: $name ($( [ "$rc" -eq 0 ] && echo ok || echo "exit $rc" ))" 2>/dev/null \
        && note "       committed" || note "       (nothing new to commit)"
    return 0
}

# --------------------------------------------------------------------------- #
# R1 — THE REDO. Annotation-derived contrast, all five models. ~45 min total.
#
#   direction_C = mean(resid | category C, ambiguous)
#               - mean(resid | category C, disambiguated)
#
# matched on the full BBQ scenario key. `context_condition` ships with the
# dataset and never consults the model, which is what closes defect M1 -- the
# floor/tilt confound measured at +0.66 to +0.77 across all five models.
#
# No generation, no judge, no margins: residual capture only, ~4,000 forward
# passes per model.
#
# CAPTURE INDEX: the script REFUSES to run without an explicit --capture-index
# and prints the last six chat-template tokens before capturing anything. -1 is
# the final prompt token. notes/19 6.1 records that the spec asks for -2 and the
# existing loader uses -1, and that the two may be the same position depending
# on the template. It does not matter for THIS comparison, because both
# contrasts are measured at the same site -- but it is recorded per model in
# capture_site.json so the choice is never implicit.
#
# KNOWN RISK, flagged before running: the disambiguated context is literally the
# ambiguous one plus a sentence, in 100% of 25,814 pairs, making it 2.0-2.3x
# longer. The direction may encode "read one more sentence" rather than bias.
# The analyse step runs a specificity control for exactly this and it may fail.
# --------------------------------------------------------------------------- #
# ============================================================================ #
# GATE 1 — SMOKE TEST. Smallest model, one category, ~2 minutes.
#
# The capture path has never been executed anywhere: this project's laptop has
# no GPU, so `capture` has run zero times, on zero models. Everything downstream
# of the residuals is tested (260 tests, and a pilot that validates each control
# against planted ground truth) but the forward pass itself is unexercised code.
#
# So prove it on the cheapest model before spending an hour on the 14B. This
# exercises every line the big runs use: chat template, tokenizer padding,
# run_with_cache, the capture index, and persistence.
# ============================================================================ #
say "GATE 1 — smoke test (qwen-1.8b, one category, ~2 min)"
if ! python3 -m scripts.run2_annotation_contrast capture \
        --model qwen-1.8b --capture-index -1 --n-per-arm 40 \
        --categories Disability_status \
        --out runs/_smoke_r1 2>&1 | tee "$LOGDIR/gate1_smoke_${STAMP}.log"; then
    note "GATE 1 FAILED — the capture path is broken. Nothing else was run."
    note "Read $LOGDIR/gate1_smoke_${STAMP}.log. This is the failure worth catching."
    exit 1
fi
python3 - <<'PY' || { note "GATE 1 FAILED — residuals are malformed."; exit 1; }
import json, pathlib, sys
import numpy as np
d = pathlib.Path("runs/_smoke_r1/residuals")
npys = sorted(d.glob("*.npy")) if d.is_dir() else []
if len(npys) != 2:
    print(f"  expected 2 residual files (one per arm), found {len(npys)}"); sys.exit(1)
for f in npys:
    a = np.load(f, mmap_mode="r")
    meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"  {f.name}: shape={a.shape} dtype={a.dtype} ids={len(meta['item_ids'])}")
    if a.ndim != 3 or a.shape[0] != len(meta["item_ids"]):
        print("  shape/id mismatch"); sys.exit(1)
    if not np.isfinite(np.asarray(a[0])).all():
        print("  non-finite values in the first row"); sys.exit(1)
print("  capture path works.")
PY
note "gate 1 (smoke): OK"

# ============================================================================ #
# GATE 2 — POSITIVE CONTROL. ~10 min on qwen-1.8b.
#
# Topic identity through the IDENTICAL pipeline. Without it a null result is
# uninterpretable: you cannot tell "the annotation contrast recovers nothing"
# from "our code is broken". Run 1 had this control and notes/11 calls it the
# single most valuable artifact of that session -- but it validated the OLD
# pipeline, so it has to be re-run through this one.
#
# It exits non-zero if any topic contrast fails to reproduce, and the queue
# stops there. That is the intended behaviour: a bias null measured with a
# pipeline that cannot find topic is not a finding.
# ============================================================================ #
say "GATE 2 — positive control (topic identity, qwen-1.8b)"
if ! python3 -m scripts.run2_annotation_contrast control \
        --model qwen-1.8b --capture-index -1 --n-per-arm 200 --n-splits 100 \
        --out runs/_control_r1_qwen-1.8b 2>&1 \
        | tee "$LOGDIR/gate2_control_${STAMP}.log"; then
    note "GATE 2 FAILED — the pipeline cannot recover a direction that must exist."
    note "STOP. Do not read any bias number until this passes."
    note "Read $LOGDIR/gate2_control_${STAMP}.log"
    exit 1
fi
note "gate 2 (positive control): OK — bias nulls below this line are interpretable"

# --- R1a: qwen-14b alone, fast read. 25-70 min. --------------------------- #
# One model answers the question. qwen-14b is the strongest and produced the
# most reproducible categories in run 1, so if the annotation contrast works
# anywhere it works here. n_splits=100 gives a 95% CI half-width of about
# +/-0.041 -- far more than enough to tell a floor of 0.9 from one of 0.3, and
# four times faster than the 400 the pre-registration fixes for the final number.
run R1a_annotation_qwen-14b \
    python3 -m scripts.run2_annotation_contrast capture \
        --model qwen-14b --capture-index -1 --n-per-arm 200 \
        --out runs/r1_annotation_qwen-14b

run R1a_analyse_qwen-14b_fast \
    python3 -m scripts.run2_annotation_contrast analyse \
        --out runs/r1_annotation_qwen-14b --n-splits 100 --n-per-arm 200

say "R1a READ THIS BEFORE THE REST RUNS"
note ""
note "R1a — annotation contrast, qwen-14b, n_splits=100:"
python3 - <<'PY' | tee -a "$SUMMARY"
import json, pathlib
p = pathlib.Path("runs/r1_annotation_qwen-14b/report_annotation_contrast.json")
if not p.exists():
    print("  R1a produced no report — capture or analyse failed, see the logs.")
else:
    r = json.loads(p.read_text(encoding="utf-8"))
    v = r.get("reproduces", {})
    yes = [k for k, x in v.items() if x == "YES"]
    print(f"  {len(yes)} of {len(v)} categories beat their own negative control")
    print(f"  they are: {', '.join(sorted(yes)) if yes else '(none)'}")
    print(f"  run 1, behavioural contrast: 10 of 46 model-category cells cleared 0.50")
    sc = r.get("specificity_control", {})
    print(f"  specificity control (is it just context length?): {sc.get('overall')}"
          f"  [{sc.get('n_failing')}/{sc.get('n_categories')} read as LENGTH]")
    cc = r.get("cross_category", {})
    print(f"  cross-category median |cos|: {cc.get('median_offdiagonal'):+.3f}")
PY
note ""

# --- R1b: the other four models. 1-2 h. ----------------------------------- #
# These run regardless, because a one-model result is not a result -- the
# cross-family replication is the part of this study that is genuinely ours.
# But R1a above has already told you the answer by the time these finish.
for M in qwen-7b yi-6b gemma-2b qwen-1.8b; do
    run "R1b_annotation_${M}" \
        python3 -m scripts.run2_annotation_contrast capture \
            --model "$M" --capture-index -1 --n-per-arm 200 \
            --out "runs/r1_annotation_${M}"
done

# --- R1c: the pre-registered analysis, n_splits=400, all five. ~37 min CPU. -- #
# notes/13 sec4 fixes 400 by calculation: at run 1's 90th-percentile split SD of
# 0.2023 it gives a 95% CI half-width of +/-0.020 on the mean. The n_splits=100
# read above is for speed; THIS is the number that goes in the paper.
for M in qwen-14b qwen-7b yi-6b gemma-2b qwen-1.8b; do
    run "R1c_analyse_${M}" \
        python3 -m scripts.run2_annotation_contrast analyse \
            --out "runs/r1_annotation_${M}" --n-splits 400 --n-per-arm 200
done

# --------------------------------------------------------------------------- #
# P0 — topic control through the probe. ~40 min.
# Decides whether the 0.50 bar transfers to the probe. If it does not, every
# probe number in the manuscript is thresholded against the wrong reference.
# --------------------------------------------------------------------------- #
run P0_probe_topic_control \
    python3 scripts/extraction_positive_control.py --model qwen-14b \
        --method probe --alphas 1 1e2 1e3 1e4 1e5 1e6

# --------------------------------------------------------------------------- #
# P1 — extend the alpha sweep past its boundary AND persist residuals. ~30 min.
# --save-residuals is the important half: it makes P2/P3 and every future
# alpha/threshold question CPU-only instead of another rental.
# --------------------------------------------------------------------------- #
run P1_alpha_extension \
    python3 scripts/probe_alpha_sweep.py --model qwen-14b \
        --alphas 1e6 1e7 1e8 1e9 1e10 --save-residuals \
        --out runs/_probe_alpha_sweep_qwen-14b_ext.json

# --------------------------------------------------------------------------- #
# P3 — the stereotyped-group split. Cheap, so it runs before P2.
# Margins are sliced from the committed cache, so this is residual capture only:
# roughly 90 s of forward passes per subset.
# The split is frozen in runs/_p3_manifest.json and must not be edited here.
# --------------------------------------------------------------------------- #
say "P3 — stereotyped-group split (11 subsets, qwen-14b)"
python3 -m scripts.p3_subgroup_manifest --out runs/_p3_manifest.json \
    > "$LOGDIR/p3_manifest_${STAMP}.log" 2>&1
note "p3 manifest: $(python3 -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('runs/_p3_manifest.json').read_bytes()).hexdigest()[:16])")"

python3 - "$STAMP" <<'PY' > "$LOGDIR/p3_plan_${STAMP}.sh"
import json, sys, shlex
man = json.load(open("runs/_p3_manifest.json", encoding="utf-8"))
for cat, info in man["categories"].items():
    for r in info["groups"]:
        if not r["tested"]:
            continue
        g = r["group"]
        tag = g.replace(" ", "-")
        print(" ".join(shlex.quote(x) for x in [
            "python3", "scripts/bias_taxonomy_run.py",
            "--model", "qwen-14b", "--ambig-limit", "600",
            "--categories", cat, "--stereotyped-group", g,
            "--method", "extremes", "--margins-cache", "runs/_margins_cache",
            "--out-dir", f"runs/p3_qwen14b_{cat}_{tag}",
        ]))
PY

while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    name="P3_$(echo "$cmd" | sed -n 's/.*--out-dir runs\/p3_qwen14b_\([^ ]*\).*/\1/p')"
    run "$name" bash -c "$cmd"
done < "$LOGDIR/p3_plan_${STAMP}.sh"

# --------------------------------------------------------------------------- #
# P2 — depth-unified + tail-trimmed re-run, all four models. ~1.5 h.
# Last because it is the most expensive and it answers a caveat the manuscript
# already discloses, rather than opening a new question.
# --------------------------------------------------------------------------- #
for M in qwen-14b qwen-7b gemma-2b yi-6b; do
    run "P2_unified_${M}" \
        python3 scripts/bias_taxonomy_run.py --model "$M" --ambig-limit 600 \
            --method extremes --cluster-usable-only \
            --margins-cache runs/_margins_cache --out-dir "runs/full_${M}_unified"
    run "P2_trim05_${M}" \
        python3 scripts/bias_taxonomy_run.py --model "$M" --ambig-limit 600 \
            --method extremes --tail-trim 0.05 --cluster-usable-only \
            --margins-cache runs/_margins_cache --out-dir "runs/full_${M}_trim05"
done

# --------------------------------------------------------------------------- #
say "DONE"
note ""
note "queue finished $(date -Is)"
note ""
note "THE NUMBER THAT MATTERS: R1_analyse_* reports how many categories beat"
note "their own negative control under the ANNOTATION contrast. Run 1 cleared"
note "0.50 in 10 of 46 model-category cells under the behavioural one."
note ""
note "IN THE MORNING, IN THIS ORDER:"
note "  1. git push          — the box can be reclaimed at any time"
note "  2. upload runs/_residuals/ to Drive or HF — NOT committed, ~2.5 GB,"
note "     and losing it turns every follow-up back into a rental"
note "  3. read $SUMMARY"
cat "$SUMMARY"
