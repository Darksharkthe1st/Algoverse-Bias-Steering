#!/usr/bin/env bash
# Overnight queue: REDO Experiment 1, then run the WP-43 diagnostics.
#
#   bash scripts/overnight_queue.sh 2>&1 | tee runs/_logs/overnight.log
#
# Start it, watch it clear preflight, go to bed. ~4-5 h wall clock.
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
for M in qwen-14b qwen-7b gemma-2b yi-6b qwen-1.8b; do
    run "R1_annotation_${M}" \
        python3 -m scripts.run2_annotation_contrast capture \
            --model "$M" --capture-index -1 --n-per-arm 200 \
            --out "runs/r1_annotation_${M}"
done

# The analysis is CPU-only and reads the cached residuals R1 just wrote.
for M in qwen-14b qwen-7b gemma-2b yi-6b qwen-1.8b; do
    run "R1_analyse_${M}" \
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
