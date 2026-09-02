#!/usr/bin/env bash
# Overnight queue: REDO Experiment 1, then run the WP-43 diagnostics.
#
#   bash scripts/overnight_queue.sh 2>&1 | tee runs/_logs/overnight.log
#
# RUN IT INSIDE tmux. The box is LAMBDA, reached over SSH (notes/25 -- it is NOT
# the Algoverse JupyterHub cluster, whatever notes/24 says). This queue is an
# 11-16 HOUR job, and a plain ssh session dies when the laptop sleeps, the wifi
# drops, or the terminal closes -- taking the queue with it while the instance
# keeps billing.
#
#   tmux new -s run3
#   bash scripts/overnight_queue.sh 2>&1 | tee runs/_logs/overnight.log
#   # Ctrl-b then d to detach; the queue keeps running
#
#   # from the laptop, any time:
#   ssh -i ~/.ssh/lambda_jeremiah ubuntu@<IP> -t 'tmux attach -t run3'
#   ssh -i ~/.ssh/lambda_jeremiah ubuntu@<IP> 'tail -40 ~/Algoverse-Bias-Steering/runs/_logs/overnight.log'
#
# Start it, watch it clear preflight, go to bed.
#
# TIMING. The generation count is arithmetic; the RATE is the uncertain term.
#
# ALL FOUR GATES RUN FIRST, before any long job, so the run becomes unattended
# in ~30 min (plus model downloads) and that is when you can go to bed:
#   GATE 1    R1 capture smoke, qwen-1.8b        ~2 min
#   GATE 2    R1 positive control                ~10 min
#   GATE R3-1 R3 generate + judge + extract      ~5 min
#   GATE R3-2 R3 positive control                ~10 min
#
# Then, unattended:
#   R3 generate   400x10x2 ambig + 100x10 answerable, x5 models    45,000 gen
#   R3 judge      one forward pass per completion (qwen-1.8b)      ~50,000
#   R3 extract    CPU only, from the cached residuals              ~1.5 h
#   R3d toggle    10 cells x 10 sweeps x 200 items, x5 models   100,000 gen
#                 (2 alphas, not 4: Phase 3 asks WHETHER the vectors do
#                  anything, so the budget buys precision per point --
#                  SE 0.035 at n=200 against 0.046 at n=120. Same cost.)
#   R3e cross     DISABLED (RUN_R3E=0) -- would add 72,000 gen
#   R1a           annotation contrast, qwen-14b, capture only      ~1 h
#   R1b/R1c, P0-P3                                                 ~4 h
#
# 145,000 generations at 48 new tokens. TransformerLens `generate` is not a fast
# serving path: at ~3-6 gen/s on a 14B that is 7-14 h of generation alone, so
# budget 11-16 h for R3 and expect R1b/P0-P3 to be what runs out of window.
#
# Earlier revisions of this header said "TOTAL 6-9 h". That was an estimate built
# from run 1's SCORING throughput, which does not transfer to generation.
#
# ORDER, AND WHY
# --------------
# R3 (the behavioural contrast and the taxonomy) runs FIRST, because it is the
# active programme and the phase that needs the most GPU. R1a follows: it is the
# cheap decisive read on the ALTERNATIVE contrast, and one hour spent on it is
# one hour not spent on cross-application. If the window ends early, R3 is the
# part you wanted.
#
# R1 exists because it is the actual redo. notes/17 concluded that run 1's
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

# --- Off-box durability. A fresh Lambda box has NO git identity: `git commit`
# --- fails, stderr goes to /dev/null, and run() prints "(nothing new to
# --- commit)" -- indistinguishable from success, at every step, all night. Set
# --- it, then prove push works NOW rather than at 8am with everything still on
# --- a box that bills until you terminate it.
git config user.email >/dev/null 2>&1 || git config user.email "run3@box.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "run3 box"
# Push is DELIBERATELY disabled on a box with no credentials: an interactive
# git auth prompt inside tmux hung this queue for 1h41m while the instance
# billed, and killing it was the only way out. The push URL is pointed at a
# dead path so git fails in 0s instead of blocking. Artifacts leave by scp
# (sync_from_box.ps1 / notes/25), which is the route notes/14 §3 wants anyway
# because the verifier must run against the LAPTOP copy.
#
# So: probe, and only HARD FAIL when push is supposed to work and does not.
PUSH_URL="$(git remote get-url --push origin 2>/dev/null)"
case "$PUSH_URL" in
  *push-disabled*)
    note "push: DISABLED on purpose ($PUSH_URL)"
    note "*** NOTHING LEAVES THIS BOX BY GIT. Pull artifacts with scp/rsync"
    note "*** BEFORE terminating, or it is defect S5 all over again."
    ;;
  *)
    if git push -q origin HEAD > "$LOGDIR/push_probe_${STAMP}.log" 2>&1; then
        note "push probe: OK — commits will reach the remote"
    else
        note "PUSH PROBE FAILED — nothing would survive this box. Fix the remote"
        note "or the credentials before spending GPU hours."
        note "See $LOGDIR/push_probe_${STAMP}.log"
        exit 1
    fi
    ;;
esac

: "${RESID_BACKUP:=}"
if [ -z "$RESID_BACKUP" ]; then
    note "*** RESID_BACKUP unset: residual tensors will exist ONLY on this box."
    note "*** They are gitignored (GB), so no commit carries them. Either export"
    note "*** RESID_BACKUP=user@host:/path before starting, or run"
    note "*** sync_from_box.ps1 from the laptop BEFORE terminating. That is"
    note "*** defect S5 and it has already cost this project a week."
fi

# --------------------------------------------------------------------------- #
# run <name> <command...>   — time it, log it, record the real exit code,
#                             and commit whatever landed.
# --------------------------------------------------------------------------- #
# Incident I-9 (notes/11 §9.4): "wait on the resource, not the process." A
# straggler holding 29 GB starved three queued steps and cost 35 minutes. The
# queue now chains 5 models x (generate + judge + steer) unattended, so poll for
# free VRAM before each step rather than starting the instant the last PID exits.
wait_for_vram () {
    local need_mb="${1:-20000}" waited=0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    while [ "$waited" -lt 300 ]; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -z "$free" ] && return 0
        [ "$free" -ge "$need_mb" ] && return 0
        [ "$waited" -eq 0 ] && printf '  waiting for VRAM (%s MB free, need %s) ...
' "$free" "$need_mb"
        sleep 15; waited=$((waited + 15))
    done
    note "       (proceeded after 5 min with VRAM still low)"
}

run () {
    local name="$1"; shift
    local log="$LOGDIR/${name}_${STAMP}.log"
    say "$name"
    wait_for_vram 20000
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
    if git commit -q -m "run3: $name ($( [ "$rc" -eq 0 ] && echo ok || echo "exit $rc" ))" 2>/dev/null; then
        if git push -q origin HEAD 2>/dev/null; then
            note "       committed + pushed"
        else
            note "       *** COMMITTED BUT PUSH FAILED — this work is box-local"
        fi
    else
        note "       (nothing new to commit)"
    fi
    # Residual arrays are gitignored (GB, over GitHub 100 MiB), so the commit
    # above does NOT carry them. Mirror them if a destination is set.
    if [ -n "${RESID_BACKUP:-}" ]; then
        if rsync -a --partial --include='*/' --include='residuals/***' \
                 --exclude='*' runs/ "$RESID_BACKUP/runs/" >/dev/null 2>&1; then
            note "       residuals mirrored to $RESID_BACKUP"
        else
            note "       *** RESIDUAL MIRROR FAILED — they exist only on this box"
        fi
    fi
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

# ============================================================================ #
# RUN 3 — behavioural contrast and bias taxonomy.
#
# Placed after R1a (the cheap decisive annotation-contrast read) and before R1b,
# because run 3 is the active programme and R1b/R1c are replication breadth.
#
# GATE R3-1 and R3-2 exist for the reason commit 6bc9a90 added gates 1 and 2 to
# R1: this GPU path has never executed anywhere, and a null taxonomy measured
# with broken code is indistinguishable from a real one.
#
# JUDGE MODEL: must not be the target (self-labelling is circular; the runner
# refuses it). yi-6b is ungated and a different family from the Qwen targets.
# ============================================================================ #
# JUDGE: gpt-4o-mini over the API, decided 2026-09-02 after every LOCAL judge
# failed C-1 on real completions. Measured order agreement against a 0.33
# chance line and a 0.95 bar: gemma-2b 0.095, qwen-1.8b 0.125, qwen-7b 0.425,
# qwen-14b 0.745 -- all five scored 5/5 on trivially clear inputs and fell
# apart on hedged prose. yi-6b could not even build a judge (SentencePiece has
# no unambiguous single token for 1/2/3). Two rescue hypotheses were tested and
# both failed: removing the duplicated option list from the scenario (0.745 ->
# 0.705) and scoring agreement on the directive LABEL rather than the raw
# OPTION_n (0.745 -> 0.770). The judge, not the metric, was the problem.
#
# The API judge also removes self-judging entirely: no target model is the
# judge, so R3f below is moot and the guard skips it.
R3_JUDGE_MODEL=gpt-4o-mini
R3_JUDGE=qwen-1.8b        # one judge for EVERY target: a judge quirk then cannot
                          # be mistaken for a model difference. Self-judging on the
                          # qwen-1.8b cell is permitted and cross-checked below.
R3_JUDGE_ALT=yi-6b        # independent second opinion for the self-judged cell

say "GATE R3-1 — smoke test (qwen-1.8b, one category, ~3 min)"
if ! python3 -m scripts.run3_behavioural_contrast generate \
        --model qwen-1.8b --capture-index -1 \
        --categories Disability_status --n-per-category 24 --n-control 8 \
        --out runs/_smoke_r3 2>&1 | tee "$LOGDIR/gate_r3_1_${STAMP}.log"; then
    note "GATE R3-1 FAILED — the run-3 generate path is broken. Nothing else ran."
    exit 1
fi
python3 - <<'PY' || { note "GATE R3-1 FAILED — run-3 artifacts are malformed."; exit 1; }
import json, pathlib, sys
import numpy as np
d = pathlib.Path("runs/_smoke_r3")
npys = sorted((d / "residuals").glob("*.npy"))
if len(npys) != 2:
    print(f"  expected 2 residual files, found {len(npys)}"); sys.exit(1)
for f in npys:
    a = np.load(f, mmap_mode="r")
    m = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"  {f.name}: shape={a.shape} ids={len(m['item_ids'])}")
    if a.ndim != 3 or a.shape[0] != len(m["item_ids"]):
        print("  shape/id mismatch"); sys.exit(1)
    if not np.isfinite(np.asarray(a)).all():
        print("  non-finite residuals"); sys.exit(1)
recs = [json.loads(l) for l in (d / "responses.jsonl").open(encoding="utf-8") if l.strip()]
if not recs:
    print("  responses.jsonl is empty"); sys.exit(1)
blank = sum(1 for r in recs if not r["response"].strip())
same = sum(1 for r in recs if r["response"] == r["response_swapped"])
print(f"  {len(recs)} completions; {blank} blank; {same}/{len(recs)} identical under option swap")
if blank > len(recs) // 4:
    print("  too many empty completions -- check the chat template"); sys.exit(1)
print("  run-3 generate path works.")
PY
# The smoke must also cover judge and extract, or those paths first execute on
# the real run at hour three. This loads $R3_JUDGE early -- a cost, not a waste:
# R1b downloads it anyway. --min-bucket 2 because 24 items cannot clear 32; the
# NUMBERS here are meaningless and are not read, only the exit codes.
if ! python3 -m scripts.run3_behavioural_contrast judge         --out runs/_smoke_r3 --judge-backend openai --judge-model "$R3_JUDGE_MODEL"         --qualify-n 16 2>&1 | tee -a "$LOGDIR/gate_r3_1_${STAMP}.log"; then
    note "GATE R3-1 FAILED — the judge path is broken (or the judge did not"
    note "qualify). Read $LOGDIR/gate_r3_1_${STAMP}.log"
    exit 1
fi
if ! python3 -m scripts.run3_behavioural_contrast extract         --out runs/_smoke_r3 --n-splits 10 --n-permutations 10 --min-bucket 2         2>&1 | tee -a "$LOGDIR/gate_r3_1_${STAMP}.log"; then
    note "GATE R3-1 FAILED — the extract path is broken."
    exit 1
fi
note "gate R3-1 (smoke: generate + judge + extract): OK"

say "GATE R3-2 — positive control (topic identity, qwen-1.8b, ~10 min)"
if ! python3 -m scripts.run3_behavioural_contrast control \
        --model qwen-1.8b --capture-index -1 --n-per-arm 200 --n-splits 100 \
        --out runs/_control_r3_qwen-1.8b 2>&1 \
        | tee "$LOGDIR/gate_r3_2_${STAMP}.log"; then
    note "GATE R3-2 FAILED — run 3's estimator cannot recover a direction that"
    note "must exist. STOP: a bias null below this line would be uninterpretable."
    exit 1
fi
note "gate R3-2 (positive control): OK — run-3 nulls are interpretable"


# --- R3: the real run. Two targets, so the taxonomy is not single-model. ---- #
# notes/13 §1 sets the bar at "at least two model families". qwen-14b is the
# discovery model; qwen-7b replicates. The judge is yi-6b for both: ungated, and
# a different family from the targets (the runner refuses judge == target, and
# same-family judging is a weaker form of the same circularity).
#
# STAGED DELIBERATELY. Phase 3 (the diagonal toggle test) runs BEFORE Phase 4.1
# (cross-application) for every model, because cross-application is only
# meaningful if a vector is causal on its own category. If the window runs out
# mid-queue you still have the phase that gates the other.
# Five targets across three families: Qwen (14b, 7b, 1.8b), Yi (6b), Gemma (2b).
# notes/13 §1 asks for at least two families -- the three Qwens are one.
# gemma-2b is gated; it is best-effort, so a 403 costs the download attempt and
# nothing else. The judge is qwen-1.8b for all five, INCLUDING when qwen-1.8b is
# itself the target; that cell is cross-checked against $R3_JUDGE_ALT afterwards.
# DECIDED 2026-09-01 by Jeremiah, ~16 h before the 5-page deadline: four
# targets, three families (Qwen 14b/7b, Yi, Gemma). qwen-1.8b is dropped as a
# TARGET -- it is the fastest model so this saves only ~20 min, but it removes
# the one cell where judge == target, so the self-judge cross-check below is no
# longer needed either. qwen-1.8b remains the JUDGE for all four.
#
# ORDER MATTERS FOR THE FALLBACK. Each model runs generate -> judge -> extract
# -> toggle to completion before the next starts, so stopping the queue at any
# point leaves COMPLETE results for every finished model rather than four
# half-done ones.
# The order is qwen-14b, yi-6b, gemma-2b, qwen-7b: BY FAMILY, not by size.
# qwen-14b first because it is the strongest model and the one most likely to
# produce testable buckets; then a different family at each step. Stopping
# after three leaves Qwen + Yi + Gemma -- the cross-family replication that
# notes/13 sec1 asks for and the part of this study that is genuinely ours.
# Under the old size order, stopping after three left two Qwens and one Yi.
#
for M in qwen-14b yi-6b gemma-2b qwen-7b; do
    run "R3a_generate_${M}"         python3 -m scripts.run3_behavioural_contrast generate             --model "$M" --capture-index -1 --n-per-category 400 --n-control 100             --out "runs/r3_behavioural_${M}"

    run "R3b_judge_${M}"         python3 -m scripts.run3_behavioural_contrast judge             --out "runs/r3_behavioural_${M}"             --model "$M"             --judge-backend openai --judge-model "$R3_JUDGE_MODEL" --judge-swapped

    run "R3c_extract_${M}"         python3 -m scripts.run3_behavioural_contrast extract             --out "runs/r3_behavioural_${M}" --n-splits 400 --require-judge

    # Phase 3 — the toggle test, each vector on its own category.
    run "R3d_toggle_${M}"         python3 -m scripts.run3_behavioural_contrast steer             --model "$M" --out "runs/r3_behavioural_${M}"             --judge-backend openai --judge-model "$R3_JUDGE_MODEL"             --alphas 0.5 1.0
done

# --- Self-judge cross-check: NOT NEEDED at the current model list ----------
# It existed because qwen-1.8b was both a target and the judge. With qwen-1.8b
# dropped as a target, no cell is self-judged and there is nothing to cross-
# check. GUARDED rather than deleted: put qwen-1.8b back in the list above and
# this runs again automatically. Without the guard it would reference a
# directory that never gets created and log a [FAIL] for no reason.
if [ -d "runs/r3_behavioural_qwen-1.8b" ]; then
    run R3f_selfjudge_crosscheck \
        python3 -m scripts.run3_behavioural_contrast judge \
            --out runs/r3_behavioural_qwen-1.8b --model qwen-1.8b \
            --judge-backend openai --judge-model "$R3_JUDGE_MODEL" \
            --labels-out runs/r3_behavioural_qwen-1.8b/judge_labels_alt.jsonl
else
    note "self-judge cross-check: skipped (qwen-1.8b is not a target)"
fi
note ""

# Phase 4.1 — cross-application, every vector onto every category.
#
# SET RUN_R3E=0 TO DROP IT. The arithmetic, at the live 5-model target list:
#
#   generate   400x10x2 + 100x10, x5 models       45,000 generations
#   R3d toggle 10 cells x 17 x 120, x5 models    102,000
#   R3e cross  100 cells x 9 x 80                 72,000
#                                                -------
#                                                219,000  (+ a judge pass each)
#
# TransformerLens generate is not a fast serving path. At ~200 tok/s aggregate on
# a 14B and 48 new tokens that is ~4 gen/s, so generation alone is 15-25 h against
# a ~20 h usable window. WITHOUT R3e: 147,000 generations, ~11-16 h.
#
# R3e is last by design precisely so it is the droppable one, and it re-covers the
# diagonal cells R3d already did at different alphas. Phase 3 (the toggle test)
# gates Phase 4.1 anyway: cross-application is only meaningful once a vector is
# shown causal on its own category.
RUN_R3E=0

if [ "$RUN_R3E" = "1" ]; then
run R3e_cross_application_qwen-14b     python3 -m scripts.run3_behavioural_contrast steer         --model qwen-14b --out runs/r3_behavioural_qwen-14b         --judge-backend openai --judge-model "$R3_JUDGE_MODEL"         --apply-to Age Disability_status Gender_identity Nationality                    Physical_appearance Race_ethnicity Race_x_SES Race_x_gender                    Religion Sexual_orientation         --alphas 0.5 1.0 --n-eval 80
else
    note "R3e cross-application: SKIPPED (RUN_R3E=0) — Phase 4.1 not measured"
fi

say "R3 READ THIS"
python3 - <<'PY' | tee -a "$SUMMARY"
import json, pathlib
for MODEL in ("qwen-14b", "qwen-7b", "yi-6b", "gemma-2b", "qwen-1.8b"):
  p = pathlib.Path(f"runs/r3_behavioural_{MODEL}/report_behavioural.json")
  print("")
  print(f"  --- {MODEL} ---")
  if not p.exists():
    print("  no report — see the logs.")
  else:
    r = json.loads(p.read_text(encoding="utf-8"))
    pc = r.get("per_category", {})
    testable = [k for k, v in pc.items() if v.get("buckets", {}).get("status") == "TESTABLE"]
    repro = [k for k in testable if pc[k].get("reproduces") == "YES"]
    print(f"  {len(testable)}/{len(pc)} categories testable; {len(repro)} reproduce")
    cm = r.get("cosine_matrix", {})
    print(f"  cross-category median |cos|: {cm.get('median_offdiagonal')}")
    sv = r.get("cross_category_survives_refusal_removal")
    if sv:
        print(f"  refusal de-coupling: {sv['verdict']}  "
              f"(raw {sv['median_offdiagonal_raw']:+.3f} -> "
              f"orth {sv['median_offdiagonal_orthogonalised']:+.3f})")
    else:
        print("  *** refusal de-coupling NOT RUN — the cross-category number")
        print("  *** cannot be read as a bias result. Pass --refusal-direction.")
PY
note ""


# COLLECT, FIRST PASS. run 3's artifacts into one folder as soon as R3 is done,
# rather than only at the very end behind ~4 h of P0-P3 that the header already
# expects to run out of window. A second pass runs at the end so a full queue
# re-collects everything.
run COLLECT_r3_only     python3 -m scripts.collect_run3 --out "results/run3_$(date +%Y-%m-%d)"

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
# COLLECT — everything into one folder, per the standing rule that every piece
# of data is saved and a finished experiment lives somewhere a reader can open
# without knowing where anything was originally written.
#
# Residual tensors are too large for GitHub, so they are recorded by manifest
# (shape, size, sha256) rather than dropped: the collection stays complete and a
# copy fetched from external storage can be verified as the one used here.
# --------------------------------------------------------------------------- #
run COLLECT_results     python3 -m scripts.collect_run3 --out "results/run3_$(date +%Y-%m-%d)"

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
note "  1. git push"
note "  2. FROM THE LAPTOP:  powershell -File sync_from_box.ps1 -BoxIp <IP>"
note "     Pulls every artifact off the box and verifies it: non-empty, JSON"
note "     parses, .npy headers valid. The residual arrays are gitignored (GB),"
note "     so this is the ONLY thing standing between you and defect S5."
note "  3. read $SUMMARY"
note "  4. ONLY THEN terminate the instance from the Lambda console. It is not"
note "     wiped automatically and it bills until you stop it. Do not terminate"
note "     until the sync reports VERIFIED with zero problems (notes/14 sec3)."
cat "$SUMMARY"
