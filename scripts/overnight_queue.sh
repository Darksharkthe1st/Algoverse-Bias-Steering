#!/usr/bin/env bash
# Overnight queue for the bias-taxonomy hardening runs (WP-43 P0-P3).
#
#   bash scripts/overnight_queue.sh 2>&1 | tee runs/_logs/overnight.log
#
# Start it, watch it clear preflight, go to bed. Every run is independent, so
# one failing does not stop the rest -- but nothing starts at all if preflight
# fails, because spending a night on a broken environment is the one outcome
# worth preventing.
#
# Total wall clock ~3-4 h. Everything here is Edward's docs/HANDOFF_GPU_HARDENING.md
# plus the P3 split; nothing new is decided while it runs.

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
note "IN THE MORNING, IN THIS ORDER:"
note "  1. git push          — the box can be reclaimed at any time"
note "  2. upload runs/_residuals/ to Drive or HF — NOT committed, ~2.5 GB,"
note "     and losing it turns every follow-up back into a rental"
note "  3. read $SUMMARY"
cat "$SUMMARY"
