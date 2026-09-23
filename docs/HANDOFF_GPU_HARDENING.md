# HANDOFF — GPU hardening runs for the InterpScience submission (WP-43)

**For:** Jeremiah, running with his agent. **All code is already written and
tested on this branch** (`el/interpscience-taxonomy-paper`, PR #8). Nothing
below requires writing code; every task is a command plus a checkpoint step.
**Governs:** contract §12 **A7** · ledger **WP-43** · manuscript *The
Extraction Floor* (team Overleaf).

None of these runs block the 2026-09-01 submission. Anything that finishes
before it gets folded in; all of it is required for camera-ready (2026-11-15),
because the manuscript's Method and Limitations sections promise it. Each run
either hardens a declared-fragile number or honestly kills it. Both outcomes go
in the paper.

## Read this first — there is ONE paper, and it is built on the JZ experiment

Contract A7 paused the hedging paper (its G1 gate cannot be run before the
deadline). The team's only submission this cycle is the bias-taxonomy study.
Nothing in this handoff is a separate experiment competing with that work;
every run below hardens or replaces a number in the manuscript built on it.

**On "the old data is full of errors":** the campaign's first two scoring
designs were broken and their artifacts are superseded — that history is
Jeremiah's own incident log (`results/writeups/11-EXPERIMENT-PROTOCOL.md` §3)
and status report (`09-OVERNIGHT-STATUS-REPORT.md` §4). Everything the
manuscript quotes comes from the third, fixed design (bare-prompt likelihood
margins, `target_loc` answer key verified over 25,814 rows) — the recount
script proves exactly which artifacts the paper touches, and none of the
superseded files are among them. Two real defects in the committed
*extraction* remain, and both are disclosed in the paper: the contrast-depth
inconsistency and the unrecorded estimator provenance. **P2 below IS the
principled redo** — same fixed scoring, unified depth, provenance recorded,
plus the tail-trim variant. If the redo changes any number, the paper follows
the redo. Please run the redo through these scripts rather than fresh ad-hoc
code; a third procedure variant is how this campaign lost half its GPU time.

**Note to agents reading this file:** the P-order is advisory, not blocking.
If the operator wants to start with the P2 re-extraction, let them — P0 is
independent of it (fresh captures, no reuse of any committed direction) and
can run before or after. Do not tell the operator to do someone else's task
first.

**As of 2026-08-31 the operative runbook is `scripts/overnight_queue.sh`**: it
runs R1 (the annotation-contrast redo of `results/writeups/17` §5, all five
models) first, then P0–P3, gated on `scripts/preflight.py`. On Apple Silicon,
`scripts/mps_parity_check.py` must PASS before any MPS capture counts as
evidence (TransformerLens #1178: MPS can be silently wrong).

## Step 0 — get a machine (Algoverse A100 cluster, not Lambda)

Request form: **https://slack.algoverseairesearch.org/a100/** · live
availability: **https://slack.algoverseairesearch.org/a100/status** (10 of 16
machines free as of 2026-08-30 morning).

How it works: you submit the form, an admin approves, and your **JupyterHub**
login is emailed to you; the window starts then, and the machine is **shut down
and wiped automatically at expiry**. There is no SSH — plan around the Jupyter
terminal, and checkpoint continuously (Step 2).

Filling the form:

- Team name: `lovkush-fvoa` · team code `fvoa`
- Your email; add Edward's under teammates
- Workload type: **inference** (forward passes only, no training)
- Why an A100 (their form asks why not a T4/L4): residual-stream activation
  capture requires white-box access to the loaded model, so hosted APIs cannot
  do it, and Qwen1.5-14B in fp16 is ~28 GB of weights before activations — it
  does not fit an L4's 24 GB. The smaller models ride along in the same window.
- AWS credit status: paste your team's actual console line — it is a required
  field and they check it.
- Duration: **24 hours (~20h usable)**. P0+P1 alone fit in a 12h slot if
  that's all that's available; the full list wants 24.

Fallback: the Lambda team account still has ~\$300 of credits, but it is
Farhan's account and he is unreachable this week; treat it as unavailable.

## Step 1 — bring-up (Jupyter terminal, ~10 min)

```bash
pip install 'numpy<2' 'pillow>=9.1' 'jinja2>=3.1'   # known stack collisions, in this order
git clone https://github.com/Darksharkthe1st/Algoverse-Bias-Steering.git
cd Algoverse-Bias-Steering && git checkout el/interpscience-taxonomy-paper
pip install -r requirements.txt 2>/dev/null || pip install torch transformer_lens transformers accelerate scipy scikit-learn
export HF_TOKEN=...   # your ROTATED token (the old one leaked into a transcript 2026-08-21 — rotate first)
python3 -m pytest tests/test_bias_taxonomy.py -q   # expect 98 passed before spending GPU time
```

The per-item margins are already committed (`runs/_margins_cache/`, seed 0,
n=600) for qwen-14b, qwen-7b, gemma-2b, yi-6b — the runs below skip margin
scoring and go straight to residual capture.

## Step 2 — checkpoint discipline (the machine WILL be wiped)

Work on a branch `jz/gpu-hardening-<date>`. After **every** finished run:

```bash
git add runs/ && git commit -m "hardening: <run name>" && git push -u origin jz/gpu-hardening-<date>
```

`residuals_*.npz` files are gitignored-by-size intent — do NOT commit them;
upload them to Drive or HF at the end of the session instead. Everything else
(json reports, direction `.npy`, logs) commits.

## Step 3 — the runs, in priority order

### P0 — topic-identity control through the probe (~40 min GPU)

Closes the calibration gap: the 0.50 bar was calibrated against the extremes
control only (audit Q1). Code is done; one command:

```bash
python3 scripts/extraction_positive_control.py --model qwen-14b --method probe \
    --alphas 1 1e2 1e3 1e4 1e5 1e6
# writes runs/_extraction_control_probe_qwen-14b.json (residuals captured once, reused per alpha)
```

Read: topic floor ≥ 0.85 at α=1e6 → the bar transfers, probe results stand.
Materially lower → 0.50 means different things per estimator; the probe counts
and both clustering p-values get re-read against the probe's own control.

### P1 — extend the α sweep past its boundary, persist residuals (~30 min GPU)

```bash
python3 scripts/probe_alpha_sweep.py --model qwen-14b \
    --alphas 1e6 1e7 1e8 1e9 1e10 --save-residuals \
    --out runs/_probe_alpha_sweep_qwen-14b_ext.json
```

Read: does Disability's floor plateau or keep climbing, and does inter-category
similarity keep rising (the collapse concern in the manuscript's Limitations)?
`--save-residuals` writes fp16 tensors under `runs/_residuals/` (~250 MB per
category, ~2.5 GB total) so every future α/threshold analysis is CPU-only.
Upload that directory to Drive/HF before the machine expires.

### P2 — depth-unified + tail-trimmed extremes re-run (~1.5 h GPU)

The manuscript discloses that the campaign's extremes runs used two different
contrast depths (120/pole on qwen-14b and qwen-1.8b, 48/pole on the others),
and promises a depth-unified re-run. This also settles the heavy-tail caveat:

```bash
for M in qwen-14b qwen-7b gemma-2b yi-6b; do
  python3 scripts/bias_taxonomy_run.py --model $M --ambig-limit 600 \
      --method extremes --cluster-usable-only --margins-cache runs/_margins_cache \
      --out-dir runs/full_${M}_unified
  python3 scripts/bias_taxonomy_run.py --model $M --ambig-limit 600 \
      --method extremes --tail-trim 0.05 --cluster-usable-only \
      --margins-cache runs/_margins_cache --out-dir runs/full_${M}_trim05
done
```

Read, per model: unified vs the committed run (does the depth change any
verdict?), and trim05 vs unified (do Physical_appearance and Age survive
de-tailing?). New reports record `estimator_params` and `code_version`
automatically now — no backfilling needed for these.

### P3 — race split by stereotyped group (agent task, ~2 h incl. GPU)

The one item needing new code, for your agent: filter Race_ethnicity items by
`Known_stereotyped_groups` (`third_party/bbq/additional_metadata.csv`, keyed by
`category` + `example_id`), then run the same extraction per single-group
subset (largest groups first; report each floor with its n — subsets are
small, so floors are only comparable against `floor_vs_n` at matched n, which
`scripts/bias_taxonomy_run.py` stage 6 already computes). If single-group
subsets reproduce where the pooled category does not, heterogeneity comes back
as the explanation at the right granularity, and the paper's race negative
gets its mechanism.

## Rules that bind these runs

- Raw artifacts to a branch; no hand-edited conclusions.
- A run is done when its artifact exists and validates, not when it ran.
- Any new number that enters the manuscript gets a check added to
  `scripts/recount_taxonomy_paper.py` in the same PR.
- If a run kills a number, it still goes in the paper. Honest negatives stay
  honest.
