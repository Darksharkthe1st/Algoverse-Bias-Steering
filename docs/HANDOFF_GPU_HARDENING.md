# HANDOFF — GPU hardening runs for the InterpScience submission (WP-43)

**For:** Jeremiah (holds Lambda team-account access; SSH key `~/.ssh/lambda_jeremiah`)
**Governs:** contract §12 **A7** · ledger **WP-43** · manuscript: *The Extraction Floor*
**Budget:** ~\$300 Lambda credits remain (per team channel 2026-08-21). Everything
below fits in well under \$40 at \$2/hr on 1× A100 40GB.

None of these runs block the 2026-09-01 submission. Any that finish before it
get folded in; all of them are required for camera-ready (2026-11-15), because
the manuscript's Limitations section promises them. Each run either hardens a
declared-fragile number or honestly kills it. Both outcomes go in the paper.

## Box bring-up (from the 2026-08-21 session notes, verified then)

1x A100 40GB SXM4, Lambda Stack 22.04, no filesystem, default firewall. Avoid
the GH200 (ARM64). Then, in this order, before anything else:

```bash
pip install 'numpy<2'        # Lambda Stack torch is compiled against numpy 1.x
pip install 'pillow>=9.1'    # system PIL predates Image.Resampling
pip install 'jinja2>=3.1'    # apply_chat_template needs it
```

Clone the repo, check out `el/interpscience-taxonomy-paper` (carries the run
artifacts and the recount script). `export HF_TOKEN=...` with your **rotated**
token (the old one leaked into a transcript on 2026-08-21 — rotate first at
huggingface.co/settings/tokens if you have not already).

## The four runs, in priority order

### P0 — topic-identity control through the probe (closes the calibration gap)

The 0.50 usability bar was calibrated against the extremes estimator only.
Until the topic control runs through the probe, every probe-derived count and
both clustering p-values are provisional (manuscript §6; audit Q1).

`scripts/extraction_positive_control.py` currently hard-codes the extremes
path. Add `--method probe --alpha A` passthrough mirroring `_extract()` in
`scripts/bias_taxonomy_run.py` (~20 LOC), then:

```bash
python scripts/extraction_positive_control.py --model qwen-14b --method probe \
    --alpha 1e6 --out runs/_extraction_control_probe_qwen14b_a1e6.json
# repeat for alpha in 1 1e2 1e3 1e4 1e5 1e6 (residuals are re-captured per call;
# if time is tight, 1e6 alone answers the load-bearing question)
```

Read: if the topic control reproduces ≥ 0.85 under the probe at 1e6, the 0.50
bar transfers and the probe results stand. If it lands materially lower, 0.50
means different things per estimator and the paper's probe numbers get
re-thresholded against the probe's own control.

### P1 — extend the α sweep past its boundary, and persist residuals

qwen-14b's optimum sat at the sweep boundary (1e6) and was still climbing.

```bash
python scripts/probe_alpha_sweep.py --model qwen-14b \
    --alphas 1e6 1e7 1e8 1e9 1e10 --out runs/_probe_alpha_sweep_qwen-14b_ext.json
```

Also persist the captured residual tensors this time (one `.npz` per category,
`(n, L, d)`), so every future analysis in this family is CPU-only. That is a
small change in `bbq_score` residual capture: `np.savez_compressed` next to the
margins cache. 600 × 40 × 5120 fp16 ≈ 250 MB per category; keep them on the box
or push to a scratch bucket, do not commit them.

### P2 — winsorised re-extraction (the heavy-tail check)

Physical_appearance and Age on qwen-14b have excess kurtosis ≈ +3.9; the top 5%
of items carry about half the variance, and the extremes contrast selects
exactly those tails. Clip margins at the 5th/95th percentile before the
quintile split (small change in `_extract()`), re-run:

```bash
python scripts/bias_taxonomy_run.py --model qwen-14b --method extremes \
    --winsorise 0.05 --out-dir runs/full_qwen14b_winsorised
```

Read: floors survive → the two heavy-tailed positives are solid. Floors
collapse → the paper's Table 2 keeps them but the caveat becomes a finding.

### P3 — race split by stereotyped group (the most interesting open lead)

Race_ethnicity pools nine distinct stereotyped-group sets. Extract a direction
for single-group subsets (e.g. Black-targeted items only) through the same
pipeline. If single-group subsets reproduce where the pooled category does not,
heterogeneity returns as the explanation at the right granularity, and the
paper's negative gets its mechanism. Requires a dataset filter on
`Known_stereotyped_groups` (see `third_party/bbq/additional_metadata.csv`);
mind the smaller n per subset — report the floor's n alongside it.

## Rules that bind these runs

- Push run dirs to a branch as raw artifacts, no hand-edited conclusions.
- Record `method`, `alpha`, seeds and denominators in the run dir (the report
  schema backfill is WP-44; new runs must not need backfilling).
- A run is done when its artifact exists and validates, not when it ran.
- Update `scripts/recount_taxonomy_paper.py` in the same PR that quotes any new
  number in the manuscript.
