# Implementation plan — judge v2.1 + 3-vector steering (qwen3-8b)

Status: DRAFT / calibration. Built pre-emptively; expect the rubric to shift as
the κ pass reveals where it breaks (that is the point — see `docs/RUBRIC_v2.md`).

Each phase ends in a **concrete observable** — an artifact or a printed table you
can inspect before the next phase runs. Nothing downstream is gated on the κ
number being confirmed; we build the whole chain and adjust.

Canonical pieces this reuses (no new machinery): `JudgeSpec` (`config.py:87`),
`neutrality_judge` (`judge.py:61`), `build_mean_difference` (`steering.py:102`),
`apply_resid_pre_add` (`steering.py:125`), `assert_steering_shape`
(`steering.py:30`), `DatasetSpec.train_split` (`config.py:63`).

Target model: **qwen3-8b** (matches the recent `runs/…_qwen3-8b` extraction runs).

---

## Phase 0 — Calibration data

**0a. 40-prompt set assembled.** ✅ done.
- Observable: `datasets/Calibration/calibration_v2_prompts.csv` — 20 old
  (comparison_200 / BBQ / Do_Not_Answer) + 20 new (issuebench / axbench),
  disjoint from cal_001–030.

**0b. qwen3-8b baseline responses on the 40 prompts.** → scoped to a branch by
the inference-handoff agent; run on GPU in parallel.
- Observable: a labeling-ready sheet `item_id, prompt, response` (one row per
  prompt), plus the raw run under `runs/…`.
- Decision recorded in the handoff: greedy decode, max_new_tokens, and whether
  qwen3-8b `<think>` traces are stripped from the judged OUTPUT.

**0c. Human + judge labels → κ on the 6-way collapse.**
- Humans label the 40 (fine 9-way ok; collapse to 6 for scoring). Judge v2.1
  labels the same 40.
- Observable: per-category Cohen's κ table (human↔human and judge↔human) +
  the 6×6 confusion matrix. Gate target κ ≥ 0.70 — **informational, not
  blocking**; low κ tells us which rubric rows to fix, not to stop.

---

## Phase 1 — Judge implementation

**1a. Wire the 6-label JudgeSpec.**
- `JudgeSpec(name="neutrality", model=…, labels=[…6…], rubric=<judge_v2.1 body>)`
  from `docs/judges/judge_v2.1.md`. Pin `seed`, `temperature=0`, and the
  judge-file commit hash.
- Observable: judge run over the 40 → a `verdict` column; spot-check it parses
  (no unexpected `nonsense`/UNMATCHED).

**1b. Judge validity readout.**
- Observable: judge↔human 6×6 confusion + per-class κ (this is 0c's judge row,
  surfaced as the deliverable that "the judge works").

---

## Phase 2 — Baseline generation + bucketing for vectors

Run on the **vector pool** (the real steering dataset, NOT the 40 calibration
items — those stay held out), split via `DatasetSpec.train_split`.

**2a. Generate baseline responses on the train split; capture residuals.**
- `capture_mean` over each response → `(n_layers, d_model)` per example.

**2b. Judge the baseline responses; bucket residuals by verdict.**
- Observable: **per-label count table** (how many train examples landed in each
  of the 6 labels), with the **group-size floor** flagged.
- Floor rule: each contrast pole needs ≥ `N_floor` examples (start `N_floor=40`)
  or its mean is too noisy for a clean difference-of-means. If a pole is under
  floor (likely `hard-refusal`), the plan flags it and we either enrich the pool
  with refusal-triggering prompts or drop that vector. **This is a gate.**

---

## Phase 3 — Build the 3 vectors

`build_mean_difference(resids_by_label, contrast)` ×3:

| vector | contrast (pos, neg) | +coeff pushes toward |
|---|---|---|
| V1 | `("soft-refusal", "hard-refusal")` | soft refusal |
| V2 | `("stance", "soft-refusal")` — `stance` = `stance-factual` ∪ `stance-evaluative` | any stance |
| V3 | `("stance", "non-engagement")` | any stance |

- `assert_steering_shape` on each (guards the 2025 1-D DC-offset bug).
- Observable: 3 saved vectors (`.safetensors`), each with logged shape
  `(n_layers, d_model)` and per-layer norm profile.

---

## Phase 4 — Coefficient sweep + evaluation

On the **held-out test split**, for each vector, apply `apply_resid_pre_add` at a
coeff grid spanning both signs (e.g. `[-8, -4, -2, 0, +2, +4, +8]`), re-generate,
re-judge.

- Observable **per vector, per coeff**: the baseline→steered **6×6 label
  transition matrix**, plus two curves vs. coeff:
  1. **target-transition rate** — e.g. for V2, fraction of baseline
     `soft-refusal` items that became a `stance` at that coeff.
  2. **`ignored` rate** — the model-broke guard. The usable operating point is
     where (1) rises while (2) stays flat.
- `coeff=0` row is the internal no-steer baseline.

---

## Phase 5 — Controls (make the effect attributable)

Reuse the remote control examples where possible (same prompts, to save GPU).

- **Random-direction control** at matched per-layer norm — does any push of this
  size move labels, or specifically this direction?
- **System-prompt baseline (AxBench)** — can a prompt alone produce the same
  shift, making the vector uninteresting? (CLAUDE.md §5a.)
- Observable: the Phase-4 curves with control curves overlaid.

---

## Phase 6 — Report

- Observable: per-vector summary — operating coeff, target-transition lift over
  both controls, `ignored`-rate cost, and the full transition matrix.
- Language bar (CLAUDE.md §5): "**a** direction," never "the direction";
  steering success ≠ identifying the representation.

---

## Open decisions to confirm

1. `N_floor` value (default 40) and what to do if `hard-refusal` is under floor
   (enrich pool vs. drop V1).
2. Vector pool identity + size, and the train/test fraction.
3. Layer selection: use the full `(n_layers, d_model)` stack (current
   `apply_resid_pre_add` splits coeff across all layers) or a best-layer probe.
4. Whether qwen3-8b `<think>` traces count toward the judged behavior or are
   stripped first (must match between calibration 0b and the vector pool 2a).
