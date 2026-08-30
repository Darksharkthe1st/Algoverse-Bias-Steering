# Edward (el) — task list

**Source:** `Algoverse — 2 Wk Plan` → *"Different Techniques for Bias Steering"* (Edward).
**Thesis to defend:** which *extraction/injection technique* steers bias best, and how each
compares to the refusal direction.

> **Read before starting:** `PROJECT_STATE.md`, `AGENTS.md`, `RUNBOOK_EDWARD.md`,
> `RESEARCH_CONTRACT.md` §12. Frozen project (2026-08-17). §N refs →
> `docs/superseded/needed-experiments.md`. The old `STEERING_TOOLKIT_2026` framing is
> **superseded** — reuse its ideas, don't cite it as doctrine.

---

## EL-0 — Lock the injection convention FIRST  (§0.1 — blocks every comparison below)

No technique comparison is valid until injection is standardized. Three incompatible
conventions exist in the archive (raw-per-layer / `coeff/n_layers` / per-layer unit-normalized).
- **Do:** pick ONE canonical rule (recommendation in §0.1: unit-normalize each layer's vector,
  single `coeff` at a chosen layer band, report `coeff` in normalized units) and re-run the
  coeff sweep under each convention on 2 models to justify the pick (smoothest, most monotonic
  dose-response wins).
- **DoE:** the decision is recorded (which rule, why, with the dose-response evidence) and
  every EL task below uses it. This is the precondition for a fair "which technique is best."

## EL-1 — Affine / Cone / Gradient steering

- **Do:** implement each technique against the existing harness (`src/bias_steer`,
  `configs/*.py`) so they are drop-in comparable to difference-of-means additive steering:
  - **Affine** — affine (not just additive) concept editing (the "ACE" family Aryaman is also
    touching — coordinate with AA-6 so there's one implementation, not two).
  - **Cone** — steer within a cone around the direction rather than a single ray.
  - **Gradient** — gradient-based / optimized steering vector.
- **⚠️ Doctrine check:** ACE/cone/gradient methods are listed under `PROJECT_STATE.md` §"Does
  not block the paper" — this is **exploratory** work, not a submission dependency. Scope it as
  a method study; entering the paper needs a §12 amendment.
- **Hygiene:** every technique asserts vector shape `(n_layers, d_model)` before injection
  (Log-213 class bug, `AGENTS.md` §6); each carries a system-prompt baseline (AxBench) and
  per-example 3×3 distributions (`AGENTS.md` §5).
- **DoE:** each method runs end-to-end on ≥2 models and produces a dose-response + coherence
  curve on the same eval set under the EL-0 convention.

## EL-2 — Compare these vectors to refusal vectors

- **Do:** for each technique's bias vector, compute per-layer cosine vs the native refusal
  vector (from Farhan FK-2) and vs difference-of-means additive. Does a fancier technique find
  a *different* direction, or the same direction reached differently?
- **DoE:** a technique×{refusal, mean-diff} cosine table + null floor; note whether any
  technique diverges geometrically from plain mean-diff.

## EL-3 — Which technique is most effective?

- **Do:** head-to-head on ONE fixed held-out eval set, matched coeff (EL-0), with the coherence
  gate (§0.3): rank techniques by debias Δ / induce Δ *subject to staying coherent* (not just
  raw label flips — the gemma inverted-U in §7 shows why coherence must be a first-class axis).
- **DoE:** a technique-ranking table (effect + usable-coefficient window width + coherence),
  one config, n≥100, coherence logged. "Best" = max effect subject to the coherence gate.

---

## EL-4 — Label the data Edward sent  (team task, EL is the owner/originator)

- **Do:** get the dataset Edward contributed into the pipeline — define the label schema
  (align to JZ-1's data map and the frozen rubric, **do not** invent a new construct), run the
  labeling, and commit it under `datasets/` following the existing layout.
- **⚠️ Depends on:** the "Soft Refusal" definition (see `needed-experiments` team section) —
  **but** that term is **retired** (`AGENTS.md` rule 5). Label against *hedging /
  over-abstention-on-answerable* + stereotype-alignment, not "soft refusal", unless a §12
  amendment revives the term.
- **DoE:** labeled dataset committed with a documented schema + inter-annotator agreement if
  ≥2 labelers (κ, cf. §4 / `scripts/kappa_from_csv.py`).

---

### Related `needed-experiments` Edward can own
- **§6** vector-normalization ablation (raw vs per-layer unit-norm) — this basically *is* EL-0,
  formalize it as the deliverable. LOW–MED.
- **§7** coefficient / layer-placement ablation — feeds EL-3's "usable window" axis. MED.

### ⚠️ Doctrine checks for EL's section
- The whole technique survey is **exploratory** (does-not-block scope). Great as a method
  contribution, but it is not on the frozen paper's critical path — keep it from expanding the
  submission scope (`AGENTS.md` rule 2).
- Coordinate ACE with Aryaman (AA-6) — one implementation.
