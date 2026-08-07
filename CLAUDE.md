# Algoverse-Bias-Steering — agent instructions

Soft-refusal steering: we study whether "does the model take a side at all"
(opinionation vs neutrality on controversial-but-not-harmful prompts) is a
measurable, steerable representation in open-weight LLMs, and how it relates to
the Arditi hard-refusal direction. Revival sprint, Aug 2026; 2025 assets
(pipeline + 244 logged runs) are inherited in `experiments/`.

## Non-negotiables for ALL agents and contributors

1. **Framing is centralized.** Before writing or editing ANY paper text, related
   work, abstract, summary, or review response, read `PAPER_FRAMING.md` and
   follow it exactly. If you disagree, PR that file — do not introduce a
   different framing in a draft, comment, or commit message.
2. **Numbers trace to artifacts — and recount from TEXT LOGS, not pickles.**
   Any number must come from a committed artifact under `experiments/` or be
   flagged as pending; cite the path in the PR description. Historical CSVs are
   **UNTRUSTED comparators** even when a recount agrees with them. Use
   `src/textlog_parse.py` + `src/recount.py`: the archived `.pkl` files are
   *cumulative* (log N holds every model 1..N appended; only the final 96
   records belong to the model in the filename), and unpickling them executes
   arbitrary code. Denominator is **n = 96 per arm**, and the arrow-named CSV
   columns (`Init->Opin`) are **per-arm label marginals, NOT transitions**.
   Judge `none`/`NONE` markers are extraction failures — never fold them into
   "nonsense" or any behavior class.
3. **The judge is part of the method.** Every judged number carries a judge
   version. Judge v1 (2025, GPT-4o-mini binary rubric) is RETIRED — its rubric
   scored factual decisiveness as opinionation; treat all v1 percentages as
   provisional and never mix judge versions in one table. Any rubric or model
   change = new judge version, documented in `docs/judges/`.
4. **Steering-claim hygiene (2026 bar).** No steering result is reportable
   without (a) a system-prompt baseline on the same prompts (AxBench), (b)
   per-example distributions — the 3×3 judge confusion counts, not just means —
   and (c) for any headline intervention, side-effect audits (capability +
   safety). Say "a direction", never "the direction" — steering success does
   not identify the representation (non-identifiability, arXiv:2602.06801).
5. **Honest negatives stay honest — but an invalid run is not a negative.**
   The 2025 CrowS-Pairs transfer failure is load-bearing motivation; do not
   soften it and do not overclaim it. The refusal↔opinion cross-application is
   **RETRACTED as invalid, not null**: the archived refusal `.pt` files are 1-D
   hidden-width tensors, and `steering_vector[layer]` on a 1-D tensor yields a
   *scalar* broadcast across the residual width (a DC offset, not a direction),
   compounded by a model/`vector_files` ordering mismatch. See
   `docs/REVIVAL_AUDIT.md`. Never cite it as evidence in either direction.
   Before shipping ANY intervention, assert tensor shape against
   (n_layers, d_model) — this class of bug is silent.
6. **Push runs to branches** as raw CSVs + pickles under `experiments/`,
   following the existing `Log_N_*` convention. No hand-edited conclusions.
7. **Open-weight models only.** Steering needs residual-stream access.

## Key docs

- `PAPER_FRAMING.md` — canonical framing, terminology, must-cites
- `docs/2026-08-01_project_analysis.md` — post-mortem + 2025–26 frontier scan
- `HANDOFF_<NAME>.md` — per-person marching orders
- Dashboard: `scripts/build_dashboard.py` (narrative constants at the top) →
  `dashboard/index.html`, CI-rebuilt on push; update constants via the
  `dashboard-update` skill in `.claude/skills/`.
