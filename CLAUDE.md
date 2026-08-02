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
2. **Numbers trace to artifacts.** Any number in the paper or dashboard must
   come from a committed file under `experiments/` (CSV or pickle), or be
   flagged as pending. Cite the artifact path in the PR description.
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
5. **Honest negatives stay honest.** The 2025 CrowS-Pairs transfer failure and
   the failed refusal↔opinion cross-application are load-bearing motivation —
   do not soften them and do not overclaim them.
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
