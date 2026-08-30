# Algoverse-Bias-Steering — agent instructions

## READ THIS FIRST — the control plane

**The project is FROZEN as of 2026-08-17.** Four files are canonical. Nothing
else governs, including anything under `docs/superseded/`.

| File | Owns |
|---|---|
| **`PROJECT_STATE.md`** | current paper, gate, blockers, next actions. **Start here.** |
| **`RESEARCH_CONTRACT.md`** | the frozen science. Changing it needs a dated amendment. |
| **`WORK_LEDGER.md`** | execution packages, evidence-of-done |
| **`docs/PREREG.md`** | frozen choices fixed before outcomes were inspected |

`DECISION_LOG.md` explains *why* earlier documents no longer apply. It is
history, not doctrine.

**Rules for every agent:**

1. **Do not plan or execute from `docs/superseded/`.** Those files describe cut
   framings — byte-fallback perturbation, a four-claim structure, the eight-way
   rubric, the Qwen-27B trajectory, the steering-technique survey. They are kept
   for provenance and are explicitly not current.
2. **Do not expand scope.** `RESEARCH_CONTRACT.md` §12 lists the only five things
   that may reopen the contract. "This would be more interesting" is not one.
3. **Root `RUNBOOK_*` and `HANDOFF_*` are retired stubs (2026-08-20)** — a stale
   runbook misdirected an agent's planning, so they now hold pointers only.
   Work from `WORK_LEDGER.md`, `docs/work-splits/`, and the canonical execution
   handoffs in `docs/` (`HANDOFF_G1.md`, `HANDOFF_WP25.md`). Do not add
   substance to a stub; nothing in one may redefine the paper, a metric, a
   model set, a deadline, or a definition of done.
4. **A task is done when its evidence exists and validates**, not when a report
   says it ran. See `WORK_LEDGER.md`.
5. **Do not coin terminology.** The behaviour is *hedging*; the failure mode is
   *over-abstention on answerable items*. "Soft refusal" is retired.

---

## Historical context (the project this used to be)

Soft-refusal steering: we study whether "does the model take a side at all"
(opinionation vs neutrality on controversial-but-not-harmful prompts) is a
measurable, steerable representation in open-weight LLMs, and how it relates to
the Arditi hard-refusal direction. Revival sprint, Aug 2026; 2025 assets
(pipeline + 244 logged runs) are inherited in `experiments/`.

## Non-negotiables for ALL agents and contributors

1. **One fact, one owner.** `docs/SOURCES_OF_TRUTH.md` says which file
   *defines* each thing — rubric, model set, venue, schedule, claim status.
   Before writing a fact, check the registry: if a file owns it, **link, never
   restate.** Restating is how this repo ended up with the judge rubric
   specified four different ways while every doc called freezing it the most
   time-critical item in the sprint. Changing an owned fact means editing the
   owner, in a PR, plus the claim ledger in the same PR if it changes what we
   assert. A new doc must claim a row in the registry or it does not get made.
2. **Framing is centralized.** Before writing or editing ANY paper text, related
   work, abstract, summary, or review response, read `RESEARCH_CONTRACT.md` and
   follow it exactly. If you disagree, PR that file — do not introduce a
   different framing in a draft, comment, or commit message.
3. **Numbers trace to artifacts — and recount from TEXT LOGS, not pickles.**
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
4. **The judge is part of the method.** Every judged number carries a judge
   version. Judge v1 (2025, GPT-4o-mini binary rubric) is RETIRED — its rubric
   scored factual decisiveness as opinionation; treat all v1 percentages as
   provisional and never mix judge versions in one table. Any rubric or model
   change = new judge version, documented in `docs/judges/`.
5. **Steering-claim hygiene (2026 bar).** No steering result is reportable
   without (a) a system-prompt baseline on the same prompts (AxBench), (b)
   per-example distributions — the 3×3 judge confusion counts, not just means —
   and (c) for any headline intervention, side-effect audits (capability +
   safety). Say "a direction", never "the direction" — steering success does
   not identify the representation (non-identifiability, arXiv:2602.06801).
6. **Honest negatives stay honest — but an invalid run is not a negative.**
   The 2025 CrowS-Pairs transfer failure is load-bearing motivation; do not
   soften it and do not overclaim it. The refusal↔opinion cross-application is
   **RETRACTED as invalid, not null**: the archived refusal `.pt` files are 1-D
   hidden-width tensors, and `steering_vector[layer]` on a 1-D tensor yields a
   *scalar* broadcast across the residual width (a DC offset, not a direction),
   compounded by a model/`vector_files` ordering mismatch. See
   `docs/REVIVAL_AUDIT.md`. Never cite it as evidence in either direction.
   Before shipping ANY intervention, assert tensor shape against
   (n_layers, d_model) — this class of bug is silent.
7. **Push runs to branches** as raw CSVs + pickles under `experiments/`,
   following the existing `Log_N_*` convention. No hand-edited conclusions.
8. **Open-weight models only.** Steering needs residual-stream access.

## Key docs

- `docs/SOURCES_OF_TRUTH.md` — **the fact-ownership registry; read first**
- `RESEARCH_CONTRACT.md` — canonical framing, terminology, must-cites
- `docs/2026-08-01_project_analysis.md` — post-mortem + 2025–26 frontier scan
- `docs/work-splits/<xx>-task-list.md` — per-person context · execution
  handoffs: `docs/HANDOFF_G1.md`, `docs/HANDOFF_WP25.md` (root `RUNBOOK_*`
  are retired stubs)
- Dashboard: `scripts/build_dashboard.py` (narrative constants at the top) →
  `dashboard/index.html`, CI-rebuilt on push; update constants via the
  `dashboard-update` skill in `.claude/skills/`.
