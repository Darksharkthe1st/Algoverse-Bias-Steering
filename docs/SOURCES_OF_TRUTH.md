# Sources of truth

**One fact, one owner.** This registry says which file *defines* each thing.
Everywhere else may reference it — no file may restate it.

Written 2026-08-07 after a doc audit found the judge rubric specified four
different ways across four files, the venue named in ten, and two people
carrying two competing task documents each. That is the same failure class the
project studies: a disagreement invisible in any single document.

## The registry

| Fact | Owner | Everyone else |
|---|---|---|
| Judge rubric / label classes | **`docs/RUBRIC_v2.md`** | link only, never restate |
| Model set, reasoning modes, TL gotchas | **`docs/MODEL_SET_2026-08-07.md`** | link only |
| Prior art, novelty verdict, **venue choice** | **`docs/PRIOR_ART_2026-08-07.md`** | link only |
| What reproduces; retractions; provenance traps | **`docs/VERIFICATION_2026-08-07.md`** | link only |
| Claim status (certified / retracted / review / open) | **`scripts/build_dashboard.py` → `CLAIMS`** | link to the live ledger |
| Schedule, gates, kill criteria, scope cuts | **`docs/2026-08-02_sprint_proposal.md`** §5, §8 | link only |
| Paper claims, terminology, must-cites | **`PAPER_FRAMING.md`** | link only |
| Per-person work | **`RUNBOOK_<NAME>.md`** | one file per person |
| Rules for agents and contributors | **`CLAUDE.md`** (mirrored to `AGENTS.md`) | see mirror rule below |
| The Aug 6 governed audit | **`docs/REVIVAL_AUDIT.md`** | **frozen** — append notes, never edit |
| 2025 post-mortem + 2026 frontier scan | **`docs/2026-08-01_project_analysis.md`** | link only |
| Measurement-validity teaching | **`docs/THE_CORRECT_PROBLEM.md`** | link only |
| Long-horizon geometry program (parked) | **`docs/RESEARCH_PROGRAM_GEOMETRY.md`** | link only |
| Pre-registered commitments (rubric hash, endpoints, l\*) | **`docs/PREREG.md`** | append amendments, never edit filed sections |
| Intervention recipes / steering toolkit adoption priority | **`docs/STEERING_TOOLKIT_2026.md`** | link only |
| Intervention / control-surface class map (S0–S6, S4 subtypes) | **`docs/INTERVENTION_CLASSES.md`** | link only |
| Future stakes / access-control motivation (non-doctrine) | **`docs/STAKES_SOFT_REFUSAL.md`** | link only; paper text still only from `PAPER_FRAMING.md` |
| Rubric calibration workflow / Saturday decisions | **`docs/RUBRIC_CALIBRATION.md`** | link only; labels still only in `RUBRIC_v2.md` |
| Team norms (humans vs agents, epistemic bar) | **`docs/TEAM_HOW_WE_WORK.md`** | link only |
| Labeling examples / difficulty walkthrough | **`docs/LABELING_WALKTHROUGH.md`** | link only; labels still only in `RUBRIC_v2.md` |
| BASI / community-intel character map (not paper) | **`experiments/community_methods/DOSSIER.md`** | link only; never a paper source or results number |
| RFC 2119 as system-prompt pattern (not paper) | **`experiments/community_methods/RFC2119_SYSTEM_PROMPTS.md`** | link only |
| Personal-site brief derived from harvest (not paper) | **`experiments/community_methods/SITE_BRIEF.md`** | hand to a site agent; **do not** upload the dossier |
| Spiralism / parasitic-AI harvest notes (00–08; not paper) | **`experiments/community_methods/notebook/`** (index: `README.md`) | link only; never a paper source; do not patch `notebooklm/` |
| Four-spirals landscape essay + infection literature (not paper) | **Blueprint vault** `research/four-spirals-field-guide.md` and `research/one-untrusted-agent.md`; public cuts at **`edward-lcl.github.io`** `/questions/do-not-collapse-the-spirals/` and `/questions/one-untrusted-agent-is-enough/` | Algoverse `notebook/09` and `10` are stubs; do not restate the essay here |

## Derived — never edit by hand

- `dashboard/index.html` — built by `scripts/build_dashboard.py`. Edit the
  narrative constants at the top of the script, rebuild, redeploy.
- `notebooklm/*.md` — regenerated from the *paper* owners above. If one is
  wrong, fix the owner and regenerate; do not patch the pack. This is
  **not** `experiments/community_methods/notebook/` (harvest notes 00–08)
  and not the four-spirals essay, which lives in the Blueprint vault.
- `AGENTS.md` — a byte-for-byte mirror of `CLAUDE.md`, because different agent
  stacks read different filenames. **Mirror rule:** change `CLAUDE.md`, then
  `cp CLAUDE.md AGENTS.md` in the same commit. Never edit `AGENTS.md` directly.

## Superseded — kept as pointer stubs

These still exist so old links resolve, but hold no content:

| Stub | Superseded by |
|---|---|
| `docs/2026-08-01_sprint_plan.md` | `docs/2026-08-02_sprint_proposal.md` |
| `docs/2026-08-01_venue_scan.md` | `docs/PRIOR_ART_2026-08-07.md` |
| `HANDOFF_FARHAN.md` | `RUNBOOK_FARHAN.md` |
| `HANDOFF_JEREMIAH.md` | `RUNBOOK_JEREMIAH.md` |

## Rules

1. **Before writing a fact, check this table.** If a file already owns it, link
   instead of restating. Restating is how four rubrics happened.
2. **Numbers and dates live in exactly one file.** If you need a number
   elsewhere, cite the owner's path next to it so a reader can check.
3. **Changing an owned fact = editing the owner, in a PR, plus the claim
   ledger if it changes what we assert.** Same PR, not a follow-up.
4. **Superseding beats editing for audit docs.** `REVIVAL_AUDIT.md` and
   anything else marked frozen keeps its original text; add a dated note above
   the affected section instead.
5. **A new doc must claim a row here or it does not get created.** If it does
   not own a fact, it belongs inside an existing owner.
6. **Stubs are not content.** If you find yourself adding substance to a stub,
   you are recreating the duplication — put it in the owner.

## Known intentional duplication

Only one: `CLAUDE.md` / `AGENTS.md`, under the mirror rule above. Everything
else that looks duplicated is a bug in this registry — report it.
