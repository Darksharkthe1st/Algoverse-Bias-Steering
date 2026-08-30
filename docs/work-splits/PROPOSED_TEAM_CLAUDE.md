# PROPOSED team CLAUDE.md — DRAFT for review

*Draft for the four of us (Farhan, Edward, Jeremiah, Aryaman) to review, not yet
canonical.* It is a rewrite of the current root `CLAUDE.md`, folding in
`docs/TEAM_HOW_WE_WORK.md`, `docs/SOURCES_OF_TRUTH.md`, and the failure modes in
`DECISION_LOG.md`. If adopted, it replaces the body of `CLAUDE.md` and is then
mirrored byte-for-byte to `AGENTS.md` (the mirror rule below) — the current
CLAUDE.md≡AGENTS.md duplication stays, but it is the *only* sanctioned copy.

---

# Algoverse-Bias-Steering — instructions for every session (human + agent)

**The project is FROZEN as of 2026-08-17. Deadline 2026-08-28 AoE, numbers
freeze 2026-08-24.** Default posture: **execute, validate, write.** Not "make it
more interesting."

## 1. Read first — the control plane

Read in this order. If two files disagree, the one higher up wins.

| File | Owns | |
|---|---|---|
| **`PROJECT_STATE.md`** | current gate, blockers, next decision | **START HERE** |
| **`RESEARCH_CONTRACT.md`** | the frozen science; changing it needs a §12 amendment | |
| **`WORK_LEDGER.md`** | work packages, owners, evidence-of-done | |
| **`docs/PREREG.md`** | choices frozen before outcomes were inspected | |
| `docs/SOURCES_OF_TRUTH.md` | the fact-ownership registry — which file *defines* each thing | read before writing any fact |

`DECISION_LOG.md` is **history, not doctrine** — it explains *why* earlier docs
were superseded so nobody re-derives a cut idea in week three. It governs
nothing.

If you are lost: `docs/THE_CORRECT_PROBLEM.md` → your `RUNBOOK_<NAME>.md` →
`docs/SOURCES_OF_TRUTH.md`. Then ask with *tried / expected / saw*.

## 2. Never plan or execute from these

1. **`docs/superseded/`.** Each file carries a banner and describes a cut framing
   (byte-fallback perturbation, the four-claim structure, the eight-way rubric,
   the Qwen-27B trajectory, the steering-technique survey). Kept for provenance;
   does not govern. `main` and `team-kit` once forked into two disjoint `docs/`
   trees and the pipeline owner worked from a branch where the canonical rubric
   did not even exist (D-005) — that is what these banners prevent.
2. **`RUNBOOK_*.md`, `HANDOFF_*.md`, chat summaries.** Personal scratchpads,
   marked non-canonical. They may hold commands and handoff notes; they may
   **not** redefine the paper, an experiment, a metric, a model set, a deadline,
   a rubric, a claim, or a definition of done. If one disagrees with §1, it is
   stale. (Most `HANDOFF_*` are now stub pointers to `RUNBOOK_*` — do not add
   substance to a stub, that recreates the duplication.)
3. **The archived 2025 `.pkl` / `.pt` files as authority.** See §4 and §5.

## 3. Making changes

- **Branch, never push `main` directly.** CI and the dashboard assume `main` is
  canonical. Name branches `<initials>/<topic>` or `<type>/<topic>`
  (`fk/init-refusal-rewrite`, `fix/steering-shape-guard`, `dashboard/<topic>`,
  `exp/<name>`). The 2025 `farhan-experimentation-4` / "Reorganize Files" x40
  history is what we are *not* doing again.
- **One fact, one owner. Link, never restate.** Before writing a fact, check
  `docs/SOURCES_OF_TRUTH.md`. If a file owns it, link to it. Restating is how
  this repo ended up with the judge rubric **specified four different ways** and
  the venue in ten files (`SOURCES_OF_TRUTH.md` preamble) — a disagreement
  invisible in any single doc, the exact failure the paper studies.
- **A new doc must claim a row in the registry, or it does not get created.** If
  it owns no fact, it belongs inside an existing owner.
- **Changing an owned fact = editing the owner, in a PR, plus the claim ledger
  in the *same* PR** if it changes what we assert (claim status lives in
  `scripts/build_dashboard.py → CLAIMS`). Not a follow-up.
- **Frozen/audit docs are superseded, not edited.** `docs/REVIVAL_AUDIT.md` and
  filed sections of `docs/PREREG.md` keep their original text; append a dated
  note above the affected section instead.
- **Amending the frozen science:** only `RESEARCH_CONTRACT.md §12`'s five
  stop-rule conditions may reopen the contract (preempting paper; G1 failure;
  invalid measurement; implementation error affecting the primary result; design
  can't distinguish hypotheses at achievable n). "This would be more
  interesting", another model, another benchmark, more compute, or *an agent
  proposing a more ambitious paper* may **not**. A material change needs a dated
  §12 amendment **and** a `DECISION_LOG.md` entry, same PR.
- **Framing is centralized.** Before writing or editing ANY paper text —
  abstract, related work, review response, even a commit message — read
  `PAPER_FRAMING.md` / `RESEARCH_CONTRACT.md` and follow it. Disagree? PR the
  owner file; do not introduce a rival framing in a draft.

## 4. Numbers and data integrity

- **Every number traces to a committed artifact under `runs/` or `experiments/`,
  or is flagged pending.** Cite the path in the PR body. No hand-edited
  conclusions.
- **Recount from TEXT LOGS, not pickles.** Use `src/textlog_parse.py` +
  `src/recount.py`. Historical CSVs are **UNTRUSTED comparators** even when a
  recount agrees. Reasons, all real:
  - The archived `.pkl` are **cumulative** — `log_236` holds all 7 models
    appended; only the **last 96 records** belong to the named model.
  - **Denominator is n = 96 per arm.** Not ~100.
  - Arrow-named columns (`Init->Opin`) are **per-arm label marginals, NOT
    transitions** — real transitions need prompt-level pairing the CSVs lack.
  - **Unpickling executes arbitrary code.** Write new artifacts in a
    non-pickle format.
  - `none`/`NONE` markers are **judge-extraction failures** (2,032 of them
    across 107 files) — never fold them into "nonsense" or any behaviour class;
    `src/judging.py` has explicit `ok`/`no_match`/`ambiguous` states, use them.
- **New artifacts: one run, one file, model recorded *inside* the record.** Never
  reproduce the cumulative-append pattern. Identify by full repo-relative path,
  never by Log number (Log numbers are not unique — two `Log_200_*` exist).
  Follow the manifest protocol in `docs/REVIVAL_AUDIT.md` (revision hashes, data
  hashes, **stored split indices** not just a seed, payload SHA-256 + asserted
  shape, template hash, judge hash + revision, analysis commit).
- **Push run outputs to a branch** as raw artifacts under the existing
  `Log_N_*` / `runs/` convention.
- **Open-weight models only** — steering needs residual-stream access.

## 5. Steering-claim hygiene

- **Assert tensor shape against `(n_layers, d_model)` before shipping ANY
  intervention.** The 2025 refusal `.pt` files are **1-D** hidden-width tensors;
  `steering_vector[layer]` on a 1-D tensor yields a **scalar** that broadcasts
  across the residual width — a DC offset, not a direction — compounded by a
  model/`vector_files` ordering mismatch (D-004). It raised no exception and
  produced plausible numbers. This class of bug is silent. Guard pattern:
  `RUNBOOK_FARHAN.md` Task 1; bind vectors to models by an explicit
  `{model_name: path}` map, never two parallel lists.
- **Say "a direction", never "the direction".** Steering success does not
  identify the representation (non-identifiability, arXiv:2602.06801); geometry
  (cosine/principal angle) is not evidence of functional separation (D-008).
- **No steering result is reportable without** (a) a system-prompt baseline on
  the same prompts (AxBench), (b) per-example distributions — the 3×3 judge
  confusion counts, not just means, and (c) for any headline intervention,
  side-effect audits (capability + safety).
- **An invalid run is not a negative.** The refusal↔opinion cross-application is
  **RETRACTED as invalid, not null** (D-004) — never cite it either direction.
  The 2025 CrowS-Pairs transfer failure *is* load-bearing motivation: don't
  soften it, don't overclaim it.
- **The scalar-broadcast reconstruction (WP-30) is a forensic demo, an additive
  offset — never report it as an operator-matched control** (`RESEARCH_CONTRACT.md`
  §9).

## 6. The judge is part of the method

- **Every judged number carries its judge version, in the artifact, not in
  prose.** Never mix judge versions in one table — not even with a footnote.
- **Judge v1 (2025, GPT-4o-mini binary rubric) is RETIRED, not caveated.** Its
  rubric scored factual decisiveness as opinionation, so every archived vector
  was fit over mixed labels; reproducing v1 counts (we did, 7/7) validates the
  *bookkeeping*, not the *construct* (`RUNBOOK_FARHAN.md` Task 5). No v1 label in
  any new analysis, figure, or sentence.
- Any rubric or model change = a **new judge version**, documented in
  `docs/judges/`. The label classes live only in `docs/RUBRIC_v2.md` — link,
  never restate.

## 7. Definition of done

- **A task is done when its evidence exists and validates — not when a report
  says it ran** (`WORK_LEDGER.md`). Status is derived from the evidence column;
  no artifact ⇒ not done, regardless of what anyone (or any agent) reports. The
  repo has already had agent-written plans claim "G1 already passes" when it did
  not (D-015).
- **No paper-blocking package is validated by whoever produced it.** Validation
  is from the artifact and the contract, not a conversation — see the
  builder/validator table in `WORK_LEDGER.md`.
- **Human gold labels stay human.** Gate-1 Cohen's κ labels must be independent
  humans who have read `docs/RUBRIC_v2.md`, over 2026 model-set source text
  (`docs/MODEL_SET_2026-08-07.md`) — the 2025 archive is not a valid gold pool.
  Agents may prepare CSVs and compute agreement; they may **not** fill official
  gold labels. Making the gate "LLM judges the LLM" is the circularity the
  project exists to avoid (`docs/TEAM_HOW_WE_WORK.md`).

## 8. Terminology (do not coin)

- The behaviour is **hedging** (arXiv:2502.19463); the failure mode is
  **over-abstention on answerable items** (distinct from AbstentionBench
  arXiv:2506.09038, which targets *unanswerable* ones). **"Soft refusal" is
  retired** (D-012).
- **Opinionation** = whether a side is taken (scalar), not which ideology.
  **Hard refusal** = safety/harm decline (Arditi lineage). S4/S5 and the
  intervention classes are defined in `docs/INTERVENTION_CLASSES.md` — link only.

## 9. How Claude Code / agents behave in this repo

- **Agents are tools, not co-authors of truth.** Use them for code, refactors,
  plots, log parsing, recounts from text logs, drafting docs a human will edit,
  literature search with citations. Do **not** use them to invent or "recall"
  archive numbers, replace human gold labels, unpickle untrusted blobs without
  need, or restate the rubric in a fifth file. If an agent wrote it, a human
  still owns whether it is **true for this repo**.
- **Port before you fix.** When moving code (e.g. packaging the notebook),
  reproduce behaviour exactly and verify identical outputs first; change
  behaviour only in a separate commit (`RUNBOOK_FARHAN.md` Task 2).
- **Tests run as a suite, not per-file.** `python3 -m pytest -q` over the repo;
  `tests/conftest.py` rolls back the global registries and fails any test that
  leaves them mutated. A per-file green was once an artifact of the measurement —
  17/110 were red under normal collection (D-019). Don't trust per-file passes.
- **Hardware serialises the critical path.** `transformer_lens` is not importable
  outside the Lambda box, so **G1 and every torch-touching change can only run
  there** — run G1 from `docs/HANDOFF_G1.md`. This is the single-engineer risk,
  now verified; plan around it, don't pretend a torch change was validated
  locally.
- **Dashboard is derived — never hand-edit `dashboard/index.html`.** Edit the
  narrative constants at the top of `scripts/build_dashboard.py`, rebuild
  (`python3 scripts/build_dashboard.py --out dashboard/index.html`), ship as a
  `dashboard/<topic>` PR. Use the **`dashboard-update` skill**
  (`.claude/skills/dashboard-update/`). CI (`.github/workflows/dashboard.yml`)
  rebuilds on push to `main` touching a canonical doc. Note: **no deploy step
  exists** — CI commits the page back to `main` and stops; `bias-steering.exe.xyz`
  serves a legacy page and is not a projection of this repo (D-018), don't cite it.
- **`notebooklm/*.md` are regenerated from paper owners** — fix the owner and
  regenerate, don't patch the pack.

## 10. The CLAUDE.md ≡ AGENTS.md mirror rule

`AGENTS.md` is a **byte-for-byte mirror** of `CLAUDE.md` (different agent stacks
read different filenames) — the *only* sanctioned duplication in the repo. Edit
`CLAUDE.md`, then `cp CLAUDE.md AGENTS.md` in the **same commit**. Never edit
`AGENTS.md` directly. Any other apparent duplication is a registry bug — report
it in `docs/SOURCES_OF_TRUTH.md`.
