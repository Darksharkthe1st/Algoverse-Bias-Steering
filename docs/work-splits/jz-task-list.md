# Jeremiah (jz) — task list

**Source:** `Algoverse — 2 Wk Plan` → *"There is more to refusal than a single direction"*.
**Thesis to defend:** the counter-story — bias is not one axis; refusal and bias are
distinguishable; a single direction is insufficient.

> **Read before starting:** `PROJECT_STATE.md`, `RESEARCH_CONTRACT.md` §12, `AGENTS.md`,
> `RUNBOOK_JEREMIAH.md`, `HANDOFF_JEREMIAH.md`. Frozen project (2026-08-17). Items that
> introduce a new taxonomy or terminology are **exploratory** and need a §12 amendment to
> enter the paper. §N refs → `docs/superseded/needed-experiments.md`.

---

## JZ-1 — Classify different types of bias

- **Do:** build a working taxonomy of the bias behaviors our datasets actually contain —
  grounded in the data, not invented: BBQ has ground-truth stereotype categories
  (`datasets/BBQ_Prompt_Sets/*.jsonl`: race, gender, religion, …), CrowS-Pairs has
  stereotype categories, GPT/Grok/Homemade sets are opinionation-style. Map each dataset →
  what construct it measures (stereotype-alignment vs opinionatedness vs hedging).
- **⚠️ Doctrine check:** a *bias taxonomy* is listed under `PROJECT_STATE.md` §"Does not block
  the paper", and the old eight-way rubric / `INTERVENTION_CLASSES` are **superseded**
  (`AGENTS.md` rule 1). Do not coin new terminology (`AGENTS.md` rule 5 — "soft refusal"
  retired; behavior is *hedging*, failure mode is *over-abstention on answerable items*).
  Keep this as a **data-grounded map**, not a new construct that competes with the frozen rubric.
- **DoE:** a table (dataset → construct → ground-truth availability → category list) committed
  under `docs/`, each row citing a real file under `datasets/`.

## JZ-2 — Confirm orthogonality (bias vs refusal, and bias-type vs bias-type)

The empirical heart of "more than one direction." Coordinate with Farhan FK-2/FK-3.
- **Do:** extract a direction per bias type (BBQ-stereotype, CrowS, opinion) with the **same**
  `mean_diff` pipeline used for refusal (FK-2), then compute the **per-layer cosine matrix**
  among all of them + the native refusal vector, against the null floor (~1/√d ≈ 0.022 for qwen).
- **Hygiene:** assert every vector is `(n_layers, d_model)` before any cosine (Log-213 scalar-
  broadcast bug, `docs/REVIVAL_AUDIT.md`, `AGENTS.md` §6). Say "a direction" (arXiv:2602.06801).
- **DoE:** a cosine heatmap/matrix + null floor + a verdict per pair (orthogonal/oblique/aligned).
  A high off-diagonal cosine *supports* the single-direction story; low cosines support "more
  than one." Report whichever the data says — honest negatives stay honest.

## JZ-3 — Compare to refusal vectors

- **Do:** the head-to-head — does the refusal direction predict/steer bias behavior and vice
  versa? Reuse Farhan's FK-4 cross-application table; add the *representational* comparison
  (JZ-2 cosines) so the claim rests on both behavior (does steering transfer) and geometry
  (are the directions aligned).
- **DoE:** a combined behavior+geometry statement: e.g. "opinion and refusal directions are
  X° apart at the best layer and cross-application recovers Y% of native effect."

## JZ-4 — "Confirm all work overall" (the identification / robustness pass)

- **Do:** sanity-check that the above survives the four frozen gates in spirit — positive
  control, direction precision, random-direction specificity, coherence (`PROJECT_STATE.md`
  §"Current gate"). At minimum: a covariance-matched **random-direction control** for every
  cosine/steering claim (does a random direction move it <0.05?), and the coherence gate (§0.3).
- **DoE:** each headline number carries its random-direction control and coherence check, or
  it is marked provisional. "Done = evidence exists and validates" (`WORK_LEDGER.md`).

---

### Related `needed-experiments` Jeremiah can own
- **§4** rubric-v2 + Cohen's κ gate (≥0.70) — the construct-validity backbone for JZ-1; without
  a trustworthy judge every bias-type number is provisional. HIGH for the paper.
- **§8** cross-model vector transfer — is "bias"/"refusal" a shared direction or model-specific?
  Directly informs "more than one direction." MED.

### ⚠️ Doctrine checks for JZ's section
- The **orthogonality** work (JZ-2/JZ-3) is in-scope and directly complements Farhan's line —
  this is the strongest paper-relevant part of this section.
- **Taxonomy** (JZ-1) is exploratory (does-not-block scope); keep it data-grounded, no new
  coined constructs, no competing rubric — else it needs a §12 amendment.
