# Pre-registration — UNFILLED SKELETON

**Status: not filed.** Every field below is blank on purpose. This document
exists so the links pointing at it resolve, and so filing it is a matter of
filling fields rather than inventing structure under time pressure.

**Filing = committing this file with the fields populated, then never editing
the pre-registered sections again.** If something has to change afterwards,
add a dated amendment at the bottom stating what changed and why. An amended
pre-registration is honest; a silently edited one is worthless.

Owner: Edward. Due before annotation begins.

---

## 1. Frozen rubric

- Rubric file: `docs/RUBRIC_v2.md`
- **Commit hash of the frozen rubric:** `________`
- Date frozen: `________`
- Category count as frozen: `________`
- Agreement target: per-category Cohen's κ ≥ 0.70, at most two iterations

> The hash is what "frozen" means. A meeting agreeing on a rubric is not a
> freeze; a commit hash recorded here is.

## 2. Primary endpoints

Declared **before** looking at any outcome:

- `________`

## 3. Model set and revisions

Set is defined in `docs/MODEL_SET_2026-08-07.md`. Record the immutable revision
hash actually used for each — not the tag, which moves:

| model | HF revision hash | n_layers | d_model (from `text_config`) |
|---|---|---|---|
| | | | |

## 4. Layer selection

- Rule for choosing `l*`: `________`
- **`l*` must be fixed from an independent sweep before any outcome measure is
  judged.** Record the hash of the commit that fixes it: `________`

## 5. Statistical plan

- Equivalence margins for near-zero cells (TOST): `________`
- Multiplicity handling across layers: `________`
- Multiplicity handling across headline tests: `________`
- Clustering for confidence intervals: `________`
- Extraction-variance estimate (contrast-set redraws): `________`
  *(Currently unknown — see `RUNBOOK_JEREMIAH.md` workstream 2. No cosine is
  interpretable until this exists.)*

## 6. Exclusions, declared in advance

- Incoherence handling: `________`
- Extraction failures (`none` markers): **never folded into a behavior class**
- Any other pre-committed exclusion: `________`

## 7. Degradation ladder

The order in which scope gets cut under time pressure, committed before the
pressure arrives (see `docs/2026-08-02_sprint_proposal.md` §8):

1. `________`

## 8. What would falsify the headline claim

- `________`

---

## Amendments

*(none yet — append dated entries here; never edit above this line after filing)*
