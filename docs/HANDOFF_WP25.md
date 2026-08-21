# HANDOFF_WP25 — stratifying the battery (S1/S2/S3)

**Canonical, like `HANDOFF_G1.md` — not a scratchpad.** It executes
`docs/PREREG.md` §3, frozen before any primary outcome was inspected. It
sequences the work; it may not redefine the rule. If anything here disagrees
with PREREG §3, PREREG wins.

**Why this blocks:** WP-25 gates WP-04 (extraction), and the labels must be
committed **before anyone inspects a λ>0 generation** — the commit timestamp is
itself part of the evidence.

**Owner:** Jeremiah (coordination). **Annotators:** Farhan, Aryaman, Jeremiah —
three, which is exactly what PREREG §3 requires. No GPU, no coding beyond a
small script edit in step 4.

**Edward is not an annotator:** on 2026-08-20 he read archived model responses
to battery items (labeling-sheet work), so his blindness to model output is
compromised for this instrument.

---

## 1. Build the labeling sheet (~30 min, CPU)

- Input: `datasets/GPT_Prompts/comparison_questions_200.csv` (296 rows).
- Remove the 3 exact duplicates PREREG §3 names → 293 unique items.
- Apply the S1 regex (first-person preference verbs, per PREREG §3)
  **deterministically** — S1 items are excluded from adjudication, not labeled
  by hand. Commit the regex and its match list with the sheet.
- Export the remaining items with **only the item text** — no model output, no
  2025 archive columns, nothing that could leak behaviour. Blindness to model
  output is the frozen requirement, and the sheet is where it is enforced.
- Shuffling presentation order per annotator is recommended (execution detail,
  not a frozen choice).

## 2. Annotate independently

- Each annotator labels every item S2 or S3 against the PREREG §3 definitions.
- No discussion of items until all three sheets are returned. No model outputs
  consulted. If an annotator remembers generations for an item from the 2025
  archive, they label from the item text alone and note the item id.

## 3. Merge — mechanical, no judgment

- Majority wins; **ties → S3** (frozen; conservative, since S3 is where hedging
  is correct).
- Flag the 18 unnameable-alternative items PREREG §3 lists for DV exclusion.

## 4. Agreement number (WP-21, ~60 lines)

- `scripts/kappa_from_csv.py` currently does 2-rater Cohen's κ only. Extend to
  3-rater Fleiss κ + bootstrap CI.
- The disagreement rate is **reported either way** — it is not a gate to pass,
  it is a number the paper carries.

## 5. Commit

- One CSV: item id, item text, per-annotator labels, majority stratum,
  exclusion flags. Plus adjudication notes: date, annotators, disagreement
  rate, any imperfect-blindness notes from step 2.
- Branch → PR into `main`, per repo rules. The ledger's WP-25 evidence column
  points at the committed paths in the same PR.

## Done when

The CSV and notes are merged and validate against PREREG §3 (counts reconcile:
293 unique = S1 + adjudicated; 18 DV-exclusion flags present). The moment this
lands, Farhan's WP-04 is unblocked.
