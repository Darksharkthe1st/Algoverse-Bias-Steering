# PROPOSAL — construct-matched opinion eval (OpinionQA / GlobalOpinionQA)

**Status: PROPOSAL, awaiting team decision. NOT enacted.** This document changes
nothing. It does not modify `RESEARCH_CONTRACT.md`, `docs/PREREG.md`, or the dataset
registry in `docs/SOURCES_OF_TRUTH.md`. It exists so the team can approve or reject a
scope change that the stop rule (§12) disfavors by default.

**Date:** 2026-09-23 · **Raised by:** FK (via session work on IssueBench extraction) ·
**Decision owner:** contract owner (Edward) + team.

---

## 1. The ask

Add a **genuine opinion-question** eval battery — **OpinionQA** (Santurkar et al.,
2023, arXiv:2303.17548) and/or **GlobalOpinionQA** (Durmus et al., 2023,
arXiv:2306.16388) — as a construct-matched testbed for the opinionate-vs-hedge
behaviour.

## 2. Governance reality (read before arguing the science)

This proposal runs **against** the frozen contract in two explicit ways; both must be
consciously accepted by the team, not waved through:

1. **§12 "May not reopen scope" lists *"another benchmark"* by name.** Adding a
   dataset is precisely what the stop rule says does *not* justify reopening. None of
   the five reopening triggers (§12.1–12.5) applies. So this needs a **deliberate,
   consensus amendment**, not a routine PR — the default answer under the contract is
   *no*.
2. **The frozen construct (§2) is narrower than "opinion."** It is *hedging on
   forced-binary comparative questions* (`Which <PROP>: <A> or <B>?`,
   `datasets/GPT_Prompts/comparison_questions_200.csv`). OpinionQA is opinion-survey
   format (Pew items), a **different question shape**. Adopting it therefore either
   (a) serves only as an *external transfer probe* leaving the construct unchanged, or
   (b) **broadens the construct** — a larger amendment touching §2.

If the team is not prepared to do (1) and pick between (a)/(b), stop here: the answer
is "keep the frozen dataset set."

## 3. Why it came up (evidence from 2026-09-23 session)

- **Toy comparative battery is a ceiling regime for the baseline.** A system-prompt
  baseline on the toy prompts hits **100% both directions** (run
  `20260923-030712`), because the target behaviour is trivially instructable on
  trivially side-able questions — so it cannot test the method's limits.
- **IssueBench (already in scope) breaks the ceiling but is an adjacent construct.**
  The prompt baseline dropped to 86% / 91% on IssueBench (run `20260923-072454`) — a
  real contest. But IssueBench measures *issue bias in writing assistance*, and its
  prompts embed the stance in the template ("X being a bad thing"), which **confounds
  natural-variation extraction** (run `20260923-080817`: 160 neutral / 40
  opinionated, the 40 almost all stance-embedding templates). Forced-contrast
  extraction fixes the *extraction* confound, but the *construct* is still
  issue-writing, not "commit-or-hedge on a question."
- **OpinionQA / GlobalOpinionQA are genuine opinion questions** — no embedded stance,
  no writing-task framing — so they are the closest available match to "on a
  subjective/controversial question, does the model take a side or hedge." That is the
  construct-validity gap this proposal targets.

## 4. Options for the team

- **Option A — Reject.** Keep the frozen dataset set; §12 stands. Cheapest; fully
  defensible.
- **Option B — Adopt as a bounded external-generalization eval only.** OpinionQA used
  as a *transfer* battery to test whether an opinion direction (fit elsewhere) moves
  behaviour on a different construct; the primary battery and §2 construct are
  **unchanged**. Smallest amendment. Recommended if adopting at all.
- **Option C — Broaden the construct** to include opinion-holding questions. Largest
  change; edits §2 and the claim ledger; only if the team wants the paper's headline
  to cover opinion questions, not just comparatives.

## 5. Draft amendment text (only if Option B or C is chosen)

To paste into `RESEARCH_CONTRACT.md` §12 Amendments, in the A1–A6 style, **after team
sign-off** (this is a draft, not an enacted amendment):

> **2026-XX-XX — A7. Add OpinionQA/GlobalOpinionQA as an external construct-matched
> eval [Option B].** Reason: **not a §12.1–12.5 trigger — a deliberate scope decision
> taken by team consensus.** The frozen comparative battery is a baseline-ceiling
> regime and IssueBench is an adjacent (issue-writing) construct; a genuine
> opinion-question battery closes a construct-validity gap for the opinionate-vs-hedge
> claim. Used as a transfer/generalization eval only; **changes no hypothesis,
> statistic, threshold, gate, or model set**, and does not alter the §2 construct.
> Adds a dataset row to `docs/SOURCES_OF_TRUTH.md` and a loader (`scripts/fetch_*`, a
> `datasets.py` registration) mirroring `issuebench`/`axbench`.

If Option C, the amendment must instead state that §2 is broadened and update the
claim ledger — a larger edit, not drafted here.

## 6. Cost if adopted

Loader + fetch script mirroring `issuebench`/`axbench` (~1 day); **no new model**;
judging via the existing neutrality judge. If accepted, this doc's row is claimed in
`docs/SOURCES_OF_TRUTH.md` and this file is retired to `docs/superseded/` or deleted.

## 7. Recommendation

Neutral-to-lean-**B if the team wants stronger construct validity** for the
opinionate-vs-hedge claim; otherwise **A**. Do **not** treat this as approved by its
mere existence — the stop rule's default is no, and only the decision owner can change
that.
