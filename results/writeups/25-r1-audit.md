# 25 — Audit of the R1 annotation-contrast result (qwen-1.8b)

**Written 2026-08-31, before any paper edit based on it.** Question under
audit: does the R1 result establish category-specific bias geometry, or
primarily a shared ambiguous-vs-disambiguated construct? Machine-readable
outputs: `runs/_r1_audit/qwen-1.8b.json` (provenance, leakage, controls,
LOCO, decomposition), `runs/_r1_audit/qwen-1.8b_margins.json` and
`_axis_alignment.json` (pre-registered behavioural tests, writeup 24 §B),
`runs/_r1_audit/cross_model_structure.json` (writeup 24 §A, fills as models
land). Analyses and statistics were frozen in the script docstrings before
execution. Prereg hash for 24: commit `1d6694b`.

## 1. Provenance

- **Item universe.** Per category: every BBQ row pairs on the true scenario
  key `(question_index, polarity, ans0, ans1, ans2)`; zero rows dropped;
  100% of pairs corroborated by consecutive example_ids. The run subsamples
  200 evenly-spaced pairs per category → arm A = 200 ambiguous rows, arm B =
  their 200 disambiguated partners. Verified from `pairing_report.json` and
  re-derived fresh: sidecar item ids match a fresh pairing exactly, arm A is
  all-ambig and arm B all-disambig in every category.
- **Split unit** is the scenario pair (both arms travel together); all 400
  split seeds re-derived, zero pair overlap between halves.
- **Estimator**: difference of arm means, no hyperparameter. **Capture**:
  `resid_pre`, chat-template index −1, template tail recorded per model.
- **"Only the contrast labels changed" is NOT literally defensible.** Held
  identical to run 1: model, dataset files, bare-prompt scoring surface, chat
  template and system prompt, capture hook and index, difference-of-means
  family. Changed beyond the labels: (i) the item universe (run 1: 600 seeded
  ambiguous items, extremes kept 240; R1: 200 pairs = 400 rows; run 1 saved
  no item ids for qwen-1.8b, so overlap is unverifiable); (ii) the floor
  statistic (run 1: q05 of 10 item-level splits, unweighted layer median;
  R1: mean of 400 pair-level splits, norm-weighted layer mean, bootstrap CI).
  The statistic gap is quantified and does not explain the flip: run 1's most
  permissive summary tops out at 0.384 on any race cell (writeup 22 §F),
  while R1's floors under run 1's own layer summary (the pre-declared
  unweighted-median sensitivity) are 0.988 to 0.990. (iii) The matched
  behaviour-derived arm on the same 400 rows (17 §5.4) has not run, so the
  "controlled comparison at matched n" is not yet available.

## 2. Leakage and the negative-control mechanism

- **Template overlap under standard splits is near-total** (halves share
  21–36 of 25–50 templates), so template leakage was a live concern.
  **Template-disjoint splits answer it**: floors drop from 0.974–0.984 to
  **0.804–0.931** and no verdict changes. The directions generalise across
  scenario templates; a 0.05–0.17 share of the floor is template-bound.
- **Why shuffled-label controls sat at 0.19–0.57**: the run's control design
  shuffles arm labels once and then splits, so both halves inherit the same
  imbalance of one Binomial(n, ½) shuffle and share the true signal it leaks.
  With the shuffle redrawn independently per half the control collapses to
  **−0.012 to +0.006**, and a label-free anisotropy null (random item
  bipartitions) sits at **−0.028 to +0.021**. Residual-population anisotropy
  contributes nothing at this capture site. Consequence: the shipped control
  is *conservative* (harder to beat than a fully independent null), so every
  YES verdict stands a fortiori; the control's level measures leaked contrast
  signal, not an artifact.

## 3. Shared axis vs category-specific geometry

- **The shared axis is category-general and transfers.** Leave-one-category-
  out: a shared direction from nine categories separates the held-out
  category's arms at **AUC 0.924–0.990** (norm-weighted across layers), and
  orders the members of a held-out scenario pair correctly in
  **99.86–100%** of pairs. Each category direction has norm-weighted
  |cos| **0.853–0.924** with its leave-one-out shared axis; the shared axis
  reproduces against itself at 0.998.
- **Category-specific geometry survives removal of the shared axis.** With
  the leave-C-out shared axis estimated per split-half (never using C or the
  other half) and projected out, residual floors are **0.836–0.930** with CI
  lower bounds 0.832–0.928, against the same conservative controls. The
  structure is one dominant shared component plus reproducible
  category-specific residues — not one axis plus noise.
- **What the residues are is not established.** Run 1's topic-identity
  control showed topic is strongly linearly decodable; reproducible
  category-specific residue is consistent with topic identity as much as
  with category-specific bias. The pre-registered residual-vs-stereotype-
  margin test (writeup 24 §B, extension B) is the first behavioural link;
  its result file is the arbiter, not this paragraph.

## 4. Deliberately not done here

No paper edits. No steering. No cross-model claims (prereg 24 §A runs as
models land). No naming of the shared axis ahead of the alignment test.

## 5. Pre-registered behavioural tests — both NEGATIVE (added same day)

`runs/_r1_audit/qwen-1.8b_axis_alignment.json`, statistics frozen in
writeup 24 §B before running:

- **Shared axis vs abstention margin: median |Spearman| = 0.110
  [0.077, 0.195].** Below the 0.30 bar. The axis separates the two arms
  nearly perfectly (LOCO AUC 0.92–0.99) yet does not grade the model's
  abstention propensity WITHIN the ambiguous arm. It behaves as a categorical
  input-property separator (context resolved vs not), not as a graded
  answerability signal. Per prereg, the axis stays unnamed.
- **Category-specific residual vs stereotype margin: median |Spearman| =
  0.030 [0.030, 0.112].** The residues do not track item-level stereotype
  behaviour. "Category-specific geometry" is supported; "category-specific
  BIAS geometry" is not licensed by this test.

Reading: R1 establishes reliability (reproducible, transferable, controlled),
and these two nulls establish that reliability has not bought behavioural
identification. Reliability and validity separate cleanly in this data, which
is itself the measurement-science point. Power caveat: within-category
item-level margins are noisy; the tests bound alignment at roughly |rho| <=
0.2, they do not prove zero.
