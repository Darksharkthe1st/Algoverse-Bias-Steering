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

## 6. Null calibration of the behavioural tests (procedure fixed before running)

A median of ABSOLUTE correlations is positively biased under the null, so §5's
readings needed a calibrated zero-reference: 2000 within-category permutations
of the behavioural variable, seed 0, applied to both tests symmetrically
(`null_calibration` in the artifact).

- Shared axis vs abstention: observed 0.110 against null median 0.048
  (q95 0.078), **p = 0.001**. The axis is weakly but reliably related to
  abstention behaviour — real signal, at about a third of the pre-registered
  alignment bar. Licensed phrasing: a shared component weakly associated
  with abstention. Not licensed: any name implying construct identity.
- Residual vs stereotype margin: observed 0.030 against null median 0.048,
  **p = 0.88**. No detectable relationship under the tested behavioural
  statistic; the point estimate sits below what chance alone produces.

## 7. Strongest claim licensed by the experiments, as of this audit

Tier 1 (specific, one R1 model so far): on the same model, dataset files,
capture site, and estimator family, a behaviour-derived contrast yields
directions that mostly fail to reproduce (best race-related cell 0.384 under
the most permissive statistic), while an annotation-derived contrast yields
directions that reproduce at 0.974–0.984, survive template-disjoint splitting
(0.804–0.931), transfer to held-out categories (AUC 0.924–0.990), and retain
reproducible category-specific residues (0.836–0.930) after leave-one-out
shared-axis removal — and whose behavioural anchoring is nonetheless thin:
the shared axis relates weakly but reliably to abstention (median |ρ| 0.110,
calibrated p = 0.001) and the residues show no detectable relation to
stereotype behaviour (p = 0.88).

Tier 2 (the general statement Tier 1 licenses): reliability and behavioural
validity dissociated under both recipes — behavioural provenance did not buy
reliability (run 1), and reliability did not buy behavioural identification
(R1). Neither check alone licenses calling something a bias direction.

Not licensed: any name for the shared axis; "category-specific BIAS
geometry"; "only the contrast labels changed" (use §1's exact formulation);
cross-model structure claims before prereg 24 §A fires; any causal or
steering claim.

## 8. P0 — the matched behaviour-derived arm (17 §5.4), now run

`runs/_r1_audit/qwen-1.8b_matched_behaviour_arm.json`,
`scripts/r1_matched_behaviour_arm.py` (all choices frozen in its docstring
before first run). Same 200 ambiguous items per category, same captured
activations, the same 400 split assignments, same layer aggregation and
reporting statistic as the annotation arm. Margins for these exact items were
scored on this machine with the run-1 scoring code.

| category | annotation | behaviour (quintile) | behaviour (median split) | ctrl fixed | ctrl indep |
|---|---|---|---|---|---|
| Age | 0.983 | 0.318 | 0.060 | −0.092 | 0.002 |
| Disability_status | 0.979 | 0.735 | 0.691 | 0.247 | −0.010 |
| Gender_identity | 0.980 | −0.093 | −0.035 | −0.180 | −0.003 |
| Nationality | 0.975 | 0.132 | 0.062 | 0.001 | 0.002 |
| Physical_appearance | 0.983 | 0.617 | 0.499 | −0.015 | −0.008 |
| Race_ethnicity | 0.976 | 0.022 | 0.002 | 0.097 | −0.003 |
| Race_x_SES | 0.982 | 0.127 | −0.001 | 0.028 | 0.003 |
| Race_x_gender | 0.978 | 0.191 | 0.256 | −0.015 | −0.006 |
| Religion | 0.983 | −0.114 | −0.328 | −0.037 | −0.003 |
| Sexual_orientation | 0.984 | 0.216 | 0.172 | −0.083 | −0.012 |

Readings, in licensing order:

- With item universe, activations, splits, aggregation and statistic all
  matched, contrast construction separates 0.975–0.984 from −0.114–0.735.
  The controlled version of the provenance claim now exists for this model.
- **Run 1's two-category island replicates.** Disability_status (0.735) and
  Physical_appearance (0.617) are again the only categories that rise, on a
  different item sample, different hardware, and a different floor statistic
  than run 1. The behaviour-derived recipe is not pure noise; it weakly
  recovers exactly the two categories it always recovered. (On run 1's own
  qwen-1.8b attempt these two were dropped by the task-control gate, so run
  1 reported 0 of 8 there; the matched arm shows the island was present at
  this universe all along.)
- The median-split sensitivity (uses all 100 items per half) is LOWER than
  the quintile variant for most categories, so the behaviour arm's weakness
  is not an artifact of discarding 60% of the items.
- Independent controls sit at 0.000 ± 0.012 across all categories.

**Remaining differences after matching, enumerated (not claimed away):**
(1) the behaviour contrast uses only the ambiguous arm (inherent to its
definition); (2) effective rows per half-direction 40 (quintile) / 100
(median split) vs 200 for the annotation arm — the median-split variant
narrows this and moves the result away from, not toward, the annotation arm;
(3) margins scored on MPS in this session vs run-1's CUDA margins (instrument
identity rests on the parity gate, not file identity); (4) single-measurement
margin reliability is unknown and attenuates the behaviour arm — this is part
of what the comparison measures.

## 9. Claim ladder (state at 2026-08-31, qwen-1.8b only)

| rung | annotation-derived arm | behaviour-derived arm |
|---|---|---|
| Reproducibility | 10/10, floors 0.975–0.984; template-robust | 2/10 weakly (0.62–0.74), same island as run 1 |
| Specificity | not length (projection); not template; pairwise distinguishable; residues reproduce 0.84–0.93 | not established |
| Behavioural association | shared component weakly associated with abstention (0.110, p=.001); residues: no detectable relationship under the tested statistic (p=.88) | by construction (circular), never validated |
| Intervention / causal | untested | untested (run-1 transfer was uncontrolled) |

No rung above behavioural association is occupied by either arm. "Bias
direction" is not licensed for any object in this study.
