# Number-by-number verification of `main.tex`

Every figure in the draft, checked against a committed artifact under `repo/runs/`.
Done 2026-08-29. The repo rule is that **a writeup is not an artifact** — the
draft was originally written from `results/writeups/`, and this pass re-derived
each number from the JSON the writeups were themselves summarising.

Status key: **OK** verified exactly · **FIXED** draft was wrong or unclear, now
corrected · **OPEN** still needs a human check before submission.

---

## Table 1 — behavioural positive control (qwen-1.8b)

**Artifact found: `runs/full_qwen18/report.json`** (`categories[*].positive_control`).
All ten rows verified exactly, n=150 each. **OK.**

| category | draft acc | artifact | draft z | artifact z | gate |
|---|---|---|---|---|---|
| Religion | 64.0% | 0.640 | +8.0 | 7.967 | pass |
| Race_x_SES | 62.0% | 0.620 | +7.4 | 7.448 | pass |
| Race_ethnicity | 61.3% | 0.613 | +7.3 | 7.275 | pass |
| Race_x_gender | 60.7% | 0.607 | +7.1 | 7.101 | pass |
| Gender_identity | 58.7% | 0.587 | +6.6 | 6.582 | pass |
| Nationality | 56.0% | 0.560 | +5.9 | 5.889 | pass |
| Age | 55.3% | 0.553 | +5.7 | 5.716 | pass |
| Sexual_orientation | 55.3% | 0.553 | +5.7 | 5.716 | pass |
| Physical_appearance | 49.3% | 0.493 | +4.2 | 4.157 | **fail** |
| Disability_status | 44.7% | 0.447 | +2.9 | 2.944 | **fail** |

Note the gate is a conjunction: Physical_appearance fails on accuracy alone
(49.3% < 50%) while clearing z comfortably. Worth one clause in the paper so a
reviewer does not read z=4.16 as a pass.

**FIXED — a real error.** The draft implied the two dropped categories fail
generally. They do not. From `runs/probe_tuned_qwen7b_a1e6/report.json` and
`runs/probe_tuned_qwen14b/report.json`, all ten categories pass in both larger
models:

| category | qwen-7b acc | qwen-7b z | qwen-14b acc | qwen-14b z |
|---|---|---|---|---|
| Physical_appearance | 65.3% | 8.31 | 68.7% | 9.18 |
| Disability_status | 59.3% | 6.75 | 67.3% | 8.83 |

n = 150 per category. The exclusion is a property of the smallest model, and
those two categories are exactly the ones whose directions extract best in the
larger models — now stated in the draft as comprehension gating extractability.

---

## Table 2 — extraction positive control

Source: `runs/_extraction_positive_control.json` (model `qwen-1.8b`, 10 splits).

| contrast | draft q05 | artifact q05 | draft median | artifact median | status |
|---|---|---|---|---|---|
| Race_ethnicity vs Gender_identity | 0.882 | 0.8819585 | 0.899 | 0.8991181 | **OK** |
| Religion vs Age | 0.893 | 0.8925758 | 0.912 | 0.9116971 | **OK** |
| Nationality vs Sexual_orientation | 0.871 | 0.8705291 | 0.906 | 0.9060717 | **OK** |

n = 320 for all three. **OK**.

**Added:** the same file carries `random_floor = 0.0221`, and the tuned reports
carry 0.0140 (qwen-14b) and 0.0156 (qwen-7b). The draft now states that random
directions agree at 0.014–0.022, which makes the 0.500 bar meaningful rather
than arbitrary. This strengthens the paper and was missing.

---

## Table 3 — per-category split-half floors

Source: `runs/_cross_model_final.json`, `runs[].floors`. **All 47 values match
to three decimals across all five models. OK.**

Spot check, qwen-14b: Religion 0.685740→0.686, Age 0.753576→0.754,
Disability 0.819644→0.820, Physical 0.647848→0.648, Race_ethnicity
−0.203645→−0.204.

**FIXED:** the caption now says these are the **default** estimator. They are not
the floors underlying the clustering result, which uses the tuned probe. The
draft previously mixed the two silently.

---

## Section 5 — the forking path (was a one-paragraph caveat)

**FIXED — this was the most serious problem in the draft.**

Default estimator, from the `verdict` and `p_value` fields of
`runs/_cross_model_final.json`:

| model | reproducible | p |
|---|---|---|
| qwen-1.8b | 0 | 0.229 |
| gemma-2b | 2 | 1.000 |
| yi-6b | 2 | not clusterable (<3) |
| qwen-7b | 2 | not clusterable (<3) |
| qwen-14b | 4 | 0.179 |

**No model shows structure under the default estimator.**

Tuned probe, α=1e6:

| model | source | p | cluster strength | null median | null q95 | n_perm |
|---|---|---|---|---|---|---|
| qwen-14b | `runs/probe_tuned_qwen14b/report.json` | 0.0298507 | 0.0962127 | 0.0473 | 0.0852 | 200 |
| qwen-7b | `runs/probe_tuned_qwen7b_a1e6/report.json` | 0.0049751 | 0.3350812 | 0.0573 | 0.1378 | 200 |

All **OK** against the writeup. The mechanism is confirmed in the artifacts:
regularisation lifts categories over the 0.500 floor. qwen-14b tuned floors give
five reproducible (Religion 0.691, Age 0.826, Nationality 0.510, Physical 0.745,
Disability 0.880) versus four under default; qwen-7b gives three (Religion 0.525,
Physical 0.663, Disability 0.876) versus two. Three is the minimum at which
clustering is computable at all.

The draft now reports both estimators in one table and declines to claim
clustering. **This is a much better paper than the version that reported only
p=0.030.**

---

## Table 4 — transfer / steering

Source: `runs/full_gemma2b/transfer_test_norm_c{2,5,10}.json` and
`runs/full_qwen14b/transfer_test_norm_c{2,4,8,16}.json`, `summary` block.

| model | coeff | own | cross | random | flips | status |
|---|---|---|---|---|---|---|
| gemma-2b | 2 | −0.0805 | −0.0627 | +0.0115 | 2/2 | **OK** |
| gemma-2b | 5 | −0.2214 | −0.1709 | +0.0305 | 1/2 | **OK** |
| gemma-2b | 10 | −0.4326 | −0.3605 | +0.0646 | 0/2 | **OK** |
| qwen-14b | 2 | +0.0001 | +0.0038 | +0.0035 | 3/4 | **OK** |
| qwen-14b | 4 | +0.0021 | +0.0093 | +0.0066 | 3/4 | **OK** |
| qwen-14b | 8 | +0.0036 | +0.0191 | +0.0126 | 2/4 | **OK** |
| qwen-14b | 16 | +0.0067 | +0.0329 | +0.0250 | 1/4 | **OK** |

Derived quantities, recomputed here:

- **7× random control**: 0.0805 / 0.011478 = **7.01×**. Verified.
- **own vs cross**: 0.0805 / 0.0627 = **1.28×**.
- **item effect**: Disability items −0.1302 / −0.1028 (mean −0.1165); Physical
  items −0.0227 / −0.0308 (mean −0.0268). Ratio **4.35×**.
- qwen-14b own < cross at all four doses. Verified.

**FIXED — the draft overstated this.** It said own and cross were
"indistinguishable". They are not: own is consistently ~1.25–1.30× cross. The
correct and stronger claim is that direction identity contributes 1.28× while
item identity contributes 4.4×, so the effect is *largely* a property of what is
steered. There is no confidence interval on any of these, so "indistinguishable"
was unsupportable in either direction.

**New finding, not in any writeup:** gemma-2b's sign-flip control degrades
2/2 → 1/2 → 0/2 as the coefficient rises 2 → 5 → 10, while the effect grows
fivefold. The apparent success at coeff=2 does not survive dose escalation. This
is now in the draft and is a second, independent reason the cell should not have
been reported as a positive result.

---

## Table 5 — what predicts extractability

**Correction to an earlier draft of this file:** I first wrote that four of the
five norm artifacts were missing. That was wrong — I searched `runs/*.json` and
missed `runs/full_*/direction_norms.json` one level down. Four of five were
committed all along. **Only qwen-7b's was genuinely absent**; I generated it with
`scripts/direction_norms.py --run-dir runs/full_qwen7b`.

I did regenerate the other four before spotting this. They reproduced the
committed values to ~14 significant figures — differences confined to the last
2–3 digits, i.e.\ BLAS summation order, not a substantive change. That is a
passing reproducibility check and worth knowing. I then restored the committed
files with `git checkout`, so the only new artifact in the tree is qwen-7b's.

Recomputed Pearson correlations, floor vs Frobenius norm of the unnormalised
mean-difference vector:

| model | n | recomputed | previously claimed | |
|---|---|---|---|---|
| qwen-1.8b | 8 | +0.8080 | +0.808 | **OK** |
| gemma-2b | 9 | +0.9453 | +0.945 | **OK** |
| yi-6b | 9 | +0.7649 | +0.765 | **OK** |
| qwen-7b | 10 | +0.9037 | *(none)* | **new** |
| qwen-14b | 10 | +0.9122 | +0.912 | **OK** |

Behavioural tilt (mean margin) vs floor, from `runs/_cross_model_final.json`:
qwen-1.8b +0.6889, gemma-2b +0.6597, yi-6b +0.6894, qwen-7b +0.7693,
qwen-14b +0.7261 — all four previously published values reproduce, and qwen-7b
is new. **The table in the draft now reports all five models.**

Also worth knowing: `quintile_separation` correlates with the floor at only
**+0.0998** on qwen-14b, against +0.9122 for the norm. So it is specifically
activation-space magnitude, not how far apart the margin poles are, that
predicts whether a direction reproduces. That sharpens the claim and is not in
any writeup.

**A confound this exposes — now addressed in the draft.** Because norm predicts
reproducibility this strongly, an intervention adding the *unnormalised*
direction at a fixed coefficient would deliver a larger dose to exactly the
categories that reproduce, and any specificity result would be confounded with
dose. `scripts/direction_norms.py`'s own docstring raises this. The transfer
numbers in Table 4 come from the `transfer_test_norm_*` files, which carry
`"normalized": true`, so they are clean — but the paper now says so explicitly,
because a reviewer will ask.

## The heterogeneity refutation

Ran `scripts/heterogeneity_vs_floor.py --run-dir runs/full_qwen14b`; artifact
written to `runs/full_qwen14b/heterogeneity.json`. **Both figures verify exactly.**

| predictor of the floor | pearson | spearman | paper |
|---|---|---|---|
| number of distinct stereotype sets | **−0.079** | −0.091 | −0.079 **OK** |
| entropy of that distribution | **−0.020** | +0.067 | −0.020 **OK** |
| share of the most common set | −0.278 | −0.370 | *(not reported)* |
| mean margin | +0.726 | +0.721 | +0.726 **OK** |
| SD of the margin | +0.140 | +0.055 | *(not reported)* |
| mean/SD (effect size) | **+0.835** | +0.794 | *(not reported)* |

Two things worth considering for the paper. **Effect size (+0.835) predicts the
floor better than mean margin (+0.726)**, which sharpens the story: it is not
just that the model leans, it is that it leans consistently relative to item
noise. Direction norm (+0.912) still beats both. And the script's own note is
worth heeding — n=10 categories, so all of this is suggestive rather than
confirmatory, and it proposes a direct test (split a high-floor category by mean
margin and check only the large-mean subset reproduces) that would take one CPU
run if there is time.

## Section 6 — silent failure modes

**The 1/99 → 27/99 figure is now artifact-backed. OK.**

Ran `python -m src.recount experiments/past_logs/refusal_experiments/official_refusal_to_opinion/Refusal_To_Opinion.csv --repo-root .`
(output committed at `runs/_sim/recount_refusal_to_opinion.json`). This is the
text-log path the repo mandates; I did **not** run
`scripts/verify_2025_results.py`, which unpickles archived files, because
`CLAUDE.md` warns those execute arbitrary code.

`Log_213_Refusal_To_Opinion_Llama-2-7b-chat-hf`:

| arm | unsafe | safe | n |
|---|---|---|---|
| initial | **1** | 98 | 99 |
| neutral-steered | 21 | 78 | 99 |
| opinion-steered | **27** | 72 | 99 |

Exactly the counts in `REVIVAL_AUDIT.md`, and the recount reports
`all_match: true` across every entry — the legacy CSV agrees with an independent
parse of the raw text logs. So the *counts* behind the rejected headline are
sound; it is only the causal label that was false, which is precisely the point
the paper makes.

| other claims in this section | source | status |
|---|---|---|
| vector/model ordering off by rotation, 5 runs | `docs/REVIVAL_AUDIT.md` payload-hash table | **OPEN** |
| `Init->Opin` columns are marginals not transitions | `docs/REVIVAL_AUDIT.md` | **OPEN** |
| unpinned `gpt-4o-mini` judge alias | `docs/REVIVAL_AUDIT.md` | **OPEN** |
| SEL≥2 fires 1.8%→64% | `RESEARCH_CONTRACT.md` §5.1, `DECISION_LOG.md` D-009 | **OPEN** |

All five are documented in governing repo docs, and the supporting code exists
(`src/recount.py`, `src/textlog_parse.py`, `analysis/sim_lambda_identifiability.py`
— all present). None is backed by a JSON artifact under `runs/`.

**To close these:** run the recount and the simulation once each, and commit
their outputs. Both are CPU-only and should take minutes. Until then these are
claims from documents, which is exactly the standard the repo's own rule 3
rejects.

---

## Closed since the first pass

**Model identifiers — all five recovered** from `runs/full_*/report.json`:

| short name | hf_id | d_model |
|---|---|---|
| qwen-1.8b | `Qwen/Qwen1.5-1.8B-Chat` | 2048 |
| gemma-2b | `google/gemma-2b-it` | 2048 |
| yi-6b | `01-ai/Yi-6B-Chat` | 4096 |
| qwen-7b | `Qwen/Qwen1.5-7B-Chat` | 4096 |
| qwen-14b | `Qwen/Qwen1.5-14B-Chat` | 5120 |

Now in the draft. Revision SHAs are still absent — the repo's own rule 3 says a
manifest carrying a bare model name is not reproducible evidence, so add them if
the manifests have them, or state plainly that they were not pinned.

**Joad et al. citation — verified** against `arxiv.org/abs/2602.02132` on
2026-08-29. Title *"There Is More to Refusal in Large Language Models than a
Single Direction"*; authors Joad, Hawasly, Boughorbel, Durrani, Sencar; February
2026. The abstract independently confirms the parallel the draft draws: eleven
refusal categories map to geometrically distinct directions, yet steering along
any of them yields comparable trade-offs, acting as "a shared one-dimensional
control knob", with direction identity governing *how* rather than *whether*.
That is the same structure as the gemma-2b own-vs-cross result, so the comparison
is sound and not a stretch.

## Still open before submission

1. **Revision SHAs** for the five models, or an explicit statement that the runs
   did not pin them.
2. **The selectivity-ratio figures need one small change.** Ran
   `analysis/sim_lambda_identifiability.py --reps 600` (Anaconda Python 3.12;
   the base 3.13 has no numpy). Output committed at
   `runs/_sim/sim_lambda_identifiability_reps600.txt`.

   The claim **holds**. Holding the world fixed at SHARED and varying only
   direction-estimate quality:

   | ρ_stance | ρ_harm | pop SEL | P(SEL≥2) |
   |---|---|---|---|
   | 0.55 | 0.55 | 0.99 | **0.020** |
   | 0.55 | 0.70 | 1.26 | 0.047 |
   | 0.55 | 0.85 | 1.66 | 0.247 |
   | 0.40 | 0.90 | 2.15 | 0.583 |
   | 0.30 | 0.95 | 2.50 | **0.615** |
   | 0.85 | 0.55 | **0.59** | 0.007 |

   So the measured range is **2.0% → 61.5%**, where the contract and the draft
   both say **1.8% → 64%**. The difference is Monte Carlo noise at 600 reps, not
   a contradiction — but the paper should cite the run it can show. **Change the
   draft to "2% to 62%"** and cite the committed output, or re-run at higher
   reps if the exact published endpoints matter.

   The inversion is a *population* quantity, so it is exact and unchanged in both
   quick and full runs: at ρ_stance=0.85 / ρ_harm=0.55 the population SEL is
   **0.59**, below 1 — the rule silently flips, precisely as
   `RESEARCH_CONTRACT.md` §5.1 states. That sentence can stand as written.

   **Note for Windows:** several repo scripts crash printing Unicode to a cp1252
   console (`direction_norms.py` on `‖`, and `kappa_from_csv.py` had the same
   bug on `κ`). Prefix with `PYTHONIOENCODING=utf-8` or the run dies after doing
   all the work.
3. **BBQ citation** — confirm exact venue and full author list.
4. **Page limit** — the draft measures ~6.1 pages against a hard 5. See
   `PAGE-BUDGET.md` for a prioritised cut list that loses no result.
