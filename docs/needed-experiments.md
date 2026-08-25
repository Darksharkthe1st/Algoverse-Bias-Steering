# Needed experiments

Experiments we intended to run but never finished, written for the agent refactoring this codebase.
Each entry says **what to run**, **what to log**, and **what data is required to actually draw a
conclusion** — because most of the original runs failed that last test (they produced a number but not
enough structure to defend it).

Context and current numbers are in `../results_analysis/REPORT.md`; the branch inventory is in
`../results_analysis/BRANCH_MAP.md`. The scoring scripts there (`parse_logs.py`, `load_csvs.py`,
`aggregate.py`) already implement the log/CSV format and can be reused as the evaluation harness.

---

## 0. Lock these conventions down first (blocks every comparison below)

The old results are hard to compare because three things were never standardized. The refactor should
pick one definition for each and record it, then re-run the anchors so everything is on one scale.

**0.1 — Where the vector is injected and how the coefficient scales.** Three incompatible conventions
exist in the archive:
- `old-results`: `value[:, :, :] += coeff * steer_vec[layer]` at **every** `blocks.{l}.hook_resid_pre`,
  all token positions, **raw** (un-normalized) vector, same scalar `coeff` at every layer → total
  injection grows with model depth.
- `farhan-batch-coeffs`: `coeff / n_layers` at every layer → fixed total budget regardless of depth.
- `aryaman_adaptive_coeffs_and_norms`: **unit-normalize** each layer's vector first, then
  `budget / len(layers)`, applied only to layers 18–24.

  *Decision needed:* one canonical injection rule. Recommendation: normalize each layer's vector to
  unit norm, apply a single `coeff` at a chosen layer band, and report `coeff` in normalized units so a
  value means the same thing across models and layers. **Data to justify the choice:** re-run the
  coeff-finding sweep (§7) under each convention on 2 models and show which gives the smoothest,
  most monotonic dose-response.

**0.2 — Per-example judge reliability.** The GPT judge (`gpt-4o-mini`, unpinned alias) was called
once per response and is nondeterministic (our own validation found ±1–2 label drift on re-runs).
*Decision needed:* pin a model/version and judge each response **k=3–5 times**, keep the majority label,
and record per-example agreement. Without this we cannot separate a real steering effect from judge noise.

**0.3 — A coherence gate.** Several "successes" were actually the model degenerating (see refusal, §11,
and the gemma opinion-overshoot). *Decision needed:* log a coherence signal for every generation —
perplexity under the base model, or a short "is this fluent English? yes/no" judge pass — and treat any
run where >X% of outputs are incoherent as void, not as a result.

---

## 1. Opinion spectrum (graded 1–5 judge)  — never run

**Status:** the judge prompt was upgraded from binary (`opinionated`/`neutral`) to a graded
1–5 scale (1 Super-Neutral … 5 Super-Opinionated) on `farhan-opinion-spectrum`, but **no run ever used
it** — every committed CSV is still binary. *(Confirmed: this run never happened.)*

**Goal:** measure the *degree* of opinionatedness and whether steering moves the whole distribution, not
just the count over a binary threshold.

**Setup:** reuse the existing pipeline; swap in the 5-label judge from
`origin/farhan-opinion-spectrum:experiments/farhan-experimentation.ipynb` (cell 36). Run the standard
9 models on the GPT comparison set at each model's tuned coeff, plus a coefficient sweep on 2 models.

**Log:** the 1–5 label per arm (init / opinion-steered / neutral-steered), not a 3-bucket collapse.
Keep the raw judge reasoning text.

**Data needed to conclude:** the mean/distribution shift on the 1–5 scale per arm, with per-example
reliability (§0.2). Success = the neutral vector shifts the distribution down by ≥1 scale point and the
opinion vector up, monotonically with coefficient, on ≥6/9 models.

**Effort:** low (judge-only change, harness exists). **Priority: HIGH** — directly upgrades the core metric.

---

## 2. CrowS-Pairs run, completed  — crashed mid-run

**Status:** `farhan-fixed-crows` added the fix (each CrowS sentence turned into a question via
`+ " Why does this make sense?"`) and lowered coeffs, but the OpenAI judge call crashed before any
transition matrix was written — `Batched_Gen.csv` has headers only.

**Goal:** does steering remove *stereotype* bias (CrowS), not just opinionatedness on comparison prompts?
This matters because the opinion vector already failed to transfer to CrowS in the archive
(`opinion-vec→CrowS ≈ 0pp induce`).

**Setup:** `datasets/Crows_Pairs/crows_pairs_anonymized.csv`, the `load_crows_pairs` question fix, 9
models, tuned coeffs, n≥100 test prompts. Add retry/back-off around the judge call (the original crash).

**Data needed to conclude:** baseline vs neutral-vector vs opinion-vector marginals with n≥100 and the
coherence gate (§0.3). Success for de-bias = neutral vector lowers opinionated rate on CrowS by a
margin clearly above judge noise; report per stereotype category (race, gender, religion, …).

**Effort:** low. **Priority: HIGH** — it's the dataset where the method is most likely to fail, so it's
the most informative.

---

## 3. Ground "bias" in a benchmark with a correct answer (BBQ bias score)  — never done properly

**Status:** BBQ was only ever scored with the opinionated/neutral judge. But BBQ has **ground truth** —
in ambiguous contexts the correct answer is "unknown", and a biased model picks the stereotyped group.
We never used that structure, so we measured *opinionatedness*, not *bias*.

**Goal:** prove that de-biasing reduces *stereotype-aligned* answers, using BBQ's real labels — the
strongest possible evidence for the "we can prove bias" claim (Theme C).

**Setup:** `datasets/BBQ_Prompt_Sets/*.jsonl` (each row has answer options, the correct `label`,
`question_polarity`, and `context_condition` ambig/disambig). Parse the model's multiple-choice answer
(not a judge). Compute the standard **BBQ bias score** and accuracy, in ambiguous vs disambiguated
contexts, for baseline / neutral-vector / opinion-vector arms, across the 9 models.

**Log:** chosen option per prompt, correctness, and whether the wrong choice was the
stereotype-consistent one.

**Data needed to conclude:** BBQ bias score (%) and accuracy per arm, with n large enough per category
(BBQ is big — use ≥200/category). Success = neutral vector moves the ambiguous-context bias score toward
0 (more "unknown") **without** tanking disambiguated-context accuracy. This is the experiment that turns
"the judge's label changed" into "measured bias went down."

**Effort:** medium (needs an MC-answer parser + BBQ scorer; no LLM judge required). **Priority: HIGH.**

---

## 4. Rubric-v2 re-judging with an agreement gate  — set up on team-kit, not executed

**Status:** `team-kit` built an ordered 6–8-way rubric (unjudgeable → incoherent → meta → hard refusal →
soft refusal → non-engagement → stance-factual → stance-evaluative) with a calibrate-then-freeze
protocol, but it is **not frozen and no gold-set annotation (Cohen's κ) has run.** The binary judge is
known to conflate ≥4 behaviors (factual decisiveness vs taste vs both-sidesing vs refusal).

**Goal:** replace the binary label with a construct that separates "took a stance" from "stated a fact"
from "refused", so the steering vector can be shown to move the *intended* behavior.

**Setup:** freeze `RUBRIC_v2` (pick the category count); annotate a disjoint calibration set (~30) then a
frozen scored set (~150) of archived responses; compute per-category κ. Then re-judge a sample of the
existing logs with the frozen rubric via `src/judging.py` / `src/recount.py` (already on team-kit).

**Data needed to conclude:** per-category Cohen's κ (gate ≥0.70), and the re-labeled breakdown of what
the opinion vector actually moves (ideally: stance-evaluative up, factual/refusal flat). If κ fails,
that itself is the finding (the construct isn't measurable as posed).

**Effort:** medium–high (needs human or high-quality-judge annotation). **Priority: HIGH for the paper**,
because every other number depends on the judge being trustworthy.

---

## 5. Combined-dataset vs single-dataset vectors, head-to-head  — missing baselines

**Status:** `farhan-combined-dataset` built one vector from BBQ+CrowS+GPT and showed the *neutral* side
generalizes while the *opinion* side fails on most models. But there are **no matching single-dataset
per-model baselines** in that run, so "does combining help?" can't be answered.

**Goal:** quantify whether a combined-dataset vector de-biases/induces better than the best
single-dataset vector, per model.

**Setup:** for each of the 9 models, build 4 vectors (BBQ-only, CrowS-only, GPT-only, combined) on a
fixed train split, and evaluate all 4 on the **same** held-out mixed test set at matched coeff.

**Data needed to conclude:** a 9×4 table of debias Δ and induce Δ on the identical eval set. Success for
"combining helps" = combined ≥ best single on de-bias for a majority of models. (Hold coeff and eval set
fixed — the archive confounded these.)

**Effort:** medium. **Priority: MEDIUM.**

---

## 6. Vector normalization ablation  — notebooks left unexecuted

**Status:** `aryaman_adaptive_coeffs_and_norms` unit-normalizes each layer's vector before injection
(so one coeff is uniform across layers), motivated by the ~1400× per-layer norm spread. The
normalized-vector notebooks have **empty result cells** — no head-to-head vs the raw-vector approach.

**Goal:** does unit-normalizing each layer's steering vector (then retuning one global coeff) give a
cleaner, more transferable effect than adding the raw difference-of-means?

**Setup:** on 3 models, run the coeff sweep twice — raw vector (old-results rule) vs per-layer
unit-normalized vector — on the same eval set. Tie to convention decision §0.1.

**Data needed to conclude:** dose-response curves for both; compare (a) smoothness/monotonicity, (b) peak
effect, (c) the coefficient window width before collapse. Success = normalized gives a wider usable
window and/or higher peak with fewer per-model coeff surprises.

**Effort:** low–medium. **Priority: MEDIUM** (mostly resolves §0.1).

---

## 7. Coefficient / layer-placement ablation  — done ad hoc, never systematically

**Status:** coeff-finding sweeps exist but each branch used a different injection rule (§0.1) and steered
"all layers" or "layers 18–24" without a controlled layer comparison. Aryaman's notes hint at strong
layer effects ("10–16 works, 18–22 nonsense, 25–30 → EOS, 29–32 → neutral").

**Goal:** find where in depth the opinion/neutral directions live and the coefficient window that steers
without breaking coherence.

**Setup:** on 2–3 models, sweep (a) single-layer injection across all layers at fixed normalized coeff,
and (b) coefficient 0→N at the best layer band. Log the coherence signal (§0.3) at every point.

**Data needed to conclude:** per-layer effect size + coherence, and a dose-response with the coherence
gate overlaid, so the "sweet spot" is defined as *max effect subject to coherence*, not just max label
flips. This directly explains the gemma inverted-U (overshoot reverts to neutral) we saw.

**Effort:** medium. **Priority: MEDIUM.**

---

## 8. Cross-model vector transfer  — not tested

**Status:** we tested cross-*dataset* transfer (vector from X applied to test set Y). We never tested
cross-*model* transfer (a vector computed on model A applied to model B), which speaks to whether
"opinion"/"neutral" is a shared direction or model-specific.

**Goal:** does a steering vector from one model steer a different model (same family / across families)?

**Setup:** compute vectors on Qwen-1.8B and Llama-3-8B; apply (with dimension/layer matching where
possible, within-family first) to sibling models. Requires the refactor to handle differing hidden dims
and layer counts (project/pad or restrict to same-architecture families).

**Data needed to conclude:** debias/induce Δ for transferred vs native vectors on the same eval set.
Success = transferred vector retains a meaningful fraction of native effect within a family.

**Effort:** medium–high (architecture matching). **Priority: MEDIUM** (novel, good for the paper's
"shared direction" story).

---

## 9. Complete the Grok controversial-questions run  — partial

**Status:** `farhan-grok-questions` ran a 50/50 blend of Grok-generated controversial questions
(`Grok_100/200.txt`, *confirmed generated by Grok*) + GPT prompts. It worked on Qwen-small/Yi at moderate
coeffs but inverted/saturated on large models, and used two inconsistent coeff blocks.

**Goal:** a clean, single-config run on the full Grok set to characterize steering on genuinely
controversial (not just comparison) prompts.

**Setup:** full `datasets/Grok_Questions/Grok_200.txt` (pure, not blended), tuned coeff per model,
coherence gate. Optionally re-judge with the spectrum (§1) since controversy is graded.

**Data needed to conclude:** baseline vs steered marginals (or spectrum) at one config, n≥100, coherence
logged. **Priority: LOW–MEDIUM.**

---

## 10. Synthetic steering, done with matched distribution  — original attempt failed

**Status:** `farhan-synthetic-steering` used 100 GPT-authored "As an AI I can't take a position…"
sentences as the neutral cluster; both +v and −v collapsed output toward neutral — the vector encoded a
*self-generated-text vs external-text* axis, not an opinion axis.

**Goal (optional):** salvage synthetic steering by removing the distribution-shift confound.

**Setup:** generate the neutral cluster from the *same model's own* outputs (steer/prompt it to be
neutral) rather than pasted GPT text, so both clusters are on-distribution; or subtract a
"text-source" control direction.

**Data needed to conclude:** compare synthetic-vector dose-response to the dataset-derived vector on the
same eval set; success = synthetic no longer collapses and tracks the dataset vector. **Priority: LOW.**

---

## 11. Refusal axis  — de-prioritized (abandoned for incoherence)

**Status (per you):** safe/unsafe steering was run briefly; the models became incoherent, so it looked
pointless and we moved on. Separately, the one archived refusal headline is a **known bug** — the
"1% → 27% unsafe on Llama-2-7B" result (`Refusal_To_Opinion.csv`, **Log 213**) loaded the
`Qwen-1_5-1_8B.pt` payload under a Llama filename and the hook broadcast a scalar (team-kit
`docs/REVIVAL_AUDIT.md`).

**If revisited, what it would take:** the incoherence is the whole problem, so this experiment is only
worth running with the coherence gate (§0.3) as a first-class constraint — find the coefficient that
increases refusal *while* staying fluent (report the refusal-rate/coherence trade-off curve), and fix the
vector-loading bug so the saved vector provably matches the model. Om's mechanistic ablation approach
(Arditi et al., on `om_experimentation`) is the more promising path than additive steering, since it
jailbroke Qwen cleanly in the qualitative demo. **Priority: LOW** unless the ablation direction is pursued.

---

## 12. Determinism check: same config + same code, run twice  — never verified

**Status:** the pipeline threads seeds through sampling and the train/test shuffle and
records them in each manifest *for* reproducibility, but no run has ever verified that two
identical runs actually produce identical results. Accidental nondeterminism (an unseeded
`random`/`torch` call, dict-ordering leakage) would go unnoticed.

**Goal:** confirm the reproducibility guarantee the seeds are meant to provide — identical
config + identical code ⇒ identical results — so any future accidental randomness is caught.

**Setup:** run the *exact same config* on the *exact same commit* twice, back to back.
Two layers:
- **Hermetic (do first, cheap):** with the stub model + stub judge already used in the phase
  tests, run `experiment.run` twice and diff outputs — belongs in the test suite / CI.
- **Full-stack:** run a small real config (e.g. 1 model, small n) twice on one GPU box.

**Log:** for both runs — the sampled example ids, the train/test split, the steering vector
(bytes / hash), `results.csv`, and the per-condition verdict counts.

**Data needed to conclude:** a diff of the two runs. **Deterministic parts must match
exactly:** sampled subset, train/test partition, response ordering, and — given identical
residuals — the built steering vector. **Known non-deterministic confounds to exclude from
the equality check:** the GPT judge (nondeterministic, §0.2 — pin/mock it or compare only
pre-judge artifacts) and CUDA float nondeterminism (can perturb generations/residuals). So
the hermetic stubbed run is the real determinism gate; the full-stack run compares only the
stable artifacts. Success = deterministic artifacts are bit-identical across the two runs.

**Effort:** low (hermetic) / low–medium (full-stack). **Priority: HIGH (validation)** — it
underpins every reproducibility claim the manifests make. (Engineering note mirrored in
`rewrite_notes.md` §8.)

---

## Priority summary

| # | experiment | effort | priority | blocked by |
|---|---|---|---|---|
| 0 | lock conventions (injection/coeff, judge reliability, coherence gate) | low | **do first** | — |
| 1 | opinion spectrum (1–5) | low | HIGH | 0.2 |
| 2 | CrowS run completed | low | HIGH | 0.3 |
| 3 | BBQ bias score (ground truth) | med | HIGH | — |
| 4 | rubric-v2 + κ gate | med–high | HIGH (paper) | — |
| 5 | combined vs single dataset | med | MED | 0.1 |
| 6 | normalization ablation | low–med | MED | 0.1 |
| 7 | coeff / layer ablation | med | MED | 0.1, 0.3 |
| 8 | cross-model transfer | med–high | MED | 0.1 |
| 9 | complete Grok run | low | LOW–MED | 0.3 |
| 10 | synthetic steering v2 | med | LOW | 0.1 |
| 11 | refusal (coherence-gated) | med | LOW | 0.3 |
| 12 | determinism double-run check | low | HIGH (validation) | — |
