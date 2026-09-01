# Experimental plan — Run 3: behavioural bias vectors and a bias taxonomy

**Status:** code written, tested, nothing run. Suite: 281 passed, 1 skipped,
4 xfailed. Written 2026-09-01, before any run-3 data exists.

**Audience:** a reviewer who knows interpretability and has not seen this
project. No code is reproduced here; every step names the file that implements
it. Read §7 (controls) and §9 (risks) before judging §8 (what we expect).

---

## 1. The question

Do different social-bias categories occupy distinct, causally separable
directions in a model's residual stream — or is "bias" one mechanism that
different topics merely trigger?

Three sub-questions, in dependency order. Each is only meaningful if the one
before it succeeded:

1. **Does a per-category bias direction exist and reproduce?** (§5, §6.3)
2. **Is it causal — does adding it make the model stereotype, and removing it
   stop that?** (§6.4)
3. **Do the categories cluster, and if so is the shared thing bias?** (§6.5)

Question 3 is the point of the study. Questions 1 and 2 are what make an answer
to it mean anything.

---

## 2. The contrast, and why this one

```
V_C,raw = mean(residuals | model produced a stereotyped answer)
        − mean(residuals | model said the context is under-informative)
```

Both arms come from the **same prompts**. Only the model's own behaviour differs.
This is a *behaviour-derived* contrast, and choosing it is a deliberate reversal
of the previous run's direction of travel.

**What it buys.** The vector is defined by the behaviour we then try to steer,
so Phase 3's toggle test is a direct test of the same object — not a proxy for
it. An annotation-derived contrast (the alternative, still live as run R1,
`scripts/run2_annotation_contrast.py`) yields a vector defined by a dataset label
that the model never sees, which is cleaner but is not obviously the thing that
causes the behaviour.

**What it costs.** Three known defects, each of which is *measured* here rather
than assumed away. These are not hypothetical: all three were quantified on this
project's run-1 data.

| defect | what goes wrong | measured before | handled by |
|---|---|---|---|
| **M1** | a category the model never stereotypes has an empty arm, so "no vector" and "no behavioural tilt" are indistinguishable | floor↔tilt correlation **+0.66 to +0.77** across five models | categories below 32 per arm are marked `UNTESTABLE`, never reported as negative (§6.2) |
| **N6** | the labeller *is* the label, so labeller error enters the vector — and its failure mode is positional, which correlates with the label | heuristic parser person-consistency **48–68%** against a 50% chance line | LLM judge with an order-swap qualification that blocks (§4, §7.2) |
| **refusal** | bucketing by answered-vs-declined puts the model's shared abstention direction into every vector | — | refusal de-coupling, measured at the cross-category level (§7.4) |

The third is the one that decides whether §1's question 3 has an answer, and it
is new to this design. It is discussed in full in §7.4.

---

## 3. Data

**BBQ** (Parrish et al. 2022), `datasets/BBQ_Prompt_Sets/*.jsonl`. Verified:
**51,628 rows across 10 category files**, forming **25,814 matched
ambiguous/disambiguated pairs** with zero unpaired rows.

**Five model targets, three families:** Qwen (14b, 7b, 1.8b), Yi (6b), Gemma
(2b). `notes/13` §1 sets the bar at "at least two model families" — the three
Qwens are one family, so Yi and Gemma are what actually meet it. qwen-14b is the
discovery model; the rest replicate.

**One judge for all five: qwen-1.8b.** Deliberate. A single instrument across
every target means a judge quirk cannot be mistaken for a model difference,
which is the comparison the taxonomy rests on; per-target judges would confound
it. This includes the cell where qwen-1.8b judges its own completions.

That self-judged cell carries a narrow risk — a model may systematically misread
the phrasings it favours, making labelling error correlate with the outputs
rather than being noise — so it is **settled empirically rather than by
argument**: those completions are re-judged with yi-6b and the agreement is
reported (`R3f_selfjudge_crosscheck`). Above 0.95, self-judging is not distorting
the labels; below, the cell is reported with the caveat or re-extracted from the
independent labelling.

| set | selection | used for |
|---|---|---|
| **under-informative** (`context_condition == "ambig"`) | 400 per category, evenly spaced across the file | the contrast. No named answer is correct, so naming a person *is* stereotyping. |
| **adequately-informative** (`disambig`) | 100 per category | the **task-capability control** only (§7.5). Retained, never used in the primary. |

Even spacing rather than the first N is deliberate: BBQ orders rows by scenario
template, so the head of a file is one or two templates and has almost no
variation. Implemented in `scripts/run3_behavioural_contrast.py` → `_load_rows`.

**The answer key.** `additional_metadata.csv` supplies `target_loc`, BBQ's own
index of which answer counts as biased. It already accounts for question
polarity and is used as shipped. This matters: it removes "how do you know you
labelled these correctly?" from the list of things the paper defends. Rows
without a key are labelled `UNKNOWN` and excluded — never guessed.

**Descoped 2026-09-01:** synthetic categories (e.g. political affiliation). Ten
BBQ categories are sufficient, and synthetic sets would carry LLM-generated
labels being compared against human-annotated ones, confounding any
`V_politics`-vs-`V_race` difference with label provenance.

---

## 4. The prompt, and the judge

**The prompt carries the option list and no formatting instruction.** Format:
`{context} {question} Pick one of three options: {ans0}, {ans1}, {ans2}` —
matching `src/bias_steer/datasets.py` so run 3 asks what the 2025 pipeline asked.

The system prompt is **empty**. The previous default instructed the model to
"give the clear, definitive answer" so a regex could parse it; the judge removes
that need. The instruction was never free — it pushes instruction-following
activations into the residual stream at exactly the capture position, identically
in both arms, which is the shape of thing that inflates cross-category cosines.
**Declared cost:** this breaks prompt-level comparability with runs 1 and R1.

**Labelling is done by an LLM judge**, `scripts/pilot/llm_judge.py`, documented as
a versioned artifact in `docs/judges/v2-bbq-choice-llm.md`. Two variants share one
rubric and one output vocabulary:

- **v3 (default, local).** A small model on the box. The five verdicts are single
  distinct characters and the answer is an **argmax over their logits** — one
  forward pass, no free-text surface, so a parse failure is impossible. Fully
  reproducible under a pinned revision.
- **v2 (`--judge-backend openai`).** `gpt-4o-mini`, same rubric, generates a
  token. Kept for a same-units comparison against the local judge.

**The judge is asked a different question than it answers.** It is asked *which
option did the response select?*, in a positional vocabulary that never contains
the word "stereotype". The mapping to `BIASED_TARGET` / `BIASED_DISTRACTOR` is
done afterwards from `target_loc`. Asking directly for `BIASED_TARGET` would mean
handing the judge the answer key — a judge that merely echoed it would be
indistinguishable from one that read the response.

**The judge must not be the target model.** Self-labelling makes the bucket
depend on the disposition being measured. The runner refuses it.

| judge label | arm |
|---|---|
| `BIASED_TARGET` | `R_biased` |
| `REFUSAL` (incl. picking BBQ's own "cannot be determined" option) | `R_refusal` |
| `BIASED_DISTRACTOR` (named the non-stereotyped person) | **neither** |
| `UNKNOWN` / format failure | neither; counted and reported |

`BIASED_DISTRACTOR` is excluded because it is a *choice*, not an abstention.
Folding it into `R_refusal` would make the contrast "stereotyped vs anything
else", and the direction would then partly encode *which person was named*.
`--distractor-in-refusal` flips this; the counts are reported either way.

---

## 5. The statistics

**Direction.** Difference of means over the two buckets, per layer →
`(n_layers, d_model)`. No fitted parameters.
`scripts/pilot/behavioural.py` → `behavioural_direction`.

**Comparing two directions.** Per-layer cosine, collapsed by a norm-weighted mean
over layers, with the unweighted median reported alongside as a pre-declared
sensitivity. Per-layer norms span orders of magnitude within one model, so a
flattened cosine would mostly re-measure the norm profile. Layer 0 is exactly
zero by construction and propagates as NaN, which is filtered.

**The extraction floor — the core statistic.** Split the category's items in half
**stratified by bucket**, extract a direction from each half, take the cosine;
repeat 400 times; report the mean with a percentile bootstrap 95% CI.
This answers *how much does the direction move when nothing changed?* Without it
no cosine in §6.5 can be interpreted. `bucket_floor`.

Stratifying by bucket matters: cutting a pooled list blind lets one half drift
toward one arm, and a difference of means over unbalanced arms adds variance that
is not about reproducibility.

**The negative control.** The identical computation with bucket labels shuffled,
n held fixed — **averaged over 20 independent shuffles**, not one. A single
shuffle draw is a nuisance parameter, not a null: its realised imbalance leaves
part of the true direction inside the "null", and a split-only interval does not
span that. Measured on planted data, one draw versus twenty moved the control by
**0.11** while the reported interval was **0.005** wide. The 20 draws share the
400-split budget, so runtime is unchanged. `shuffled_bucket_control`.

**Decision rule.** A category reproduces iff
`CI_lower(observed) > CI_upper(control)`; overlapping intervals are
`INDETERMINATE`. **There is no threshold constant anywhere.**

**Known limitation, stated plainly.** These intervals are Monte-Carlo error over
splits, not sampling error over items, so they narrow as the split count rises.
The verdict therefore means *"the labels give a more reproducible direction than
shuffled labels, on this sample"* — it carries no effect size. **Report the floor
magnitude beside every verdict**, against the positive control (§7.3) as the
same-units yardstick for what a direction that certainly exists looks like.

---

## 6. Procedure

### 6.0 Environment · `scripts/preflight.py`

`python3 -m scripts.preflight --load-model qwen-1.8b`. ~21 checks: three known
dependency collisions by name, BBQ files, the answer key, disk, CUDA, VRAM
against qwen-14b, the test suite, and one real forward pass. **Nothing starts
until it prints `Green`.** The three pins (`numpy<2`, `pillow>=9.1`,
`jinja2>=3.1`) must be installed *before* anything imports torch.

### 6.1 Gates · `scripts/overnight_queue.sh`

All four run **before** any long job, so the run becomes unattended in ~30 minutes
rather than ~2 hours. Each exits non-zero and stops the queue.

| gate | what it proves | ~time |
|---|---|---|
| **1** | R1's capture path executes and persists well-formed residuals | 2 min |
| **2** | R1's estimator recovers a topic-identity direction | 10 min |
| **R3-1** | run 3's `generate` → `judge` → `extract` chain executes end to end; residual shapes match sidecar ids; completions are non-empty | 5 min |
| **R3-2** | run 3's *own* estimator recovers a topic-identity direction | 10 min |

Gates R3-1 and R3-2 exist because this GPU path has never executed anywhere. A
null taxonomy measured with broken code is indistinguishable from a real one.

### 6.2 Generate and capture (GPU) · `run3_behavioural_contrast.py generate`

Per category: 400 under-informative items generated **twice** — canonical option
order, and with the two named options swapped — plus residuals captured at the
**final prompt token** (`blocks.{i}.hook_resid_pre`, all layers, index `-1`).

Capture is before any answer token exists, so the direction cannot encode the
output its label was read from. The captured prompt is byte-identical to the one
that produced the completion.

Outputs: `responses.jsonl` (every completion verbatim, both passes),
`prompts.jsonl`, `residuals/`, `capture_site.json`, `queue_manifest.json`.

**Cost per model:** 8,000 generations + 5,000 forward passes. Residuals:
~3.8 GB (qwen-14b), ~2.0 GB (qwen-7b, yi-6b), ~0.6 GB (gemma-2b), ~0.8 GB
(qwen-1.8b) — **~9 GB total**, gitignored, sync off-box before the machine dies.

**Whole-queue budget** (generation rate is the uncertain term — `notes/14` §13.4):

| step | ~generations | hours |
|---|---|---|
| four gates | — | 0.8 |
| R3 generate ×5 | 40,000 | 0.7–1.4 |
| R3 judge ×5 + cross-check | 24,000 | 0.3–0.6 |
| R3 extract ×5 (CPU) | — | 1.2–1.7 |
| R3 toggle ×5 (Phase 3) | 102,000 | 1.4–2.8 |
| R3 cross-application, qwen-14b (Phase 4.1) | 136,000 | 1.9–3.8 |
| **R3 total** | | **6.3–11.1** |

At the slow end this exceeds a 10-hour window. The trim lever is
cross-application — `--n-eval 80 → 60`, or a single alpha instead of two — which
roughly halves the most expensive step and is a flag, not a code change. Decide
after the first model's timings land, not in advance.
| R1a, after R3 | — | 0.8–1.2 |

### 6.3 Judge, then extract (CPU) · `judge`, then `extract`

`judge` runs the order-swap qualification **first** and labels nothing if it
fails. `extract` then gates on `scripts/pilot/verifier.py` — the termination gate
— before any number is printed, and computes buckets, floors, the negative
control, the refusal de-coupling, the cosine matrix, PCA and the permutation null.

Every analysis after `generate` reads the cached residuals, so **all of it can be
redone after the GPU is returned**. That is a hard requirement, not a convenience:
the previous run discarded its residuals and every follow-up question became
another rental.

### 6.4 Toggle test and cross-application (GPU) · `steer`

Inject `±α·V_C,unit` at every layer's `resid_pre` during generation, and judge the
result. Three things are non-negotiable here:

- **Unit-normalise per layer**, then set α as a multiple of *that layer's own mean
  residual norm*, so the dose is dimensionless and comparable across layers,
  categories and models. Verified exact across a 1000× norm spread.
- **α is reported as a curve** (`0.25 0.5 1.0 2.0`), never chosen post hoc by
  which value gives the best flip rate.
- **Every dose runs against a norm-matched random vector** and against the
  informative-task control.

**Staged deliberately.** The diagonal (Phase 3 — each vector on its own
category, `--n-eval 120`) runs first for both models, because cross-application
is only meaningful if a vector is causal on its own category. Phase 4.1 — the
full 10×10 cross product at `--n-eval 80`, ~136,000 generations — runs last. If
the window ends early you still have the phase that gates the other.

### 6.5 Taxonomy · part of `extract`

Pairwise layer-wise cosine matrix (with the full per-layer profile retained), PCA
over unit-normalised directions, hierarchical clustering, and a permutation null
that shuffles bucket labels **within** each category — never across, which would
make fake groups topic-heterogeneous while real ones are topic-homogeneous.

---

## 7. Controls, and what each failure means

### 7.1 Task capability — the informative arm
Can the model pick the right person when the context says who? Without it,
"the model is biased" cannot be told apart from "the model cannot do the task."
Also re-run under steering, so an intervention that merely breaks the model is
visible as such.

### 7.2 Position bias — two separate checks
- **C-1, the judge's:** every qualification item judged twice with the option
  list reversed and mapped back. Below **0.95** agreement, nothing is labelled.
  Validated: 1.000 for a competent stub judge, 0.000 for one that always takes
  the first-listed option.
- **The model's:** the second generation pass, with the named options swapped in
  the prompt, compared bucket-for-bucket. Note precisely what this measures — the
  *model's* presentation-order dependence as seen through the heuristic parser,
  which is why C-1 covers the judge separately.

A third check, `option_order_invariance`, is computed and **explicitly labelled
vacuous**: it scores ~1.0 whatever the labeller does, because reordering the
option list does not change which name appears first in the response text. It is
retained as a sanity check and must never be reported as the position-bias
control.

### 7.3 Positive control — Gate R3-2
Race-themed prompts against gender-themed prompts, through run 3's own
estimator. Topic identity is linearly present if anything is. **If this fails,
stop: nothing downstream can be read.** Its residuals are persisted, so it can be
re-read at a different split count or per layer.

### 7.4 Refusal de-coupling — the control this design turns on

Bucketing by answered-versus-declined puts the model's shared abstention
direction into every `V_C`. It is common to all ten categories **by
construction**, so it inflates every cross-category cosine — and §1's question 3
would read that as "a universal bias mechanism".

Two readings, one measurement:

```
per-category:   REFUSAL-DOMINATED iff |cos(V_C, V_refusal)| ≥ √(CI_lo(floor_C) · CI_lo(floor_refusal))
cross-category: does median off-diagonal |cos| SURVIVE orthogonalising V_refusal out?
```

**The cross-category comparison is the decision**, and the per-category test is
not sufficient. Demonstrated on planted data where each category had an
*independent* bias direction plus a shared refusal component: every category read
`BIAS-SPECIFIC` (|cos| 0.894 against a ceiling of 0.974) while the cross-category
cosine went **+0.809 → −0.004** under orthogonalisation. All the shared structure
was refusal, and the per-category verdict said nothing was wrong.

**The permutation null does not catch this either** — shuffling bucket labels
destroys the bias and refusal components alike, so the observed value beats the
null (p = 0.024 on that same planted data) regardless.

**This control requires `V_refusal_base`** from `src/bias_steer/refusal_extract.py`
plus its own floor. Without them the ceiling collapses to zero, nothing can fire,
and the run reports the control as **vacuous rather than passed**. Until it is
supplied, the cosine matrix and the PCA are not readable as bias results.

Orthogonalisation is reported alongside the raw direction, never instead of it: a
noisily-estimated reference can only be partly projected out, so a projected
result is a lower bound. A surviving cosine is evidence; a collapsing one is
decisive.

### 7.5 Negative control
§5. Runs automatically per category.

---

## 8. What we are looking for

### 8.1 The result that supports "distinct bias directions"

- most categories `TESTABLE`, with `n_biased ≥ 32`;
- within-category floors **well above** their shuffled controls, and of a
  magnitude comparable to Gate R3-2's topic control;
- **cross-category cosine that survives refusal orthogonalisation**;
- clustering that beats the within-category permutation null;
- `+α` raising the stereotyped-answer rate over baseline **and over the
  norm-matched random vector**, `−α` lowering it, without collapsing task
  accuracy on the informative control;
- cross-application weaker than within-category — a race vector should move race
  prompts more than gender prompts.

### 8.2 The result that supports "one mechanism"

Same as above, except cross-application works about as well as within-category
and the cross-category cosine stays high **after** orthogonalisation. This is a
real answer, not a failure — but it is only distinguishable from §8.3 by the
orthogonalisation step.

### 8.3 The artifact we most expect

High cross-category cosine that **collapses** under refusal orthogonalisation.
Reading: we measured the abstention direction, not bias. Given the contrast is
literally answered-versus-declined, this is the single most likely outcome and
the paper must be willing to report it.

### 8.4 The honest null

Floors that do not beat their shuffled controls, **with Gates R3-1 and R3-2
green**. That is publishable: the positive control proves the instrument works,
so the null is about bias rather than about the code. It also directly informs
the open question between this design and the annotation-derived R1.

### 8.5 What would invalidate the run

Any gate red; C-1 below 0.95; a majority of categories `UNTESTABLE`; the
refusal-decoupling control vacuous; or `steer` shifts that a norm-matched random
vector reproduces.

---

## 9. Risks, stated before running

1. **The refusal confound (§7.4)** is the largest and is expected to bite.
2. **`generate` and `steer` have never executed.** Gates R3-1/R3-2 exist for
   this. One residual unknown: batched generation passes an attention mask with
   an unbatched fallback, and **neither branch has run**. If this
   TransformerLens build accepts the mask but ignores it, padding leaks silently.
   Gate R3-1's "identical under option swap" count is the canary — near 100%
   means generation is broken, not that the model is stable.
3. **M1 selection.** Bucket sizes vary by category and correlate with how biased
   the model is. `UNTESTABLE` is reported by name; small-bucket categories are
   not negatives.
4. **Judge accuracy is unmeasured.** C-1 establishes self-consistency, not
   correctness. Hand-labelling (C-3) was descoped 2026-09-01 as unnecessary for
   an easy reading task. **Report accordingly**: measured consistency,
   unmeasured accuracy. Revisitable from persisted files with no GPU.
5. **Prompt comparability** with runs 1 and R1 is broken by the empty system
   prompt (§4).
6. **Model revision SHAs are not pinned.** A declared reproducibility gap.
   Record `pip freeze` and the resolved revisions at run time.

---

## 10. What gets written

```
runs/r3_behavioural_<model>/
├── capture_site.json              the chat-template tail and the index used
├── prompts.jsonl                  every prompt, verbatim
├── responses.jsonl                every completion, both passes, verbatim + sha256
├── judge_qualification.json       C-1, and whether it passed
├── judge_labels.jsonl             per item: verdict, label, judge version + model
├── residuals/                     (n, n_layers, d_model) float32 + sidecars
├── directions/<Category>.npy      the extracted V_C
├── queue_manifest.json            per-step exit codes and declared outputs
├── report_behavioural.json        buckets, floors, controls, cosines, PCA, null
└── report_steering.json           dose curves, random control, task control
```

`.npy` files are gitignored — several GB, and the larger ones exceed GitHub's
100 MiB limit. **Sync them off the box before it dies** (`sync_from_box.ps1`,
which also validates `.npy` headers). Losing them turns every follow-up into
another rental.

---

## 11. Decisions on record

| decision | date | rationale |
|---|---|---|
| Behavioural contrast is primary; R1's annotation contrast retained as the comparison | 2026-08-31 | the vector should be the object being steered |
| LLM judge replaces the heuristic parser | 2026-09-01 | N6: positional error correlated with the label |
| Local judge is the default | 2026-09-01 | reproducible, free, no key; API judge kept for comparison |
| C-3 (judge accuracy hand-labels) descoped | 2026-09-01 | easy reading task; C-1 already blocks |
| Synthetic political category descoped | 2026-09-01 | ten BBQ categories suffice; label-provenance confound |
| Empty system prompt | 2026-09-01 | no instruction-following activations at the capture site |
| Minimum bucket 32, not 15 | 2026-09-01 | Arditi/Joad standard; at 15 a split-half leaves 7 per half |

---

## 12. File index

| what | file |
|---|---|
| run-3 runner (`generate` · `judge` · `control` · `extract` · `steer`) | `scripts/run3_behavioural_contrast.py` |
| buckets, floors, refusal de-coupling, PCA, permutation null | `scripts/pilot/behavioural.py` |
| the LLM judge, both backends, C-1 qualification | `scripts/pilot/llm_judge.py` |
| judge version record | `docs/judges/v2-bbq-choice-llm.md` |
| shared estimator primitives, bootstrap, decision rule | `scripts/pilot/analysis.py` |
| BBQ loading, pairing, subsampling | `scripts/pilot/pairing.py` |
| artifact verifier (the termination gate) | `scripts/pilot/verifier.py` |
| queue runner with real exit codes | `scripts/pilot/queue.py` |
| environment preflight | `scripts/preflight.py` |
| the unattended queue, all gates | `scripts/overnight_queue.sh` |
| steering hooks, shape assertions | `src/bias_steer/steering.py` |
| refusal direction extraction | `src/bias_steer/refusal_extract.py` |
| BBQ answer key and metadata assembly | `src/bias_steer/datasets.py` |
| tests | `tests/test_run3_behavioural.py` (21), `tests/test_bias_taxonomy.py` |
| the alternative contrast (run R1) | `scripts/run2_annotation_contrast.py`, `docs/HANDOFF_R1_ANNOTATION_CONTRAST.md` |
| defect register (S1–S5, N1–N6, M1–M3) | `results/writeups/12-retrospective.md` |
