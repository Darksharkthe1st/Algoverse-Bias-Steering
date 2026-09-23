# Experiment protocol — read this before Experiment 2

**Audience: the next Claude Code agent, and Jeremiah.**
This is a directive document. It is derived from a post-mortem of the
2026-08-20/21 session, in which roughly **half the GPU time produced artifacts
that were later superseded**. Ten result files on disk are dead. That waste was
almost entirely preventable, and every rule below traces to a specific incident.

Read §1 and §2 first. §3 is the evidence. §4–§9 are the rules. §10 is what went
right and must be preserved.

---

## 1. The one structural diagnosis

**We designed the experiment while executing it.**

Every serious problem in that session is a consequence of this one thing. The
plan (`notes/03-experiment-1-plan.md`) specified a contrast and a procedure but
did not specify the capture site, the estimator, the estimator's
hyperparameters, the intervention normalisation, or the confounds to rule out.
Those got decided *after* a paid GPU was already running, under time pressure,
one at a time, each triggered by a failure rather than by design.

Three consequences, in order of cost:

1. **Three complete method designs** were tried; two were discovered broken only
   at full scale, on the box.
2. **Every mid-flight change invalidated earlier artifacts**, so results had to
   be re-run and superseded files tracked through the writeup.
3. **The sunk-cost dynamic Jeremiah named** — "we were in too deep to start
   [over]" — is real and it degrades judgement. Once the meter is running, the
   pressure is to patch rather than stop.

**The fix is structural, not a matter of trying harder: no GPU is provisioned
until the pipeline has run end-to-end at tiny scale and the specification is
frozen.** Everything that broke could have been caught by a 20-item pilot on a
0.5B model, or by reading a file, or by a checklist.

---

## 2. The phase protocol — mandatory

Five phases with hard gates. **Do not enter a phase until the previous gate
passes.** If a later phase forces a change to an earlier decision, you have
failed a gate: stop, return to PLAN, and re-freeze. Do not patch forward.

```
PLAN ──▶ PILOT ──▶ FREEZE ──▶ EXECUTE ──▶ ANALYSE
  │        │          │          │           │
  └── no GPU ─────────┘          └─ GPU only from here ─┘
```

### Phase 1 — PLAN (no GPU, no model)
Produce the pre-registration in §4. Every field filled. Unfilled field = not
ready.

**Gate P1:** every field in the §4 template has a value, and every threshold has
a stated justification that does not depend on data you have not collected yet.

### Phase 2 — PILOT (CPU or smallest available model, ≤20 items per condition)
Run the **entire** pipeline end-to-end — loader, prompt construction, scoring,
labelling, bucketing, extraction, floor, matrix, null, report — at a scale where
a full pass takes under five minutes. The pilot's job is not to produce results.
Its job is to execute **every line of code the real run will execute**.

**Gate P2, all must hold:**
- Every code path ran; no `NotImplementedError`, no silently skipped branch.
- Every confound in the §5 checklist was explicitly checked and recorded.
- The positive controls in §6 pass at pilot scale.
- The output report renders with plausible-shaped numbers (not necessarily
  meaningful ones).
- Someone read 15 raw model outputs by hand and confirmed the parser agrees with
  a human on all 15.

### Phase 3 — FREEZE
Write the frozen spec to a file, commit it, record the commit SHA. **After this
point, changing the contrast, metric, capture site, threshold, estimator or
hyperparameter requires stopping the run and returning to PLAN.**

**Gate P3:** spec committed; SHA recorded in the run manifest.

### Phase 4 — EXECUTE (GPU)
Provision the box. Run the frozen protocol. **Write no analysis code during this
phase.** If you find yourself editing a `.py` file while the GPU bills, the
planning phase was not finished — record it as a gate failure.

### Phase 5 — ANALYSE (CPU)
All analysis runs on cached artifacts. Nothing in this phase needs a GPU. The
box can be terminated before this phase begins.

---

## 3. Incident log — what actually happened

Each entry: what broke, what it cost, and which rule now prevents it.

### I-1 · Three method designs, two discovered broken on the GPU
**Cost: ~4 hours of box time, plus 7 superseded artifacts.**

| # | design | how it failed | when discovered |
|---|---|---|---|
| 1 | generate an answer, parse which option was named | swapping the two names changed the model's pick ~half the time; person-consistency 48–68% vs a 50% coin-flip line | after 4 full base-rate runs |
| 2 | score option likelihood, options listed in prompt | scored text the prompt had just displayed; moving one option shifted its score 0.38 nats, and 0.38·√2 = 0.54 exactly matched the observed mean margin — the entire signal was list position | after full runs on 2 models |
| 3 | score with no option list | worked | — |

**Both failures were predictable without a GPU.** Design 1 fails a 20-item
order-swap check. Design 2 fails a first-principles rule (§5.1) *and* a 20-item
check. Neither needed a full run to detect.

→ **Rules: §2 Phase 2 (pilot), §5.1 (never score prompt-resident text), §5.2
(permutation invariance check).**

### I-2 · Ridge penalty left at a default; conclusion reversed
**Cost: ~2.5 hours (the entire `probe_qwen14b` run), and a nearly-published
wrong claim.**

The probe estimator was run at `alpha=1.0` against `d_model=5120` with n=600 —
effectively unregularised. It returned 0/10 reproducible directions and was
written up as "the probe is the worse estimator." Sweeping α over six orders of
magnitude later showed it gives **5/10**, beating the alternative, and
**recovered Nationality**, whose failure had been attributed to the model.

→ **Rule: §7.1 — a hyperparameter is never left at a library default. Either
sweep it in the pilot or derive it from a stated rule.**

### I-3 · Steering intervention confounded by direction magnitude
**Cost: ~2.5 hours (transfer tests on 2 models), 2 superseded artifacts.**

`apply_resid_pre_add` multiplies by a fixed coefficient. Direction norms varied
**5×** across categories (Frobenius 100→314) and correlated with the extraction
floor at **r=+0.91**. So the categories whose directions reproduce received a 5×
stronger dose — and those were exactly the categories whose sign-flip control
then failed, looking like generic damage.

**This was foreseeable from data already in the repo.** `docs/VERIFICATION_2026-08-07.md`
reports per-layer norm spreads of 600–1391× and was read on day one. The
connection was not made.

→ **Rules: §8.1 (normalise before comparing interventions), §8.2 (sweep the
dose), §5.6 (audit magnitudes before any cross-condition comparison).**

### I-4 · The dataset's own answer key was found late
**Cost: ~1 hour writing a heuristic that was then discarded.**

BBQ ships `supplemental/additional_metadata.csv` with a `target_loc` column
naming which option counts as biased. It is not in the per-category JSONL
release. A token-matching heuristic was written first, reached 80.4% coverage,
and was replaced by the authors' labels at 99.97%. The heuristic agreed with the
key on 99.64% of rows — competent, and entirely unnecessary.

→ **Rule: §5.7 — dataset provenance audit before writing any loader.**

### I-5 · Capture site never specified
**Cost: near-miss, no wasted compute, but the plan permitted a circular design.**

`notes/03` never said which token position residuals came from. The repo's
default path reads the **generated response** — and the bucket label is computed
*from* that response, so a direction built on it partly encodes the output
tokens. Caught by an outside critique, not by the plan.

→ **Rule: §4 requires `capture_site` as a mandatory field.**

### I-6 · A gate required buckets for a secondary comparison
**Cost: ~30 min, one false STOP.**

The base-rate gate demanded ≥40 items in `biased`, `other` **and** `unknown`.
The primary contrast only needs the first two. Qwen-1.8B never abstains, so
`unknown` was empty and the gate halted a run whose primary contrast was viable
in all ten categories. Relaxing it after seeing it fail is also exactly the shape
of goalpost-moving.

→ **Rule: §4 — gates bind only the primary analysis; secondary comparisons get
reported as unavailable, never block.**

### I-7 · Two silent logic bugs in analysis code
**Cost: ~1 hour, and both would have produced confident wrong tables.**

- `nonstereo` collided with `biased` on non-negative-polarity items, so the
  option-order swap swapped an option **with itself** on 17 of 40 sampled rows.
  Detected only because `person% + slot%` summed to 145%, which is impossible.
- `distinguishable()` inverts when the floor collapses: "cosine below the floor
  means distinct" is sound at a floor of 0.9 and nonsense at 0.05. The run
  printed `Age vs Nationality cos=-0.100 floor=0.057 DISTINCT` — asserting a
  difference between two directions neither of which reproduces.

→ **Rules: §9.1 (invariant assertions), §9.2 (a metric must declare its valid
domain).**

### I-8 · Verdict text asserted claims the data could not support
**Cost: low, risk high.**

`verdict()` reported *"NO STRUCTURE: bias topics are not separable"* when the
extraction floors had collapsed — conflating "we measured no difference" with
"we could not measure." A second instance printed *"clustering is within the
permutation null (p=1.000)"* when no clustering had run at all.

→ **Rule: §9.3 — every verdict string states its own preconditions.**

### I-9 · GPU memory not released; three queued steps starved
**Cost: ~35 min.**

The queue waited for the previous **process** to exit, but GPU memory release
lags process exit. The next step OOM'd during model load and then **hung holding
29 GB**, starving the two behind it.

→ **Rule: §9.4 — wait on the resource, not the process.**

### I-10 · Environment collisions discovered serially
**Cost: ~30 min at the start of the session.**

NumPy 2.x vs a NumPy-1.x-compiled torch; Pillow too old for `transformers`;
jinja2 too old for `apply_chat_template`. Each found by running and failing.

→ **Rule: §9.5 — environment smoke test as step zero.**

---

## 4. Pre-registration template — fill every field

Write this to `notes/PREREG-exp2.md`, commit it, record the SHA. **A blank field
is a blocked gate.**

```yaml
experiment_id:
one_sentence_question:          # what is being measured, in one sentence
falsifiable_prediction:         # what result would mean the hypothesis is WRONG

dataset:
  source:
  provenance_audit:             # §5.7 — what ships with it that we are not using?
  filters:                      # exact row filter
  n_items_per_condition:
  labelling_path:               # authors' labels, or ours; if ours, why

model_set:
  models:                       # exact HF ids + revision SHAs
  why_these:
  excluded_and_why:

measurement:
  what_is_scored:               # generation? likelihood? logits?
  prompt_construction:          # exact template, verbatim
  capture_site:                 # MANDATORY: which layer(s), which token position
  capture_timing:               # before or after generation, and why that is not circular

contrast:
  positive_pole:
  negative_pole:
  why_this_axis:                # what varies between the poles OTHER than the construct
  rejected_alternatives:        # and why

estimator:
  method:
  hyperparameters:              # every one, with a value and a justification
  hyperparameter_selection:     # swept in pilot / derived by rule / fixed by prior work
  known_variance_properties:    # what makes this estimator noisy, and when

thresholds:                     # every number that gates a decision
  usability_floor:
  significance:
  minimum_n:
  justification_each:           # must not depend on unseen data

controls:                       # §6 — all four are mandatory
  task_control:
  extraction_control:
  null_model:
  intervention_controls:

confounds_ruled_out:            # §5 checklist, each with the check that clears it

stopping_rules:
  proceed_if:
  stop_if:
  what_a_null_result_licenses:  # write this BEFORE running

artifacts:
  what_gets_written:
  what_gets_cached:

frozen_at_commit:
```

---

## 5. Confound checklist — check every item, record the check

Run against the **pilot**, before FREEZE. Each item names an incident it comes
from.

**5.1 · Prompt-resident text.** Never score, as a continuation, text that appears
in the prompt. Models copy recent context, and the copy signal will dominate.
*(I-1, design 2.)*

**5.2 · Permutation invariance.** If the prompt contains an ordered list of
options, re-run a 20-item sample with the order permuted. Correlate the per-item
metric across orderings. **Require r ≥ 0.5 and sign agreement ≥ 70%.** *(I-1.)*

**5.3 · Selection-induced imbalance.** If buckets are defined by model behaviour,
the buckets are *not* the dataset. Check the distribution of every prompt-level
nuisance variable (option position, name, length) **within each bucket**, not
just in the dataset. A balanced corpus does not give balanced buckets. *(session:
BBQ balances answer position to 33.4/33.5/33.4 and the buckets could still skew.)*

**5.4 · Circularity of the capture site.** If the bucket label is computed from
text X, do not extract from activations over text X. State explicitly why the
capture site cannot encode the label. *(I-5.)*

**5.5 · Estimator variance vs effect size.** For any difference-of-means
estimator, reproducibility rises with pole separation **by construction**. Before
claiming "categories with more signal are more separable," state how that claim
is distinguishable from the estimator's own variance property. *(session: H2 —
still unresolved.)*

**5.6 · Magnitude parity across conditions.** Before comparing any two
interventions, measure the norm of each. Report max/min. If it exceeds 1.2×,
normalise. *(I-3.)*

**5.7 · Dataset provenance.** Before writing a loader: open the dataset's own
repository, read its README, and list every file it ships. Ask specifically
whether the label you are about to reconstruct already exists. *(I-4.)*

**5.8 · Heterogeneity of the unit.** If a category pools sub-groups, record how
many and their distribution. Decide *before* running whether the unit of analysis
is the category or the sub-group. *(session: tested and refuted as an explanation
— r = −0.079 — but it should have been a planned dimension, not a post-hoc probe.)*

---

## 6. Controls — all four mandatory, all pre-registered

The single most valuable thing built in the last session was the **extraction
positive control**. It is what licensed reading any negative at all. Do not run
an experiment without all four.

| control | question | last session's result |
|---|---|---|
| **Task control** | can the model do the task when the answer is available? | 67–89% vs 33% chance |
| **Extraction control** | can this pipeline recover a direction we *know* exists? | 0.86–0.92 for topic identity |
| **Null model** | does shuffling the labels produce the same structure? | p = 0.030 / 0.005 |
| **Intervention controls** | norm-matched random direction, sign flip, dose sweep | see §8 |

**The extraction control is the one people skip.** Pick a contrast that must be
represented if anything is — topic, language, prompt length — and run it through
the *identical* capture site, estimator, and split-half procedure at matched n.
Without it, a null is uninterpretable: you cannot distinguish "nothing there"
from "our pipeline is broken."

**A control that has not been run is a control that failed.** Record the number.

---

## 7. Estimator rules

**7.1 · No library defaults.** Every hyperparameter is either swept during the
pilot or set by a stated rule. `alpha=1.0` cost a reversed conclusion. For ridge
on `(n, d)` residuals with n ≪ d, the sensible starting rule is α scaled to
`d_model` — sweep `1 → 1e6` on a pilot category and take the value that maximises
the extraction floor, then **fix it across all models** so it is not a per-model
degree of freedom.

**7.2 · Two estimators, not one.** Run both a difference-of-means contrast and a
regularised probe. If they disagree about which units are recoverable, that
disagreement is a finding about the estimator, not about the model — and you need
to know which you are looking at. Last session they disagreed: 4/10 vs 5/10.

**7.3 · Declare the estimator's variance property.** Write down, before running,
what makes it noisy. Difference-of-means: variance falls with pole separation and
with n. Ridge probe: overfits when under-penalised. This is what lets you read a
null correctly.

**7.4 · Report every hyperparameter that was tuned on the same data as the
result**, and say what it was tuned against. Last session α was tuned against
extraction floors, not against the p-value — but α is also what made clustering
possible at all, and that must be declared.

---

## 8. Intervention (steering) rules

**8.1 · Unit-normalise before any cross-condition comparison.** Put the dose in
the coefficient, never in the vector. Otherwise you compare interventions of
different strengths and call the difference specificity. *(I-3.)*

**8.2 · Sweep the dose.** A single coefficient is a single point on a curve you
have not seen. Last session's sweep (2/4/8/16) showed the sign-flip control
degrading monotonically — invisible at any one dose.

**8.3 · Three controls minimum:** covariance-matched random direction (not merely
norm-matched), sign flip, and a system-prompt baseline. Without a coherence check
on the generated text, "the model got worse at everything" is not excluded and no
result is causal.

**8.4 · Restrict to units that reproduce.** A transfer number for a direction
that does not reproduce against itself is not interpretable. It also cut the run
from ~35k forward passes to ~6k.

**8.5 · Never write "the direction."** Steering success does not identify the
representation (arXiv:2602.06801). Always "a direction."

---

## 9. Engineering rules

**9.1 · Assert invariants that must hold arithmetically.** The
`person% + slot% ≤ 100%` violation is what exposed I-7. Write assertions for
quantities whose relationships are fixed by construction, and check them in the
pilot.

**9.2 · A metric declares its valid domain.** `distinguishable(cos, floor)` is
meaningful only when `floor` is high. Encode that: return *indeterminate* outside
the valid domain rather than a number. *(I-7.)*

**9.3 · Verdict strings state their own preconditions.** Never emit "X is not
separable" when the precondition for measuring separability failed. Distinguish
`NO_EFFECT` (measured, absent) from `UNMEASURABLE` (could not measure). *(I-8.)*

**9.4 · Wait on the resource, not the process.** Poll `nvidia-smi` for free
memory before starting a GPU step, and kill stragglers after every step. *(I-9.)*

**9.5 · Environment smoke test is step zero.** Before any experiment code:
import the full stack, print versions, assert `torch.cuda.is_available()`, load
the smallest model, run one forward pass. Pin `numpy<2` on Lambda Stack. *(I-10.)*

**9.6 · Cache the expensive stage.** Per-item scores do not depend on the
estimator. Cache them keyed by `(model, unit, limit, seed)` and validate against
item ids before reuse. Added late last session; it turned a 2-hour re-run into
minutes.

**9.7 · Write results incrementally and flush.** Buffered stdout made a 2-hour
run look hung. Flush progress lines, and write per-unit results as they complete.

**9.8 · Never transfer source files via shell string replacement.** It corrupted
UTF-8 in one file. Use the editor; generate patches with `git diff --output=` so
the tooling controls encoding and line endings.

---

## 10. What went right — preserve these

Do not "improve" these away.

1. **Acceptance criteria written before running.** C1/C2 for the margin design
   were fixed in a docstring, committed, and printed at the top of the run. C2
   passed and C1 failed — and C1 is what caught the position confound. Without
   the pre-commitment, C2 alone looked like a green light.
2. **The extraction positive control.** Highest-value single artifact of the
   session.
3. **The extraction floor itself.** Reporting cosines against a measured noise
   floor rather than against zero.
4. **Refusing to guess.** Unlabelable rows were counted as `unresolved` and
   excluded, never folded into a behaviour class.
5. **Floors reported with their n**, and a warning when n varies across units.
6. **Superseded artifacts labelled, not deleted.** `notes/09` names every dead
   file and why.
7. **Distinguishing "we measured no difference" from "we could not measure."**
8. **Stopping to check when arithmetic looked impossible** rather than reasoning
   around it.

---

## 11. Checklist to hand the next agent

```
[ ] PREREG-exp2.md written, every field filled          §4
[ ] Every threshold justified without unseen data       §4
[ ] Confound checklist run against pilot, recorded      §5
[ ] All four controls specified and passing at pilot    §6
[ ] Hyperparameters swept or derived, never defaulted   §7.1
[ ] Estimator variance property written down            §7.3
[ ] Interventions unit-normalised, dose swept           §8.1, §8.2
[ ] Invariant assertions in place                       §9.1
[ ] Verdict strings state preconditions                 §9.3
[ ] Environment smoke test green                        §9.5
[ ] Expensive stage cached                              §9.6
[ ] Spec committed, SHA recorded                        Gate P3
[ ] ---- ONLY NOW PROVISION THE GPU ----
[ ] No .py file edited while the GPU runs               Phase 4
```

**If any box is unchecked, the answer to "can we start the run?" is no.**
