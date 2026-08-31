# Run plan — Experiment 1, run 2

> ### ⚠ AMENDED 2026-08-23 — read `17-reference-paper-and-contrast.md` first
>
> The contrast changed after this file was written. See §0 immediately below for
> what that does to the plan. **Everything in §1 (caching), §2 (report schema),
> §3 (queue runner) and §4 (pre-rental checklist) stands unchanged** — those are
> engineering requirements and are independent of which contrast we run.
>
> §5 (steps and budget) is **re-derived in §0.3** and the figures in §5 are stale.

---

## 0. What the reference paper changed

### 0.1 The passes per model

The five passes are no longer margins / generation / residual capture / task
control / topic control. They are now:

| # | pass | purpose | needed for |
|---|---|---|---|
| 1 | **residual capture, both arms** | ambiguous and disambiguated items, matched on `question_index`, at chat token −2 | the primary contrast |
| 2 | **task control** | disambiguated items, accuracy vs 33% chance | validity of the instrument |
| 3 | **generation** | verbatim responses, stored character for character | behavioural scoring + the steering stage |
| 4 | **steering sweep** | α ∈ {5,10,20,30,60}, unit-normalised directions | the "one shared knob" test |
| 5 | **margins** | log-probs per option | the **third analysis** only |

Passes 1 and 2 are the critical path. Pass 5 is now secondary — it exists to
demonstrate that the behaviour-derived contrast fails.

### 0.2 The model list — UNCHANGED from run 1

> **DECIDED 2026-08-23 by Jeremiah: the SAE stage is OUT OF SCOPE.**
> `google/gemma-2-9b-it` was proposed here only because it is the one model with
> public GemmaScope SAEs. With the SAE stage cut, **it is not added.**

**The model list is the five run-1 models, unchanged:** qwen-14b, qwen-7b,
qwen-1.8b, yi-6b, gemma-2b. All ungated except gemma-2b, whose access was granted
2026-08-20. The cross-family activation-space result is genuinely ours — the
reference paper uses only two models.

### 0.3 GPU budget is NOT a constraint — decided 2026-08-23 by Jeremiah

**Compute is unlimited for this project.** Do not optimise for dollars, do not
cut a model, a control, a sensitivity analysis or the pilot to save money, and do
not present cost as a reason for any design choice. Every dollar figure in §5 and
elsewhere in this file is retained for the record only and must not drive a
decision.

**Two things are still finite and still govern:**

1. **Wall-clock time.** The 12-hour kill criterion in §6 stands. It exists to force
   a re-plan when something is wrong, not to save money — a run that has gone 12
   hours has a problem that more hours will not fix.
2. **Attention.** Every extra model or arm is another set of artifacts to verify
   and another way for a defect to hide. Scope discipline is still required; it is
   just no longer a budget argument.

Time estimates remain useful for *sequencing* — knowing qwen-14b is 40% of the
GPU time is why it runs first, so a failure surfaces early. Re-derive timings from
measured throughput at bring-up, and ignore the cost column.

### 0.3.1 The old note on re-deriving the budget (superseded, kept)

The §5 table below is anchored on run-1 throughput for a 14B model doing scoring
passes. Two things now break that anchor:

1. ~~gemma-2-9b-it is new~~ — no longer applies; the model list is unchanged.
2. **The steering sweep is generation, not scoring**, at five α values across
   every category and every direction. Generation is slower per token than
   scoring, and run 1's own budget review flagged that a 2× error here is
   plausible.

**Do not reuse the $11 figure.** Re-derive from measured throughput at bring-up
(step 0), before committing to the full queue. The kill criterion at 12 h
wall-clock in §6 is unchanged and is the real guard.

### 0.4 Two new stages

**STEERING** — mandatory. Follows the reference paper §2.6–2.7:
```
induction:  x' = x + α·r        r unit-normalised, α ∈ {5,10,20,30,60}
ablation:   x' = x − (x·r)·r
```
Build a **controlled test set balanced across item type × the unsteered model's
response**, so the unsteered baseline is fixed at 50% by construction and any
movement is attributable to the intervention. Their four cells are HR/HC/BR/BC;
ours are the bias analogue — stereotyped/not × ambiguous/disambiguated.

~~**SAE**~~ — **VOID. Cut by Jeremiah 2026-08-23, before any planning cost was
sunk into it.** Not in scope for run 2. The paper's claim is descriptive (bias
directions are distinct and steer interchangeably), not mechanistic. Do not
re-open this without an explicit instruction from Jeremiah.

---

Companion to `notes/13-preregistration.md`, which fixes the science. This fixes
the execution. Planning only; no GPU used to write it.

**The governing constraint, stated first:** *every analysis in this plan must be
redoable tomorrow with the GPU already returned.* If any step would require
re-renting, the plan is wrong. Run 1 failed this test and it converted every
statistical defect into a blocked one.

---

## 1. Artifact caching — the hard requirement

### 1.1 What is persisted, and why residuals are the whole point

Run 1 cached margins and discarded residuals (S5), so fixing anything meant
renting again. Run 2 persists the expensive object.

| artifact | format | when |
|---|---|---|
| **residuals** — `(n_items, n_layers, d_model)` float32 | `.npy` + sidecar `.json` with item ids, layer count, dtype, capture site | at capture, before any extraction |
| **prompts** — the exact string scored, verbatim | `.jsonl`, one row per item | at scoring |
| **responses** — generated text, character for character | `.jsonl`, same row keys | dedicated generation pass |
| **per-option log-probs** — all three, not just the margin | `.jsonl` | at scoring |
| margins + abstention margins | `.json` (as run 1) | at scoring |
| directions, floors, matrices, nulls | `.npy` / `.json` | analysis (CPU) |

### 1.2 Sizing — computed, not guessed

n = 600, 9 categories, float32:

| model | layers | d_model | fp32 | fp16 |
|---|---|---|---|---|
| qwen-1.8b | 24 | 2048 | 1.06 GB | 0.53 GB |
| gemma-2b | 18 | 2048 | 0.80 GB | 0.40 GB |
| yi-6b | 32 | 4096 | 2.83 GB | 1.42 GB |
| qwen-7b | 32 | 4096 | 2.83 GB | 1.42 GB |
| qwen-14b | 40 | 5120 | 4.42 GB | 2.21 GB |
| **subtotal** | | | **11.9 GB** | 6.0 GB |
| + topic-control residuals | | | +4.3 GB | +2.1 GB |
| **TOTAL** | | | **≈16 GB** | ≈8 GB |

**Decision: float32, ≈16 GB.** fp16 would halve it, but 16 GB is trivial to store
and transfer (≈15 min at 20 MB/s), and precision in the stored object is the one
thing that cannot be recovered later. This is not where to economise.

**The A100 instance's local SSD is 0.5 TiB** — 16 GB is 3% of it.

### 1.3 Continuous sync — nothing lives only on the box

- A sync daemon runs `rsync`/`scp` **every 10 minutes** from the box to the
  laptop for the whole session, not once at the end.
- Every step writes its outputs and then **immediately** triggers a sync of that
  step's directory before the queue advances.
- The post-run verifier (§3) runs against the **laptop copy**, not the box.
- **Termination gate:** the box is not terminated until the verifier passes on the
  laptop copy and a byte-count comparison against the box shows zero missing
  files. Run 1 recovered its artifacts by manual checking and luck.

---

## 2. Report schema

Every run writes `report.json` containing, at minimum:

```
model, hf_id, revision_sha, n_layers, d_model
method                      "extremes" | "probe"
estimator_params            { C, alpha_rule, quintile, layer_summary }
sampling                    { n, seed, n_splits, split_seeds, permutations }
capture                     { site, layers, dtype, residual_path }
abstention                  { eligible, ineligible, fraction } per category
dropped_categories          { cat: {stage, reason, accuracy, z} }
denominator                 { requested, scored, eligible }
code_version                git SHA
prereg_sha                  SHA of notes/13-preregistration.md
per category:
  positive_control          { n, accuracy, z, passes }
  observed_floor            { mean, ci_lo, ci_hi, n_splits, cosines[] }
  negative_control_floor    { mean, ci_lo, ci_hi, cosines[] }
  reproduces                true | false | "INDETERMINATE"
```

This closes the Q2–Q6 class of run-1 failures: the report records the choices that
produced it, so no number is recoverable only from a folder name.

---

## 3. A queue runner that cannot hide a failure

Run 1's `run()` returned 0 unconditionally and blanket-`pkill`ed after every step,
so a killed step and a successful one were indistinguishable.

**Replacement, with four required properties:**

1. **Real exit codes.** The runner records the true exit status per step and
   writes it to a machine-readable `queue_manifest.json`, not just a log line.
2. **No blanket pkill.** Each step's PID is tracked and only that PID is signalled.
   GPU-free is verified by polling `nvidia-smi` before the next step starts —
   which run 1 eventually adopted and which fixed the OOM cascade.
3. **Declared expected outputs.** Every step declares the files it must produce.
   The runner checks them immediately on completion; a missing or unparseable
   output fails that step loudly and marks it `INCOMPLETE`.
4. **Post-run verifier.** After the queue, a verifier walks the manifest and
   asserts: every step exited 0, every declared output exists on the **laptop**,
   every `.json` parses, every `.npy` has a valid header and the expected shape,
   and every residual file's item ids match its margins file. It exits non-zero
   and prints a diff if anything fails.

**The verifier's report is the termination gate.**

---

## 4. Pre-rental checklist — before any money is spent

Run 1 queued a model whose weights the account could not access and found out on
a paid box.

```
[ ] HF token valid; `model_info()` returns for EVERY model in the list
[ ] BBQ ambig/disambig pairing verified: every item used has a partner at the
    same question_index, and per-arm counts are balanced per category
[ ] llama3-8b: access confirmed, or the model is dropped from the plan now
[ ] prereg committed; SHA recorded
[ ] pilot green: full pipeline on 20 items x 2 categories on CPU or a 0.5B model,
    exercising every code path including residual persistence and the verifier
[ ] 16 GB free on the laptop for the sync target
[ ] one session only — no second agent on this repo (run 1 nearly collided)
```

**Any unchecked box means do not rent.**

---

## 5. Ordered steps, with time and cost

Anchored on run 1's **measured** throughput: qwen-14b did 18,000 forwards in
~75 min = 4.0 ops/sec. Ops per model = margins 16,200 + generation 5,400 +
residual capture 5,400 + task control 4,050 + topic control 1,920 = **32,970**.

| # | step | model | ops/sec | hours | $ @1.99 | critical path |
|---|---|---|---|---|---|---|
| 0 | bring-up, env pins, verify all model loads | — | — | 0.4 | 0.80 | ✅ |
| 1 | qwen-14b: all five passes | qwen-14b | 4.0 | 2.29 | 4.56 | ✅ |
| 2 | qwen-7b | qwen-7b | 8.0 | 1.14 | 2.28 | ✅ |
| 3 | yi-6b | yi-6b | 9.3 | 0.98 | 1.95 | |
| 4 | gemma-2b | gemma-2b | 28.0 | 0.33 | 0.65 | |
| 5 | qwen-1.8b | qwen-1.8b | 31.1 | 0.29 | 0.59 | |
| 6 | final sync + verifier + byte comparison | — | — | 0.3 | 0.60 | ✅ |
| | **total** | | | **5.7 h** | **$11** | |
| | **with 50% contingency** | | | **8.6 h** | **$17** | |

**Critical path** is qwen-14b: it is 40% of the GPU time and the only model that
produced ≥3 reproducible categories in run 1, so it is the only one that can
support the clustering secondary. Run it **first**, so a failure surfaces while
there is still budget to re-plan.

**Everything after step 6 is CPU on the laptop and costs nothing:**

| analysis | extractions | est. |
|---|---|---|
| observed + negative-control floors, extremes, 400 splits | 36,000 | ~1.0 h |
| permutation null, 1,000 perms | 25,000 | ~0.7 h |
| probe secondary + matched-n control | — | ~2 h |
| **total CPU** | | **~4 h, $0** |

---

## 6. Kill criteria — stop and re-plan, do not keep spending

| at step | condition | action |
|---|---|---|
| 0 | any model fails to load, or HF returns 403 | **stop.** Drop that model from the plan in writing before continuing |
| 0 | residual persistence not verified working | **stop.** This is the one requirement that cannot be retrofitted |
| 1 | qwen-14b task control fails on >3 of 9 categories | **stop and re-plan.** The task is not being measured |
| 1 | qwen-14b extraction control (topic identity) fails to beat its negative control | **stop.** The pipeline cannot recover a direction we know exists; nothing downstream is interpretable |
| 1 | **zero** categories beat their negative control on qwen-14b | **continue but downgrade.** This is H1's falsification, and it is a result — finish the cheap models to establish it across families, then stop |
| any | verifier reports a missing or unparseable output | **stop that step**, fix, re-run that step only |
| any | wall-clock exceeds 12 h | **stop.** Re-plan rather than extend |

---
---

# STEP 5 — Hostile review of this plan

## 5.1 Defect-by-defect closure

| id | defect | status | where |
|---|---|---|---|
| **S1** | decision statistic from 10 draws, no interval | **CLOSED** | §13-prereg §4: n_splits=400, calculated to ±0.020; statistic is the mean with a bootstrap CI; decision is CI-based (§3.1) |
| **S2** | estimator comparison confounded with n | **CLOSED** | §13-prereg §7: matched-n estimator control — probe fit on only the 2×120 pole items |
| **S3** | alpha selects a direction, chosen on the gated quantity | **MITIGATED** | §13-prereg §2.2: `alpha = C·trace(XXᵀ)/n`, `C=1.0` fixed, reads only the scale of X. **Not closed** — the probe's output still depends on `C`; declared in §12.2 |
| **S4** | threshold never calibrated per estimator | **CLOSED** | §13-prereg §3.1: no threshold constant exists; each category is judged against its own negative control, computed per estimator by construction |
| **S5** | residuals never cached | **CLOSED** | §1 above; 16 GB, float32, verified before termination |
| **N1** | layer summary unjustified, estimator-dependent | **CLOSED** | §13-prereg §8: norm-weighted mean fixed in advance, median as pre-declared sensitivity; the rule also cancels to first order between observed and control arms |
| **N2** | clustering statistic has no error estimate | **CLOSED** | §13-prereg §10: `cluster_strength` reported with a bootstrap CI over splits |
| **N3** | 23.5% of items are abstentions treated as choices | **CLOSED** | §13-prereg §6: eligibility rule, counts reported, sensitivity pre-declared |
| **N4** | null tests against noise, not a matched alternative | **CLOSED** | §13-prereg §10: within-category margin shuffle holds topic, n and format fixed |
| **N5** | split-half not stratified by pole | **CLOSED** | §13-prereg §9 |
| **M1** | floor confounded with behavioural tilt | **CLOSED 2026-08-23** — the primary contrast is labelled by `context_condition`, a dataset annotation, so behavioural tilt is never consulted. Was "accepted as limitation"; see `17-reference-paper-and-contrast.md` §4.1 | §13-prereg §12.1 |
| **M2** | two positives rest on heavy tails | **MITIGATED** | §13-prereg §12.3: winsorised and unwinsorised both reported; Disability_status carries the headline |
| **M3** | n varies with floor | **CLOSED** | §13-prereg §5: n matched at 600 across 9 categories |

**Limitations paragraph — ⚠ STALE as of 2026-08-23.** All three limitations below
are properties of the margin-quintile contrast, which is now the *third analysis*
rather than the primary. Do not paste this into the paper as written.

The revised limitations for the primary contrast are:

> *Two limitations bound these results. First, the ambiguous and disambiguated
> arms differ slightly in context length, so we report a cross-category control
> establishing that the recovered directions are category-specific rather than
> encoding context specificity. Second, all conclusions are BBQ conclusions and no
> claim generalises beyond that dataset.*

The original paragraph is retained below, and remains the correct limitations
statement **for the third analysis only**:

> Three limitations bound these results. First, our procedure detects a bias
> direction only where the model exhibits a systematic behavioural tilt on that
> category; categories without such a tilt are untestable by this method rather
> than shown to lack a representation. Second, two of our positive results
> (Physical_appearance, Age) rest on heavy-tailed margin distributions in which
> the top 5% of items carry roughly half the variance, and the extremes contrast
> selects exactly those items; we report winsorised and unwinsorised estimates
> and note that Disability_status, with excess kurtosis −0.9, is the most robust
> positive. Third, all secondary probe results depend on a regularisation
> constant fixed by rule rather than derived from theory; the primary analysis
> uses an estimator with no such parameter.

## 5.2 Independent re-derivation of the budget

Derived a second way, from total operations rather than per-model timing:
5 models × 32,970 ops = **164,850 operations**. Weighting throughput by parameter
count against the run-1 anchor (14B at 4.0 ops/sec) gives per-model times of
2.29 / 1.14 / 0.98 / 0.33 / 0.29 h = **5.04 h**, matching the table to two
decimals. With bring-up and verification, **5.7 h ≈ $11**; with 50% contingency,
**8.6 h ≈ $17**.

**Where this could be wrong:** the run-1 anchor came from *scoring* passes.
Generation is slower per token and I have modelled it at the same rate. If
generation runs 2× slower than scoring, the total rises to ≈7.4 h ≈ $15
(≈$22 with contingency). **Still comfortably inside any sane ceiling**, so the
estimate is robust to being wrong by a factor of two — which is the only claim
worth making about it.

## 5.3 The single weakest point in this plan

**The `C = 1.0` in the probe's alpha rule.**

Everything else is either closed by a calculation or declared as a limitation.
`C` is a number I chose because it is the natural scale-free default, and I have
**no evidence it is the right scale for this data** — run 1's optima were at
α = 1e4 for a d=2048 model and 1e6 for a d=5120 one, which `trace(XXᵀ)/n`
normalisation *should* absorb, but that has not been checked because checking
requires residuals that do not exist.

**What it would take to fix:** run the extraction control (topic identity)
through the probe at `C ∈ {0.1, 1, 10}` on cached residuals — CPU, ~20 minutes,
zero dollars — and adopt whichever `C` makes the *control* reproduce best. That
selects on a control we already know the answer to, never on the categories being
tested, so it is legitimate. **This should be added as an explicit step and it is
the one change I would make to this plan.** It cannot be done before the run
because it needs residuals, so it must be pre-declared now and executed on
cached data.

*Recommendation: add "calibrate `C` against the extraction control on cached
residuals, before any category floor is computed" to §13-prereg §2.2. Declared
here so it is not a post-hoc choice.*

## 5.4 What would make run 2 a failure

Not "H1 is falsified" — a clean negative with passing controls is a result.

Run 2 has **failed** if any of these is true when it ends:

1. **Any analysis in §13-prereg cannot be run because an artifact was not saved.**
   This is the run-1 failure and it is the only unforgivable one.
2. **The extraction control does not beat its negative control** on the primary
   model — the pipeline cannot recover a direction known to exist, so no result
   is interpretable.
3. **A parameter not listed in §13-prereg §14 turns out to matter**, meaning a
   degree of freedom was hidden rather than declared.
4. **A reported number cannot be traced** to a stored artifact and a recorded
   parameter set.
5. The re-run produces different conclusions from run 1 **and we cannot say
   which defect caused the difference.**

Note that (5) is not "the answer changed." The answer *should* be allowed to
change — Nationality has a 26.6% chance of flipping on redraw and that is the
point. The failure is being unable to attribute the change.

---
---

# 6. Look ahead

## 6.1 The strongest claim if everything goes right

> Across five open-weight models spanning three architecture families and 2B–14B
> parameters, bias directions extracted from BBQ stereotype-margin extremes
> reproduce against a topic-matched, n-matched negative control for a specific
> and consistent subset of categories — disability and physical appearance — while
> no race-related category does in any model; among the categories that do
> reproduce, the directions are pairwise distinguishable and cluster above a
> within-category permutation null.

That sentence is defensible only if the extraction control passes per estimator
and the negative controls are reported alongside every positive.

## 6.2 The likely claim — probe fails calibration, only `extremes` stands

This is the **expected** outcome, and it is still publishable:

> Using a mean-difference estimator with no free hyperparameter, we measure how
> well a bias direction reproduces against a topic- and n-matched negative
> control, and find that reproducibility is confined to two of ten BBQ categories
> and replicates across three model families — while every race-related category
> fails everywhere. We report this as a limit on what current extraction methods
> recover, not as evidence that those categories lack a representation.

**The methodological contribution stands on its own here.** The extraction floor
plus a within-category negative control is a reusable instrument for deciding
whether *any* claimed steering direction means anything, and the field currently
reports cosines against zero.

## 6.3 The three attacks a reviewer makes first

| # | attack | the artifact that answers it |
|---|---|---|
| 1 | **"Your negative result just means your method is too weak."** | The **extraction control**: a topic-identity direction through the identical pipeline, estimator and n, reproducing at 0.86–0.92 in run 1. The instrument recovers a direction that exists. Reported per estimator (closes S4). |
| 2 | **"You picked the threshold to get this answer."** | There is no threshold. §13-prereg §3.1 replaces the constant with a per-category negative control, and the pre-registration is committed with a SHA recorded in every run manifest. Run 1's 0.50 is retired precisely because this attack lands on it. |
| 3 | **"Reproducibility just tracks how biased the model behaves — you've measured effect size, not representation."** | **Partially answered, and this is the one that hurts.** The within-category negative control holds tilt fixed and shuffles only labels, and the run-1 dose-response was monotone at fixed n (0.795 / 0.480 / 0.065 on Age). But M1 is real at +0.660 to +0.769 and is **declared as a limitation**, not rebutted. |

Attack 3 is the one to prepare for. Do not pretend it is closed.

## 6.4 NeurIPS-quality vs workshop note

**What would make it NeurIPS-quality:**

1. The negative-control methodology stated as a general instrument, not a
   project detail — with the demonstration that published steering results would
   change if it were applied.
2. Replication across ≥3 architecture families with a **pre-registered**
   protocol. Run 1 has the replication; run 2 adds the pre-registration.
3. A causal arm that survives its controls — covariance-matched random direction,
   coherence check, system-prompt baseline. Run 1 has **none** of the three.
4. Either a positive structural result, or a negative with power analysis showing
   what effect size the design could have detected.

**Is that reachable?** In compute, easily — the whole run is ~6 GPU-hours and
~$11–17. **The binding constraint is the causal arm.** Items 1, 2 and 4 are
achievable with the plan as written. Item 3 is a separate experiment of similar
size that does not exist yet in any plan.

**Honest read:** with run 2 as specified, this is a **strong workshop paper and a
borderline conference paper**. The methodological contribution is real and the
replication is solid, but a reviewer will observe that the headline is a negative
result about two of ten categories with no causal validation. Adding the causal
arm — and having it survive its controls — is what moves it across.

**Recommendation:** run 2 exactly as specified, then decide on the causal arm
using its results. Do not attempt both in one session; that is the mistake run 1
made in a different form.
