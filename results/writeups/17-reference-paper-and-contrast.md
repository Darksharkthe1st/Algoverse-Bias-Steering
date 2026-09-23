# 17 — The reference paper, and the contrast we are replicating

**Written 2026-08-23. This document supersedes parts of `13-preregistration.md`
and `14-run-plan.md`. Read it before either of them.**

Jeremiah supplied the paper this project is modelled on. Reading its methods
changed the design, not just the parameters. This document records what the paper
does, what we were doing differently, and exactly what we are changing as a
result — so that a new session can pick this up with no ambiguity.

**Governing instruction from Jeremiah, recorded verbatim in spirit:** *mimic their
successful method rather than invent something new. Follow the formula, add our
own twist (bias). Do not overcomplicate.*

---

## 1. The paper

> **There Is More to Refusal in Large Language Models than a Single Direction**
> Faaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani, Husrev Taha Sencar
> Qatar Computing Research Institute, HBKU. 21 pages.
> Local copy: `Downloads/refusal more directions.pdf`
> Extracted text: scratchpad `refusal_paper.txt`

**Their claim.** Prior work says refusal is mediated by a single activation-space
direction. They show that across 11 categories of refusal and non-compliance, the
behaviours correspond to *geometrically distinct* directions — yet steering along
any of them produces nearly identical refusal/over-refusal trade-offs. The
directions differ in **how** the model refuses, not **whether**. They then use
sparse autoencoders to explain this: a small reusable core of shared refusal
latents plus a long tail of style- and domain-specific latents.

**Our twist.** The same study, on bias instead of refusal, using BBQ categories in
place of refusal splits.

---

## 2. What they actually do — the method, in order

### 2.1 The contrast (the part that matters most)

> "A refusal direction is computed by contrasting residual stream activations
> elicited by prompts **labeled to require non-compliance** with those elicited by
> **benign prompts**. For each prompt set, we collect residual stream activations
> at a fixed token position, average them across prompts, and define the refusal
> direction as the difference between these activation vectors."

**The labels come from dataset annotations. The model's own behaviour is never
used to build the contrast.** This is the single most important sentence in the
paper for us, and §4 below explains why.

### 2.2 Sample size

> "Each evaluation split consists of a balanced pair of prompt sets: at least **32
> prompts** labeled to elicit non-compliant behavior, paired with an equal number
> of benign prompts. Following Arditi et al., we adopt 32 prompts per class as a
> standard unit for estimating stable activation centroids."

**32 per class.** Not 600. See §4.3.

### 2.3 Capture site

- Residual stream, hook `resid_pre`, at a **fixed mid-layer** — layer 20 for
  gemma-2-9b-it, layers 15–16 for Llama-3.1-8B-Instruct.
- Token position: **the chat-template token immediately preceding the assistant's
  response (index −2)**, which they call "the model's decision state."

Note they use *one* layer, not all layers. Our all-layer capture is a superset and
is strictly more informative; we keep it and report their single-layer choice as
the primary read for comparability.

### 2.4 Within-category stability — they have an extraction floor

Appendix C, Table 8, "Within-category HR–BC direction similarity." For each
category with multiple independent 32/32 splits, they report pairwise cosine
between directions estimated from **that category alone**:

| category | mean | sd |
|---|---|---|
| SorryBench (crimes/torts) | 0.9900 | 0.0030 |
| SorryBench (hate speech) | 0.9892 | 0.0032 |
| CoCoNot (Humanizing) | 0.9887 | 0.0035 |
| SorryBench (inappropriate topics) | 0.9884 | 0.0034 |
| XSTest (all) | 0.9859 | 0.0049 |
| WildGuard-Mix (all) | 0.9834 | 0.0057 |
| SorryBench (all) | 0.9834 | 0.0053 |
| CoCoNot (Indeterminate) | 0.9815 | 0.0061 |
| SorryBench (unqualified advice) | 0.9770 | 0.0077 |
| CoCoNot (safety concerns) | 0.9757 | — |
| CoCoNot (Incomplete) | 0.9749 | 0.0073 |
| CoCoNot (all) | 0.9559 | 0.0142 |

Their summary: "All categories exhibit extremely high internal alignment
(typically ≥0.95), indicating that small 32/32 training sets suffice to recover a
stable category-level refusal direction."

### 2.5 Cross-category similarity

Table 1 is an 11×11 cosine matrix. Typical values **0.4–0.6**, several
near-orthogonal (Incomplete-CCN vs OverRefusal-XST = **−0.062**), some very high
(HateSpeech-SB vs CrimeAssistance-SB = **0.917**).

**The logic of their claim is exactly our extraction-floor logic:** within-category
≥0.95 establishes the noise floor; cross-category 0.4–0.6 sits far below it;
therefore the directions are genuinely distinct.

### 2.6 Steering and ablation

```
induction:  x'_t  ←  x_t + α·r        (r is the unit-normalised direction)
ablation:   x'_t  ←  x_t − (x_t · r)·r
```

α chosen by grid search over `{5, 10, 20, 30, 60}` on a held-out validation pool
— the smallest value achieving ≥90% refusal on harmful prompts while keeping
benign over-refusals below a threshold. They report α = 60 for gemma-2-9b-it.

### 2.7 The controlled test set

200 prompts, balanced across prompt type (harmful vs benign) **and** the unsteered
model's response (refusal vs compliance). Four equal subsets:

| subset | meaning | correct? |
|---|---|---|
| HR | harmful → refused | yes |
| HC | harmful → complied | no (jailbreak) |
| BR | benign → refused | no (over-refusal) |
| BC | benign → complied | yes |

By construction the unsteered model scores exactly 50%. This is an elegant design
— the baseline is fixed at 50% by the sampling, so any movement is attributable to
the intervention.

### 2.8 The SAE analysis

- Pretrained JumpReLU residual-stream SAEs from **GemmaScope**, layers 9 / 20 / 31,
  on gemma-2-9b-it. Also `saes-llama-3.1-8b-instruct` for Llama.
- Firing-rate separation, following Ferrando et al. (2024):
  ```
  a_ij = 1[z_ij > 0]                       binary firing indicator
  f_j(y) = mean of a_ij over examples with label y
  Δ_j = f_j(refusal) − f_j(non-refusal)    separation score
  ```
- Rank latents by Δ, take top-K with Δ > 0 → the "refusal latents" for that split.
- SAE-based direction = **mean of the decoder directions** of those latents.
- Steering: `x' = x + β·d_SAE`.
- Ablation: encode, zero the selected latents, decode back.
- **Controls:** random latent subsets of the same size K, and random unit vectors
  in residual space.

### 2.9 Models

`google/gemma-2-9b-it` and `meta-llama/Llama-3.1-8B-Instruct`. Two models only.

---

## 3. Their datasets — and our substitution

| theirs | what it provides | our equivalent |
|---|---|---|
| WildGuardMix | ground-truth comply/decline annotations | BBQ `context_condition` |
| SorryBench | 4 policy domains | BBQ categories |
| CoCoNot | 5 non-compliance types | BBQ categories |
| XSTest | over-refusal (benign but adversarially framed) | BBQ disambiguated items |

They build **11 splits**. We have **10 BBQ categories**. Comparable granularity.

---

## 4. The three findings that change our design

### 4.1 Their contrast is externally labelled — M1 is closable after all

Run 1 built directions by ranking items by **the model's own stereotype margin**
and contrasting the top and bottom quintiles. That is what created defect **M1**
(floor confounded with behavioural tilt, +0.660 to +0.769 across five models),
which `12-retrospective.md` and `13-preregistration.md` both record as **not
closable by this design**.

It is not closable *by that design*. The reference paper never uses that design.
Their labels are dataset annotations, so a category with no behavioural tilt can
still yield a direction.

**M1 is therefore reclassified from "declared limitation" to "closed by adopting
the reference paper's contrast."** This is the single biggest change in this
document.

### 4.2 The extraction floor is not our novel contribution

Appendix C Table 8 *is* an extraction floor, computed the same way we compute
ours: independent splits within one category, pairwise cosine between the
resulting directions.

Earlier project documents — including `08-results`, the published "Extraction
Floor" artifact, and the protocol document written 2026-08-23 — claim or imply
that this validity check is absent from the literature. **That claim is false with
respect to the paper we are modelling, and must be removed everywhere it appears.**

What remains genuinely ours:
- Applying it to **bias** rather than refusal.
- **Five models across three families**, versus their two.
- The demonstration in §4.3 that a behaviour-derived contrast fails where an
  annotation-derived contrast succeeds — which is a real methods contribution and
  is now a *deliberate* experiment rather than an accident.

### 4.3 The diagnostic number: 32 beats 600

| | reference paper | our run 1 |
|---|---|---|
| items per class | **32** | 240–600 |
| contrast | annotation-derived | model-behaviour-derived |
| within-category floor | **0.95–0.99** | −0.204 to 0.880, mostly < 0.50 |

They achieve near-perfect reproducibility with roughly **one twentieth** of our
data. Sample size cannot explain the gap. **The contrast can, and it is the only
thing left that differs.**

Run 1's conclusion was never "bias has no reproducible directions." It was "this
particular contrast does not produce them."

---

## 5. The contrast decision

BBQ ships two model-independent labels on every item. Verified directly in
`datasets/BBQ_Prompt_Sets/Age.jsonl` — both are evenly balanced (200/200 in the
first 400 rows):

```
question_polarity   neg | nonneg
context_condition   ambig | disambig
```

Items also carry `question_index`, which lets us match the two arms on the
underlying scenario.

### 5.1 PRIMARY contrast — ambiguous vs disambiguated, matched on question_index

```
direction_C = mean(resid | category C, ambig)
            − mean(resid | category C, disambig)
```

**Why this is the right analogue.** Joad et al. contrast prompts where the target
behaviour *should* occur against prompts where it *should not*. In BBQ:

- **ambiguous** — the context does not identify anyone, so there is no correct
  named answer. Any named choice is stereotyping. This is the bias-eliciting arm.
- **disambiguated** — the context does identify the person, so a named answer is
  simply correct. Stereotyping is not what is being expressed. This is the benign
  arm.

**Why it is well matched.** For a given `question_index`, the ambiguous and
disambiguated versions share the same question text and the same three answer
options. They differ only in whether the context resolves the referent.

**The confound to control.** The disambiguated context contains an extra
resolving clause, so the two arms differ slightly in length and specificity. The
direction could partly encode "context specificity" rather than bias. See §5.3.

### 5.2 SECONDARY contrast — negative vs non-negative question polarity

```
direction_C = mean(resid | category C, neg polarity)
            − mean(resid | category C, nonneg polarity)
```

Same context, same answer options; only the question's valence changes ("Who was
not comfortable using the phone?" vs the positive framing). This is even more
tightly matched than the primary, but it measures a **valence** axis rather than a
bias axis, since both polarities can elicit stereotyping in opposite directions.

Its job is to be a comparison, not the headline.

### 5.3 The specificity control — mandatory

Because §5.1's two arms differ in context length, we must show the direction is
not merely "long context vs short context."

**Control:** build the identical ambig-vs-disambig contrast on items from a
*different* category, and measure cross-category cosine. If the direction is
really encoding context specificity, every category's direction will be nearly
identical (cosine → 1.0) because specificity is category-independent. If the
directions are bias-specific, cross-category cosines should land in the reference
paper's 0.4–0.6 band, well below the within-category floor.

**This control is the whole experiment in miniature and should be run first.**

### 5.4 THIRD analysis — the run-1 contrast, kept deliberately

Keep the margin-extremes contrast from run 1 as a declared secondary, on the same
items and models.

**Purpose:** to demonstrate directly that a behaviour-derived contrast does not
reproduce where an annotation-derived contrast does. Run 1's numbers already exist
for this; run 2 makes it a controlled comparison at matched n.

This converts run 1's failure into a finding, and it is the most defensible piece
of novelty we have.

---

## 6. What changes in `13-preregistration.md`

| section | change |
|---|---|
| §2.1 primary estimator | **CHANGED.** Was `extremes` over margin quintiles. Now: difference of means between annotation-labelled arms (§5.1 above). Still has no free hyperparameter. |
| §2.2 secondary probe | **DEMOTED.** The ridge probe was built to squeeze signal out of a graded behavioural margin. With an annotation-derived binary contrast there is no graded target, so the probe is not applicable to the primary. Retain it only for §5.4. |
| §3 usability criterion | **KEPT.** Beats-own-negative-control with disjoint CIs remains correct and is strictly better than the reference paper, which reports a floor but no formal criterion. |
| §4 n_splits = 400 | **KEPT.** Their stability estimates come from a handful of splits; 400 with a bootstrap CI is a genuine improvement and costs nothing once residuals are cached. |
| §5 sampling | **CHANGED.** n is no longer 600 per category by margin rank. It is now *all available matched items per arm*, with **32/32 as the declared minimum** following Arditi et al. and the reference paper. Report the achieved n per category. |
| §6 abstention | **DEMOTED to §5.4 only.** Abstention eligibility was a rule about pole assignment by margin. The primary contrast does not assign poles by margin, so it does not apply. Still reported descriptively. |
| §8 capture site | **AMENDED.** Keep all-layer capture, but add the reference paper's site as the primary read: `resid_pre`, chat token index **−2**, mid-layer. |
| §9 stratified splits | **KEPT**, now stratified by arm (ambig/disambig) rather than by margin pole. |
| §10 permutation null | **KEPT**, now shuffling arm labels within a category. |
| §12.1 M1 | **CLOSED** — see §4.1. Remove from the limitations paragraph. |
| §12.2 S3 residual | **MOOT for the primary**, since the primary has no alpha. Applies only to §5.4. |
| §12.3 M2 heavy tails | **MOOT for the primary.** Kurtosis mattered because the extremes contrast selected distribution tails. The annotation contrast uses all items in each arm. Applies only to §5.4. |

---

## 7. What changes in `14-run-plan.md`

| item | change |
|---|---|
| residual caching, 16 GB, float32 | **KEPT — still the hard requirement.** |
| continuous sync, verifier, termination gate | **KEPT unchanged.** |
| queue runner with real exit codes | **KEPT unchanged.** |
| pre-rental checklist | **KEPT**, plus: verify GemmaScope SAE availability before renting if the SAE stage is in scope. |
| model list | **AMENDED.** Add `google/gemma-2-9b-it` — it is the reference paper's model and the only one with public GemmaScope SAEs. Without it the SAE stage is impossible. |
| per-model passes | **CHANGED.** The five passes become: (1) residual capture for both arms, (2) generation for behavioural scoring, (3) task control on disambiguated items, (4) steering sweep, (5) margins — now only for §5.4. |
| budget | **REVISED UPWARD.** gemma-2-9b-it is larger than anything in run 1's list, and the steering sweep adds generation passes. Re-derive before renting; the run-1 anchor of 4.0 ops/sec on a 14B model still applies. |
| new stage | **STEERING**, following §2.6–2.7: unit-normalise every direction, sweep α, use a balanced controlled test set so the unsteered baseline is fixed by construction. |
| new stage (optional) | **SAE**, following §2.8. Requires gemma-2-9b-it. Treat as stretch scope — cut it before cutting the steering stage. |

---

## 8. What does NOT change

Everything in `11-EXPERIMENT-PROTOCOL.md` about process. Everything in
`12-retrospective.md` about what went wrong in run 1 — that audit is historical
and remains accurate. And these run-1 results, which are unaffected because they
never depended on the contrast being reinterpreted here:

- Design 1 (generation + parsing) is **weak, not random** — see
  `16-method-1-reexamined.md`, written concurrently. Pooled person-consistency is
  **58.0%** (n=1142, z=+5.39, p=7.2e-08), not a coin flip. It is still the wrong
  instrument here, but the original "48–68% against a 50% line" framing overstated
  the case and must be corrected wherever it appears.
  **⚠ SUSPENDED 2026-08-23 by defect N6 — see `18-parser-audit.md`.**
  This document previously stated that category-level consistency (Disability
  73.1%, Physical_appearance 68.4% … Race_ethnicity 48.3%, Sexual_orientation
  47.5%) ranks the categories in the same order as the extraction floor, and
  called that converging evidence.
  **Do not report that as converging evidence until the parser is validated.**
  N6 established that the choice parser resolves ties by *earliest mention*, so
  every one of its failure modes lands on whichever option is named first. A
  first-mention-biased parser produces the exact signature we read as
  "person-consistency below 100%" *even if the model is perfectly consistent* —
  the model says the same words in both orders and the parser flips the label.
  The 58.0% pooled figure and the whole per-category ranking are therefore
  contaminated by an unknown amount, and run 1 cannot separate the two
  explanations because the raw response text was never saved (S5, biting twice).
  The ranking may well survive validation — it is a striking correlation — but it
  is currently **UNVERIFIED**, not converging evidence.
- Design 2 (likelihood with option list) fails on copying: 0.38 nats per slot,
  0.38·√2 = 0.54 = observed mean margin.
- Design 3 (likelihood, no option list) is valid; task control 67–89% vs 33%.
- BBQ's own answer key must be used; coverage 30.7% → 99.97%.
- llama3-8b is gated **and** behaviourally inert (6/96 baseline). Excluded.

---

## 9. Corrections to earlier documents

Recorded so nobody re-derives the old version:

1. **"The extraction floor is a validity check this literature skips."** False for
   the reference paper. Remove wherever it appears — `08-results`, the published
   artifact, and any draft framing.
2. **"M1 is not closable by this design."** True of the run-1 design, false of the
   design we are now adopting. `12-retrospective.md` and `13-preregistration.md`
   both need this corrected.
3. **"BBQ ships no external label."** False. `additional_metadata.csv` carries
   `target_loc` (86,157 rows), and the item files carry `question_polarity` and
   `context_condition`. Jeremiah caught this; it is what led to reading the paper.
4. **"Run 1 showed bias has no reproducible directions."** Overstated. It showed a
   behaviour-derived contrast does not reproduce.

---

## 10. Decisions still open for Jeremiah

1. ~~**Is the SAE stage in scope?**~~ **DECIDED 2026-08-23 by Jeremiah: OUT OF
   SCOPE.** Do not build it, do not budget for it, and do not re-open this.
   Consequences, so nobody has to re-derive them:
   - **`gemma-2-9b-it` is NOT added.** It was only ever proposed to make
     GemmaScope SAEs usable. The model list stays at the five run-1 models.
   - The GemmaScope pre-rental check is removed from `14-run-plan.md` §4.
   - `14-run-plan.md` §0.4's "SAE — stretch scope" paragraph is void.
   - **The steering stage stays and is mandatory.** It is the other half of the
     reference paper's activation-space result and is unaffected by this decision.
   - The paper's claim becomes descriptive (bias directions are distinct and steer
     interchangeably) rather than mechanistic (why). That is a complete result and
     stands on its own.
2. **Two models or five?** The reference paper uses two. We used five in run 1 and
   the cross-family replication is genuinely ours.
   *Recommendation: keep five for the activation-space work; gemma-2-9b-it only
   for SAEs.*
3. **Does the steering stage need a bias judge?** Their controlled test set is
   labelled by WildGuard with manual validation. Our equivalent needs a way to
   score whether a steered response is stereotyped. BBQ's likelihood scoring gives
   this for free and needs no judge — confirm that is acceptable before building
   anything heavier.
