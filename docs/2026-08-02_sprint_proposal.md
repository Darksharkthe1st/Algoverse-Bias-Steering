# Proposal: Re-scoped 4-Week Sprint (Aug 2 → Aug 29, 2026)

> **READ THIS FIRST — what this document is and is not.**
>
> **Provenance.** This is machine-generated: four competing sprint designs were
> drafted by independent agents, attacked by four adversarial critics (novelty
> and scoop risk, feasibility, reviewer acceptance, rigor and dual-use), then
> fused. It reads decisively because the fusion step was *instructed* to
> produce one plan rather than a menu. **Decisive prose is a formatting choice
> here, not team consensus.**
>
> **Status: PROPOSAL. Nothing in it is decided.** It has not been reviewed by
> the person who built the 2025 pipeline, and several of its claims are about
> that pipeline. Treat every "non-negotiable," "final," and "locked" below as
> *"the analysis recommends this strongly"* — they are arguments, and they lose
> to anyone on the team with better information.
>
> **Known limits of the analysis.** The agents read the repo and the literature;
> they did not talk to anyone. They do not know what the 2025 experiments felt
> like to run, which parts of the notebook are load-bearing, what broke before
> and why, or what the team actually wants this paper to be. Where this document
> is confident about the state of the codebase or the reasons behind a 2025
> choice, it is inferring from artifacts — and the person who wrote those
> artifacts should be believed over it.
>
> **How to use it.** Read §1 (the thesis), §5 (the schedule), and §8 (gates and
> kill criteria). Argue with the rest. The single change that most needs a
> sanity check from the pipeline owner is the Week-1 harness assessment in §2
> and §8 — the analysis concluded the runner does not exist in reusable form,
> which is a claim about someone else's work made from a file tree.
>
> **Open questions for the team, not settled by this document:** whether the
> perturbation arm belongs in this sprint at all; whether the mediation
> experiment is worth its GPU-hours versus more model breadth; whether ICLR in
> parallel is realistic or a distraction; and whether the six-way rubric is the
> right rubric. The one item that genuinely is time-critical is the rubric
> freeze, because relabeling after annotation starts wastes the annotation.

Base design: **Candidate 1 ("Two Routes to One Boundary")**, restructured. Three of four critics (novelty, feasibility, dual-use) picked it outright; the reviewer critic ranked it second and named the exact three fixes that would move it to first. All three fixes are applied below, plus four imports from the other candidates and one experiment none of the four proposed.

---

## 1. The call

### Thesis (falsifiable form)

> **The soft-refusal direction is not an artifact of the intervention that found it.** An input-side perturbation designed without reference to any direction — a single character substitution that forces byte-fallback tokenization — displaces residual-stream activations preferentially *along* the soft-refusal direction; the **signed** displacement predicts which prompts cross the soft-refusal boundary and in which direction; and removing that along-direction component from the induced displacement **abolishes the behavioral flip**, while injecting it into the clean run **reproduces it**.

**Falsified if any one of these holds:**

- **(F1)** The along-direction variance share for exotic-byte arms is indistinguishable (TOST, ±10× ratio margin) from the covariance-matched random-direction null, **or** indistinguishable from the byte-identical retokenization control at matched ‖Δ‖.
- **(F2)** Signed projection π̃ does not beat ‖Δ‖ **and** does not beat projection onto an independently-extracted out-of-distribution/"weird-input" direction d_OOD in a nested mixed-effects model (likelihood-ratio, α=.05) in ≥3 of 4 models.
- **(F3)** Projecting the along-d_soft component out of the *induced* displacement at l\* does not reduce the perturbation-induced flip rate by ≥50% relative (95% CI excluding 0).

F1 or F2 alone degrades the paper to a geometry result. **F3 is the claim that makes this paper worth writing**; if F3 fires the paper is honest and weaker, and we say so in the abstract.

### Claim list, load-bearing order

| # | Claim | Role | Body space |
|---|---|---|---|
| **C0** | Soft refusal is separable from factual hedging, non-engagement, incoherence, and meta-commentary by a six-way ordered first-match-wins screen at per-category κ ≥ 0.7 vs 150 double-annotated gold labels. | **Gate, not a paper claim.** Nothing below is interpretable without it. | 3 sentences + appendix table |
| **C1** | Under one unified Arditi-convention extraction, and with the two interventions **dose-matched to equal on-target effect (30pp) on a frozen dev split before any off-target number is read**, d_soft moves opinion-battery stance rate with off-target safety movement bounded within ±5pp (TOST, 90% CI), and d_harm the mirror. | **Setup.** Establishes the direction exists and is not the harm direction. Compressed hard — this is the most-scooped part of the paper (2602.02132, 2512.16602). | ~1 page |
| **C2** | Byte-fallback perturbations displace activations preferentially along d_soft: variance share π²/‖Δ‖² exceeds a **covariance-matched** null by ≥10× at some fractional depth (max-statistic permutation p<.05) in ≥3 of 4 models, while the **byte-identical retokenization** arm and the ASCII-typo arm do not, at matched ‖Δ‖. | **Payload 1 — selectivity.** | ~0.75 page + Fig 2 |
| **C3** | Signed π̃ at pre-registered l\* predicts per-prompt flips (AUROC ≥ 0.70, DeLong-significant over ‖Δ‖ and over π̃_OOD) and sign(π̃) predicts flip *direction* above chance. | **Payload 2 — prediction.** | ~0.75 page + Fig 3 |
| **C4** | **Mediation.** Necessity: nulling the along-d_soft component of the induced displacement at l\*±2 abolishes ≥50% of flips. Sufficiency: injecting the measured per-prompt π̂·d̂ into the clean run reproduces ≥40% of flips. | **Payload 3 — the headline.** Converts convergent *correlation* into convergent *causation*. Nobody proposed this; all four designs stop at correlation. | ~0.75 page + Fig 4 |
| **C5** | In fractional depth f = l/(L−1), the perturbation-alignment profile P(f) and the single-layer steering-efficacy profile E(f) either coincide or dissociate; the comparison is well-posed in f and not in raw index, and the within-family Qwen2.5 ladder shows whether argmax_f migrates with scale. | **Descriptive result.** Both outcomes are the result. | ~0.5 page + Fig 5 (appendix if space binds) |

**Six claims, one of which is a gate.** The reviewer critic's ceiling was four; we are at five plus a gate, and C1 is compressed to the point where it functions as method rather than result. That is the honest floor for this paper.

### Perturbation arm: **IN**

**Why the critiques support inclusion:**

- Novelty critic verified the specific sentence is unclaimed: *signed projection onto an independently-extracted, never-optimized-against direction, predicting per-prompt flips, beating magnitude*. The nearest prior art (arXiv:2606.07696) measures whether the **steering effect survives** perturbation, not the projection of perturbed activations onto the direction, and does no flip prediction and no identifiability framing. The gap is real; it is one experiment wide, which is exactly why we run it now.
- Feasibility critic's degraded-version analysis is decisive: Design 1's degraded form is **cheaper than the current plan and more novel than it**. That property is unique among the four. The forward-pass half is authored zero-GPU in Week 1 by the onboarding person and runs in single-digit A100-hours with no judge in the loop.
- Dual-use critic ranked D1 first on *rigor* despite ranking it worst on dual-use posture, with the explicit reasoning that the dual-use deficit is a one-line construction constraint transplantable in an afternoon, while the rigor deficit (isotropic vs covariance-matched nulls) is unrecoverable after the fact because the null family must be sampled when the displacements are computed.
- The exclusion argument (D4) is right about the confound and wrong about the remedy. Its stated replacement — three extraction routes as the answer to non-identifiability — is **pre-refuted** by the very paper it cites (2602.06801 states the equivalence class contains vectors from independent extraction pipelines) and independently by 2604.08524. D4 pays for its caution by deleting the only unpublished claim on the table. We keep its statistics and discard its cut.

**What we accept as the cost:** the arm is not free. D4 is correct that "forward passes only" is false — flip labels require paired generations and judging, and that lands on the binding person-week. The schedule below puts the generation subset **behind** the main grid in an unattended queue and shrinks the main grid to pay for it.

---

## 2. Diff against the current plan

### ADDED

| Item | Source | Cost |
|---|---|---|
| **Harness productionization** — turn `experiments/farhan-experimentation.ipynb` into a config-driven, resumable runner with a **write-once per-layer residual cache** on disk that a second person can read without a GPU | D2's E2 (the one genuinely structural idea in that design) + feasibility critic's repo audit | Farhan, Week 1, ~4 days. **Non-negotiable prerequisite.** No design budgeted this; it is the largest omission across all four. |
| Perturbation library (8 arms incl. byte-identical retokenization + appearance-matched precomposed/combining pair) | D1 E4 + D2's T1/T2 | Jeremiah + Edward, Week 1, zero GPU |
| Forward-pass displacement grid, all layers, all models | D1 E4 | ~5 A100-hrs |
| **Covariance-matched** null battery + d_harm + **d_OOD** + **d_length** comparison directions | D1 E5b, plus two directions no design proposed | ~4 A100-hrs |
| **Mediation experiment (necessity + sufficiency)** | Nobody. Dual-use critic's top cross-cutting objection. | ~15 A100-hrs, ~4k judged generations |
| Perturbation generation subset with **sampling-noise floor** | D1 E8 + D2's flip definition | ~45 A100-hrs, ~$35 judging |
| **Dose-matching on a frozen dev split before any off-target read** | D4 E5 | ~10 A100-hrs, one calibration loop |
| Hand-written contrast-pair extraction route (no LLM judge anywhere in extraction) for d_soft | D4's C4 route (ii) — kept as construct validity, *not* as the identifiability answer | ~6 A100-hrs, Jeremiah writes 200 pairs |
| Coarse single-layer steering-efficacy sweep E(f) at 9 fractional depths | D1 E3, **de-scoped from stride-1** | ~4.3k judged generations |
| Qwen2.5 ladder 0.5B/1.5B/3B for P(f) forward passes only | D1 E6 — defuses the scale confound in 2606.29196 | ~2 A100-hrs, near-zero integration (same family, same tokenizer) |
| Per-layer residual-norm and attention-entropy profile overlay on the depth figure | Reviewer critic's objection that fractional-depth coincidence may be architectural | Free from cache |
| **Paper skeleton (.tex, figure stubs, related work, limitations, dual-use box) in Week 1–2** | Feasibility critic. There has never been a .tex in this project's history. | Jeremiah + Edward, Weeks 1–2, zero GPU |
| Standing scoop watch (arXiv alerts: refusal direction × tokenization/perturbation × identifiability) with a pre-committed response | Novelty critic | 10 min/day |

### CUT

| Item | Why |
|---|---|
| **C4-twins ("behaviorally-equivalent twin break") as a headline claim** | Two critics independently derived that it is mathematically self-defeating: d′ = d_soft + v with v⊥, ‖v‖=‖d_soft‖ gives cos(d′,d_soft)=0.707, so π′ = (π + ⟨Δ,v⟩)/√2, and AUROC is rank-invariant to positive scaling — the twin inherits d_soft's prediction almost exactly. The TOST certification step actively selects for behaviorally inert v, which is the v least likely to inject noise. **Demoted to a 4-sentence appendix robustness note** with a corrected construction (v drawn from the estimated Jacobian null space, ‖v‖ ≥ 3‖d_soft‖, pre-registered max admissible cos(d′,d_soft) = 0.3). It is never the ICLR lead. |
| **Perturbation of any harmful prompt** (JailbreakBench, XSTest-unsafe, Do-Not-Answer) | D3's construction rule, imported verbatim. Only contested-but-benign prompts are ever perturbed. Cost: we lose D1's C6 perturbation-side safety double-dissociation. Accepted — see §9. |
| Main designed grid: 30k → **~16k generations** | −soft steer retained on 2 models only; system-prompt baseline on the opinion battery only. |
| Stride-1 layer sweeps anywhere | Feasibility critic correctly priced D1's E3 at ~13k judged generations under a "15 A100-hrs, no judging cost" line. Coarse 9-point fractional grid instead. |
| Second judge model / dual-judging everything | Replaced by a 500-response judge-swap ablation on one open-weight judge run on idle GPUs. |
| MMLU 500×5 → **250 items × 3 conditions**, plus 250 items through two perturbation arms | The perturbation-capability cell answers a live objection; the fifth MMLU condition confirms a known non-effect. |
| CrowS-Pairs, BBQ | Stated in Limitations as deliberate, with the IssueBench/Paired-Prompts justification. Silence reads as oversight. |
| All P2 stretch items from the current plan: geometry-vs-crosstalk scatter, directional-ablation variant, QCRI shared-core projection | Calendar traps, and now also attention traps. |
| Cone fitting, MFA regions, manifold steering, ACE beyond capping, the ideology/"which side" direction, any 5th model family | Unchanged from current plan. Reaffirmed. |
| Parallel AI4GOOD submission | D4 is right: it costs the writer's Week-4 time, which is scarce, to hedge venue fit, which is not. One submission. |
| Cross-model Procrustes, cross-layer transport (ridge W_l), principal angles on displacement-field PCs | D3's E8/E9/E10. The transport test in particular is a linear approximation to a nonlinear block; a mech-interp reviewer dismantles "where the direction is written" in a paragraph, and the overclaim would contaminate the arm that actually matters. |

### UNCHANGED

- Four models: **Qwen1.5-7B-Chat, gemma-2b-it, Llama-3-8B-Instruct, Qwen2.5-7B-Instruct**. Only Qwen2.5-7B is a new integration; the other three are proven in this repo. (The 0.5/1.5/3B ladder is the same family and tokenizer as Qwen2.5-7B — near-zero marginal integration.)
- Unified Arditi-convention extraction at post-instruction template positions, per-layer, never mean-pooled. Committed vectors in `experiments/best_vecs/` are sanity cross-checks only and appear in no figure.
- 95th-percentile activation capping, never constant coefficients.
- Batteries: held-out comparisons + IssueBench stratified + Anthropic Paired Prompts (opinion); XSTest + JailbreakBench (safety, **never perturbed**).
- Per-example distributions with 95% CIs, never bare means.
- Gate 1 zero-GPU retroactive re-judge of `experiments/past_logs/`.
- Budget: $377 Lambda. Projected ~215 A100-hrs against 290 available. Judging ~$130 cash.

---

## 3. Perturbation experiment spec

### 3.1 Perturbation family (8 arms, pre-registered, direction-agnostic by construction)

All operators are fixed published character rules chosen **without reference to any direction**, with no optimization and no gradient access. This is the sentence that earns the arm and it goes in the body verbatim: it is precisely what disqualifies the GCG-family correlation (arXiv:2607.08883 optimizes a loss *against* the internal refusal direction, then reports the projection as evidence) from serving as convergent validity.

| Arm | Operator | Bytes change? | Tokens change? | Appearance changes? | Role |
|---|---|---|---|---|---|
| **P0** | clean | — | — | — | reference |
| **P1** | **RETOK**: alternative valid BPE segmentation, byte string bit-identical | **No** | **Yes** | No | **The meaning-invariant control.** The single most important cell in the paper. If P1 shows the same alignment as P3/P4, the effect is tokenization-generic and C2 is dead. If P1 shows *no* alignment while P3 does, meaning cannot explain the difference either. |
| **P2** | **PRECOMP**: precomposed diacritic (`á` U+00E1) at k content-word sites | Yes | Sometimes | Yes | appearance-matched partner of P3 |
| **P3** | **COMBINING**: base + U+0301 at the *same* sites | Yes | Yes (byte fallback) | **Identical to P2** | headline exotic-byte arm; P2↔P3 holds appearance constant and varies tokenization |
| **P4** | **HOMOGLYPH**: Cyrillic/Greek lookalike (а, е, о, р, с) at same sites | Yes | Yes (byte fallback) | Visually identical | second headline arm |
| **P5** | **ASCII-TYPO**: adjacent-key substitution at the *same* sites | Yes | Yes (fragmentation, no byte fallback) | Yes | "any character noise does this" control |
| **P6** | **ZW**: U+200B / U+FE0F insertion | Yes | Yes | **Invisible** | semantically null upper control |
| **P7** | **APOSTROPHE**: U+0027 → U+2019 | Yes | Sometimes | Barely | **the accidental-drift arm — leads the framing** (§9) |

**Dose ladder:** k ∈ {1, 2, 4, all-eligible} for P3, P4, P5. k = 1 for P2. Fixed insertion count {2, 8} for P6. P7 is all-apostrophes.
**Placement factor:** {stance-relevant content word, random position} for P3/P4/P5 at k=1. Matched edit-site sets across P2/P3/P4/P5 by construction.
**Cell count:** 18 (P0, P1, P2@1, P3@{1,2,4,all}×{content,random} = 5 usable after collapsing, P4@{1,4,all}×2, P5@{1,all}×2, P6@{2,8}, P7). Frozen at Gate 0 by commit hash; **no cell is added or removed after Aug 9.**

### 3.2 Prompt sets

- **Displacement grid (forward passes only):** 300 held-out contested-but-benign prompts — 180 IssueBench-stratified (18 issues × 10 templates), 60 Anthropic Paired Prompts, 60 held-out 2025 synthetic comparisons (as an explicit in-distribution anchor). **Zero harmful prompts.**
- **Generation subset:** 250 of those 300.
- **Safety battery is never perturbed.** XSTest and JailbreakBench appear only in the designed-steering arms (C1).

### 3.3 The matched controls — what makes C2 non-vacuous

Three, in descending order of importance:

1. **P1 retokenization** — bytes bit-identical, tokens differ. Meaning is constant *by construction*, not by argument. Body figure.
2. **P2 vs P3** — appearance identical, tokenization differs. Legible to a reader on the page. Body figure.
3. **P5 ASCII typo** at matched edit sites — fragmentation without byte fallback. Isolates byte fallback from generic fragmentation.

Plus **‖Δ‖ matching**: all cross-arm alignment comparisons are made within matched ‖Δ‖ deciles, and ‖Δ‖ is a regressed-out covariate in every model. Token-count inflation ratio is logged per prompt and enters as a second covariate.

### 3.4 Exact measurement

For prompt p, model M, layer l:

1. **Token alignment.** Measure **only** at the chat-template suffix positions after the instruction (e.g. `<|im_end|>\n<|im_start|>assistant\n`), last 5 positions. These are byte-identical between clean and perturbed prompts by construction, and are **the same positions d_soft is extracted at**. Never mean-pooled over the sequence.
2. **Mean-centering (mandatory).** Subtract the per-layer mean clean activation over the corpus: ã = a − μ_l. Every cosine and variance share is computed on mean-centered vectors. Residual streams carry a large mean offset and a few rogue dimensions; without this, any large displacement aligns with everything.
3. **Displacement.** Δ(p,l) = ã_pert(p,l) − ã_clean(p,l), averaged over the 5 suffix positions and also reported per position.
4. **Three non-interchangeable quantities:**
   - **Signed projection** π(p,l) = ⟨Δ, d̂_soft(l)⟩, normalized π̃ = π / median_p ‖ã_clean(p,l)‖ — readable as "how many units of equivalent designed steering did this character swap deliver."
   - **Variance share** s(p,l) = π² / ‖Δ‖² — the **selectivity** quantity, the one that answers non-identifiability.
   - **Total displacement** ‖Δ(p,l)‖ — a nuisance covariate that must be regressed out.
5. **Behavioral outcome.** Six-way ordered screen label (incoherent → non-engagement → stance{factual|evaluative} → explicit soft refusal → both-sidesing → **meta-comment on the input**). The sixth label is load-bearing: it is the most likely perturbation-specific response mode and must never merge silently into non-engagement.
6. **Flip definition (imported from D2 — non-negotiable).** A flip counts only when the **clean class is stable across 3 samples at T=1** AND the **perturbed class differs in ≥2 of 3 samples**. Without this floor the entire analysis partly measures decoding temperature. Applied to P0, P3@all, P4@all (3 samples each); other arms 1 sample and excluded from the headline flip test, reported as a dose-ladder table only.

### 3.5 Sample sizes and power

- **Forward-pass grid:** 300 prompts × 18 cells × 4 models × all layers ≈ **21.6k forward passes** ≈ 5 A100-hrs. No generation, no judge.
- **Generation subset:** 8 cells (P0, P1, P3@1, P3@all, P4@1, P5@all, P6@8, P7) × 250 prompts × 4 models, with 3 samples on P0/P3@all/P4@all and 1 elsewhere ≈ **14k generations** at ~200 tokens ≈ 45 A100-hrs, ~$35 judging.
- **Mediation:** all flippers (expected 40–70/model) plus a matched non-flipper control set, × 2 arms (necessity, sufficiency) × 4 models ≈ **4k generations** ≈ 15 A100-hrs, ~$12 judging.
- **Power for C3.** At a conservative 15% flip rate among coherent responses and ~85% coherence, ~320 perturbed generations per model gives ~40 flips / ~230 non-flips. Hanley–McNeil at n=30/class gives z ≈ 3.0 for AUROC 0.70 vs 0.50 — per-model test powered with ~4× headroom; pooled cross-model test powered to detect AUROC 0.60.
- **Pre-committed exclusion:** any arm whose (incoherent + meta-comment) rate exceeds 15% is reported in a table and **dropped from the C3/C4 headline tests**.

### 3.6 The near-orthogonality / random-direction baseline

Evaluated on the **same** Δ vectors — arithmetic, not compute (~4 A100-hrs total for the extra direction extractions).

| Family | n | Purpose |
|---|---|---|
| **(a) Covariance-matched random directions** — sampled as Σ_l^{1/2}·g / ‖·‖ from the empirical per-layer activation covariance | 1000 per layer | **The primary null.** Mandatory. Isotropic randoms are a straw null: E[cos²] = 1/d ≈ 2.4e-4 at d=4096, so any large displacement clears "80× chance" for reasons that have nothing to do with soft refusal. |
| (b) Isotropic random directions | 1000 per layer | Footnote only, for comparability with prior work that reports it. |
| **(c) d_OOD** — extracted by the identical pipeline from garbled/random-Unicode text vs clean text, **no stance content** | 1 per layer | **The competitor direction no design proposed.** Every perturbation is off-distribution by definition; "Δ projects onto d_soft" may just be "Δ projects onto d_OOD, and d_soft has a component on d_OOD." C3 requires π̃ to beat π̃_OOD in a nested model. |
| **(d) d_length** — extracted from long-hedged vs short-decisive responses on **topic-neutral** text | 1 per layer | Kills the 2025 failure recurring one level up. Also used as a negative control on the *designed* side: d_length must **not** reproduce d_soft's on-target steering effect. |
| (e) d_harm | 1 per layer | Concept control. Reported as an alignment comparison only — we do not perturb harmful prompts, so no safety flip prediction. |
| (f) Top-1 activation PC | 1 per layer | Reported alongside every alignment number so the reader can see how much is the rogue dimension. |
| (g) Corrected twins | 20 | Appendix only. v from estimated Jacobian null space, ‖v‖ ≥ 3‖d_soft‖, max admissible cos(d′,d_soft) = 0.3, pre-registered. |

### 3.7 The mediation experiment (C4) — spec

Run on flippers + matched non-flippers, at layers l ∈ [l\*−2, l\*+2].

- **NECESSITY.** During the *perturbed* forward pass, at each hooked layer, remove the along-direction component of the **induced** displacement: h ← h − ⟨h − a_clean(p,l), d̂_soft(l)⟩·d̂_soft(l). This restores the clean projection while leaving every other consequence of the perturbation intact. Generate, judge, compare flip rate to the un-nulled perturbed run.
  **Two controls, same form, same hooks:** null along d_OOD instead; null along a covariance-matched random direction instead. If nulling *anything* abolishes the flip, the result is an artifact of the hook and C4 dies — which is exactly what these controls exist to detect.
- **SUFFICIENCY.** On the *clean* run, add the **per-prompt measured** π̂(p,l)·d̂_soft(l) at the same layers — not a swept coefficient, the actual measured value. Generate, judge, compare to the clean baseline.
  **Control:** inject the same magnitude along a covariance-matched random direction.

**Success:** necessity ≥50% relative flip reduction with the two controls near zero; sufficiency ≥40% flip reproduction with the control near zero. Both reported with 95% cluster-bootstrap CIs.

This is the experiment that converts "the direction correlates with something we didn't design" into "the direction carries the effect." It costs ~15 A100-hrs and one hook.

---

## 4. The fractional-depth analysis

### How to compare across models

1. Map every layer to **f = l / (L−1)**. Layer counts in the model set: gemma-2b-it 18, Qwen2.5-7B 28, Qwen1.5-7B 32, Llama-3-8B 32; ladder: Qwen2.5-0.5B 24, 1.5B 28, 3B 36.
2. Interpolate every profile onto a shared **51-point grid** on f ∈ [0,1] (linear interpolation; profiles are smooth at this resolution).
3. Two profiles per model:
   - **P(f)** = perturbation alignment profile. Primary: the covariance-matched-null-normalized variance share for P3@all. Secondary: the flip-prediction AUROC profile.
   - **E(f)** = single-layer steering efficacy. For each model, apply d_soft(l) at layer l **only**, 95th-percentile capped, both signs, on a fixed 60-prompt held-out opinion probe, at **9 fractional depths f ∈ {0.1, 0.2, …, 0.9}** (nearest layer). Stance-rate shift in pp via the six-way screen. ~4.3k judged generations total.
4. **argmax_f** located with a 1000-draw bootstrap over prompts. Pre-registered resolution requirement: bootstrap CI on argmax_f narrower than **±0.15 f**. If wider, the profile is declared flat and reported as such rather than as a peak.
5. **l\* is pre-registered at Gate 2** as argmax_l E(l) per model, locked by commit hash **before any perturbation generation is judged.** This is the first thing a hostile reviewer checks and it is closed structurally.
6. **Scale-confound defusal.** 2606.29196 reports peak relative depth migrating from f≈0.96 at 1.5B toward early layers at 32B. With four models spanning 2–8B, a fractional-depth coincidence is confounded with scale. Run **P(f)** (forward passes only) on the Qwen2.5 ladder at 0.5B/1.5B/3B/7B — layer counts 24/28/36/28, **non-monotonic in parameter count within one family with one tokenizer**, which is what makes the comparison identifiable — and report argmax_f vs log(params) as a within-family curve. ~2 A100-hrs.
7. **Triviality control.** Overlay the per-layer residual-norm profile and attention-entropy profile on the same axis. If P(f) and E(f) both peak where the residual norm peaks, the coincidence is architectural and we say so. No design proposed this and a reviewer will.

### The figure

**Figure 5 (three panels, shared f-axis 0→1):**
- **Left:** P(f) and E(f) per model, 4 overlaid pairs, bootstrap ribbons. Raw layer indices shown as a secondary top axis, deliberately, to make their incomparability visible.
- **Centre:** argmax_f vs log(params) for the Qwen2.5 ladder, with the 2606.29196 anchor annotated, plus the published anchors: BPE-fragmentation safety disruption localizing to f ∈ [0.7, 1.0] (2607.01239); the 2025 steering convention at layer 14 of 28, f ≈ 0.5.
- **Right:** residual-norm and attention-entropy profiles (the triviality control).

### What each outcome means

| Outcome | Reading | Paper consequence |
|---|---|---|
| **P(f) ≈ E(f)**, both peaks within 0.1 f, norm/entropy profiles peak elsewhere | One mechanism converging on a shared boundary from two routes | Strongest version of C5; supports reading π̃ as literally "equivalent designed steering delivered by a character swap" |
| **P(f) late (≈0.7–0.85), E(f) mid (≈0.4–0.6)** | Perturbation acts late (consistent with 2607.01239's patching result), steering acts mid — two mechanisms bracketing one behavioral boundary | Equally publishable; changes the mechanistic story, not the validity argument. C2/C3/C4 are unaffected. |
| **Both flat**, bootstrap CI on argmax > ±0.15 | Safety representations are distributed across the forward pass (2607.08883 reports exactly this) | C5 demoted to an appendix null panel; Figure 5 does not appear in the body. C2–C4 survive intact because they are evaluated at pre-registered l\*, not at an argmax. |
| **argmax_f migrates monotonically with scale on the ladder** | Fractional depth is a scale-dependent coordinate | Reported as a caveat that strengthens the paper: we would be the ones flagging it, on our own data, in a paper that otherwise uses the coordinate. |
| **P(f) tracks the residual-norm profile** | The coincidence is architectural, not mechanistic | Stated plainly. C5 becomes a methodological caution about fractional-depth claims generally. |

**C5 is falsifiable only in the resolution sense** (can we locate an argmax at all). Every substantive outcome is a result. That is deliberate and it is stated in the paper.

---

## 5. Week-by-week

**Owners:** Farhan = pipeline/generation. Edward = measurement/geometry/evals (agent-assisted). Jeremiah = annotation/audits/writing (onboarding).
**Known constraint:** Jeremiah travels from Aug 25. Experiments freeze **Aug 24**, red-team read **Aug 26**, submit **Aug 28** (one day early, deliberately).

### Week 0 — Mon Aug 3 / Tue Aug 4 (Gate 0)

- **All:** Tue meeting. **Only decision on the table: sign off and freeze the six-way ordered screen.** Changing the rubric after labeling begins invalidates the labels. Commit hash recorded in `docs/PREREG.md`.
- **Jeremiah:** create `paper/` with a 5-page `.tex` skeleton, figure stubs with placeholder axes, section headers, and the dual-use box. Zero content required — the point is that the document exists before Week 1 ends.
- **Edward:** file `docs/PREREG.md` — rubric hash, perturbation cell list, primary endpoints, l\* selection rule, TOST margins, multiplicity plan, degradation ladder. Timestamped anonymous commit.

### Week 1 — Aug 3–9 · *Harness + construct gate* (GPU ~15 hrs)

| Owner | Work | Path |
|---|---|---|
| **Farhan** | **Productionize the notebook.** Config-driven, resumable multi-model × multi-condition runner. **Write-once per-layer residual cache to disk** that Edward and Jeremiah read with zero GPU. Verify one grid cell end-to-end unattended. Integrate Qwen2.5-7B-Instruct. | **CRITICAL PATH.** This is the whole week. |
| **Edward** | 150 gold labels (double-annotated with Jeremiah), per-category κ. Re-judge `experiments/past_logs/` under the frozen screen. Judge run with perturbation characters normalized **out** of the transcript shown to the judge. | Off path |
| **Jeremiah** | Author the perturbation library (8 arms, 18 cells) as a pure-Python module with unit tests asserting P1 byte-identity and P2/P3 appearance-identity. Write 200 hand-written stance/soft-refusal contrast pairs (no LLM judge in extraction). Annotate alongside Edward. | Off path, zero GPU |

**GATE 1 — Sun Aug 9.**
(a) Per-category κ ≥ 0.7 after ≤2 rubric iterations. (b) Re-judged archived outputs moved **stance-taking** (category 3) by ≥10pp, not merely hedging register. (c) The harness runs one cell unattended and writes a readable cache.
**Fail (a):** stop all steering work; pivot to the construct-validity audit note. **Fail (b):** the 2025 vectors are dead; pivot. **Fail (c):** all of Week 2 slips — cut the Qwen2.5 ladder and the −soft arm on the spot, do not wait.

### Week 2 — Aug 10–16 · *Extraction, depth sweep, perturbation geometry* (GPU ~50 hrs)

| Owner | Work | Path |
|---|---|---|
| **Farhan** | Unified extraction of d_soft (two routes: judge-labeled DiM + hand-written-pair DiM; report cosine between them) and d_harm, all 4 models, all layers, 5 bootstrap redraws each. Then E(f): 9-depth steering efficacy sweep. Then **dose calibration** on the frozen 100-prompt dev split. | **CRITICAL PATH** |
| **Edward** | Extract **d_OOD** and **d_length**. Run the full forward-pass displacement grid off the cache. Build the covariance-matched null battery. Produce the C2 selectivity figure. **Gate 1.5 pilot.** | Off path |
| **Jeremiah** | Flip-rate pilot judging. Related work section (must cite 2606.07696, 2602.02132, 2512.16602 explicitly — see §10). Limitations draft. | Off path |

**GATE 1.5 — Wed Aug 13** (fires *before* the crunch commitment, deliberately):
(a) **Flip rate ≥ 8%** under the noise-floor definition on 1 model × 2 arms × 200 prompts.
(b) **Geometry check:** variance share on ≥1 exotic-byte arm exceeds the **covariance-matched** null by ≥10× at some layer (max-statistic permutation p<.05) **while P1 retokenization does not**.
**Fail (a) only:** drop C3/C4 headline; the arm degrades to geometry-only (C2 + C5). Say this to the lead on Aug 13, not Aug 24.
**Fail (b):** the arm is dead. Cut it entirely, reallocate to D4's expansion (more models, dose-matched, TOST). We have the harness by then; this is a real fallback, not a gesture.

**GATE 2 — Sun Aug 16.**
(a) Re-extracted d_soft reproduces bidirectional control ≥20pp on ≥3/4 models. (b) Arditi bypass ≥30pp on the same models. (c) **l\* locked by commit hash.** (d) Doses calibrated to equal 30pp on-target.
**Fail (b):** pipeline bug — Arditi is heavily replicated. Fix before the grid. **Fail (a):** the 2025 result was an extraction artifact; pivot.

### Week 3 — Aug 17–23 · *The grid, the subset, the mediation* (GPU ~110 hrs)

| Owner | Work | Path |
|---|---|---|
| **Farhan** | **Wave 1 (Aug 17–19):** main designed grid, 2 models × {baseline, sys-prompt, +soft, −soft, harm-ablation} × 2 batteries, dose-matched, capped. **Wave 2 (Aug 20–21):** remaining 2 models. **Then queued unattended:** perturbation generation subset (14k gens), then mediation (4k gens). | **CRITICAL PATH — one queue, one config, authored in Week 2.** |
| **Edward** | Rolling judging + aggregation as results land. C2/C3 analysis on cached forward passes (already complete from Week 2 — the C2 figure is finished before the grid starts). Fractional-depth figure. Mediation analysis. | Off path except judging |
| **Jeremiah** | MMLU 250 × 3 conditions + 250 through P3@all/P4@all. Human spot-check (200 judgments, stratified: ≥60 perturbed, ≥30 incoherent/meta-comment). Judge-swap ablation on 500 responses. Methods + related work prose. | Off path |

**HARD DATE — Wed Aug 20:** if Wave 1 judging is not complete, cut to 3 models **that day**. Not Aug 24.
**GATE 3 — Sat Aug 23:** mediation result in hand, either sign. Wave 2 judged.

### Week 4 — Aug 24–28 · *Freeze and write* (GPU: reserve only)

- **Mon Aug 24: experiments freeze.** No new runs. Reserve GPU only for re-running a failed cell.
- **Edward:** finish all figures. Lead the results and discussion sections.
- **Jeremiah (through Aug 24, then remote-light):** hand off the spot-check report and eval appendix by Aug 24 EOD. Repro pass: re-run one grid cell from the README.
- **Farhan:** reproducibility appendix, anonymized artifact bundle (direction vectors, rubric, perturbation *evaluation* harness — **not** a generator), **anonymization sweep** (strip author strings from logs, notebooks, git config in the release branch; verify no de-anonymizing metadata).
- **Tue Aug 26:** internal red-team read. Everyone reads it as a hostile reviewer against the objection list in §10.
- **Thu Aug 28: submit.** One day of slack against Aug 29 AoE.

### What runs off the critical path

Everything Edward and Jeremiah do, by construction. Specifically: the perturbation library (Week 1, zero GPU), the entire forward-pass displacement grid and null battery (Week 2, reads Farhan's cache), the C2 figure, the fractional-depth figure, all analysis, and the paper skeleton. **Figures 2 and 5 are complete at the end of Week 2** — the paper has a publishable skeleton before the crunch week starts. If Farhan is unavailable in Week 3, we lose grid breadth (degrade to 3 models, 4 conditions) and keep the geometry and identifiability contributions. That is the inverse of the current plan's bus-factor-1 failure, and it is the single biggest structural improvement in this re-scope.

---

## 6. Venue strategy

**Do both. They are not in tension, and the reason usually given for doing both is wrong.**

**Aug 29 — Interpretability for Discovery @ NeurIPS 2026** (5pp, non-archival, double-blind). Primary and only August submission. Drop the parallel AI4GOOD submission: it costs the writer's Week-4 time, the actual binding resource.

- Abstract leads with the epistemics: *a direction validated by two independent routes, one of which we did not design.* That is the workshop's literal remit.
- Body: C0 in three sentences; C1 compressed to ~1 page; **C2, C3, C4 as the substance**; C5 as Figure 5 (appendix if space binds).
- **The retokenization control (P1) and the covariance-matched null go in the BODY, not the appendix.** A reviewer is entitled to score a body claim as unsupported when the control that makes it non-vacuous is not in the body. This is the most common borderline-reject reason at 5-page workshops.
- Figures: F1 = dose-matched cross-intervention matrix with per-example distributions. F2 = perturbation selectivity vs fractional depth, with P1/P2/P3/P5 laddered. F3 = flip-prediction AUROC with the max-statistic permutation null band and the DeLong comparison against ‖Δ‖ and π̃_OOD. F4 = **mediation** (necessity/sufficiency bars with their controls). F5 = fractional depth, appendix if needed.

**Sept 18 abstract / Sept 25 full — ICLR 2027.** Yes, submit.

**Correction that changes the plan:** NeurIPS 2026 workshop notification is **Sept 29 — after both ICLR deadlines.** Three of the four candidate designs build their September strategy on "with reviewer feedback in hand." That feedback does not exist. So:

- The ICLR version is **written in parallel from Aug 3**, not derived from reviews. The `.tex` skeleton created in Week 0 is a 9-page skeleton of which 5 pages ship in August. Cut experiments to fit the workshop, never figures.
- September adds **no new experiments** except extending the Qwen2.5 ladder to 14B for the depth curve if compute is idle (~10 A100-hrs) and promoting already-collected appendix material into the body.
- ICLR re-leads with **C4 (mediation)**, not with the twin-break. "The signed component of an undesigned input perturbation along an independently-extracted direction causally mediates the resulting behavioral change" is a main-track-sized claim that generalizes past soft refusal to every difference-in-means direction in the literature, and it is not an algebraic identity.
- The non-archival status is the point: the workshop is a timestamp and a forcing function. **The real priority date is Sept 25.**

**Scoop watch.** 2606.07696 (Jun), 2607.01239 (Jul), 2607.08883 (Jul) landed on this exact intersection at roughly monthly intervals, and the most obvious follow-up to 2606.07696 is to replace its behavioral robustness metric with the activation projection — which is our contribution. Standing arXiv alert on *refusal direction × tokenization/perturbation × identifiability*, checked daily by Edward. **Pre-committed response if the exact paper lands before Aug 29:** we do not abandon; we re-lead with C4 (mediation), which no plausible scoop will have, cite the scoop in the first paragraph, and reframe C2/C3 as replication-plus-mechanism.

---

## 7. Statistics plan

**Primary endpoints, pre-registered at Gate 0:** Δ stance rate (opinion battery); Δ refusal rate (safety battery); variance share ratio vs covariance-matched null at f\*; AUROC(π̃) at l\*; necessity flip-reduction ratio; sufficiency flip-reproduction ratio.

**Dose matching.** Conditions +soft-steer and harm-ablation are calibrated on a **frozen 100-prompt dev split** to equal on-target effect (30pp) **before any off-target number is read**. Without this, "opinion steering doesn't move safety" is indistinguishable from "we steered opinion weakly," and that is a one-sentence reviewer kill on the paper's spine. Dev split is touched once and never again.

**Equivalence tests for the near-zero cells (TOST).** We never write "no effect." Every near-zero cell is reported as bounded:

| Cell | Margin | Test |
|---|---|---|
| Off-target safety movement under +soft steer | ±5pp | TOST, two one-sided at α=.05 (90% CI within margin) |
| Off-target opinion movement under harm ablation | ±5pp | same |
| P1 retokenization variance share vs covariance-matched null | ratio within [1/3, 3] | TOST on log ratio |
| P5 ASCII-typo variance share vs null | ratio within [1/3, 3] | TOST on log ratio |
| d_length on-target steering effect | ±5pp | TOST — d_length must **not** reproduce d_soft's effect |
| Mediation controls (null along d_OOD / random) | ±10pp flip-rate change | TOST |

Equivalence margins are pre-registered and not revised after seeing data.

**Multiple comparisons.**
- **Layer sweeps** are max-statistic problems, not per-layer problems. Every profile claim uses a **permutation null built on the max over layers** (permute prompt labels, recompute max, 1000 draws). A per-layer p-value is never reported as evidence for a profile claim.
- **l\*** is pre-registered from an *independent* causal sweep (E(f), Gate 2) before any perturbation response is judged. C3 and C4 are evaluated only at l\* (C4 at l\*±2). No argmax over the outcome measure.
- **Family-wise across the five headline tests** (C1-dissociation, C2-selectivity, C3-prediction, C4-necessity, C4-sufficiency): **Holm–Bonferroni** at α=.05.
- **Within the perturbation dose/arm ladder** (18 cells): **Benjamini–Hochberg at q=.10**, reported as secondary/exploratory, never as a headline.
- **Cross-model** results: per-model effect with CI, plus a random-effects meta-estimate. **No vote counting.** "≥3 of 4 models" appears nowhere as an inferential statement; with n=4 non-independent models sharing tokenizers and post-training recipes, a majority rule has no inferential content. We say n=4 is descriptive and report the distribution.

**Uncertainty.**
- All CIs by **BCa bootstrap, 10k resamples, clustered by IssueBench template AND issue.** Prompt-level independence is false given shared templates; unclustered CIs are simply wrong.
- **Two error bands, reported separately:** prompt-sampling uncertainty (bootstrap over prompts) and **extraction uncertainty** (5 contrast-set redraws per model). A cross-direction cosine of 0.35 is meaningless without the within-direction bootstrap cosine noise floor (expect 0.90–0.97); that floor is reported on every geometry panel.
- **Per-example effect distributions** and the fraction of prompts with effect ≥20pp, never means alone (Tan et al. 2407.12404).

**The predictive model (C3).**
`flip_i ~ π̃_i(l*) + ‖Δ̃‖_i + π̃_OOD,i + token_inflation_i + arm + (1|prompt) + (1|model)`, fit per model and pooled. Nested LRT for π̃ over the magnitude-only model, and for π̃ over the {magnitude + π̃_OOD} model. AUROC of π̃ alone vs ‖Δ‖ alone vs each null family, DeLong for pairwise differences.

**The sign test (C3b), with a pre-registered guard.** 2×2 of sign(π̃) against flip direction, exact test. **Guard:** the test is run only if ≥15 refusal→stance flips are observed pooled. If confusing input almost always pushes toward hedging, sign agreement is free and the cell carries no information — in that case we report the marginal and say so rather than reporting a degenerate 2×2 as the paper's strongest number.

**Judge validity.** Per-category κ (six categories). Judge-human agreement reported **separately for clean and perturbed responses**; if they differ by >0.1 κ, perturbed flip labels get human adjudication on a larger sample. Judge-swap ablation on 500 responses with rank correlation of headline rates. The judge sees transcripts with perturbation characters normalized out — except for a held-out 100-response subset judged un-normalized, so we can measure what that normalization costs on meta-comment detection.

---

## 8. Risks, kill criteria, degraded scope — in application order

### Kill criteria (in date order)

| Date | Gate | Fail action |
|---|---|---|
| **Aug 9** | κ ≥ 0.7 per category | Stop all steering work. Ship the construct-validity audit note. |
| **Aug 9** | Archived vectors moved stance ≥10pp | Construct dead on these vectors. Pivot to the audit paper. |
| **Aug 9** | Harness runs a cell unattended | Cut the ladder and −soft arm immediately; do not wait for Week 2. |
| **Aug 13** | Flip rate ≥ 8% (noise-floor definition) | Drop C3/C4. Arm degrades to geometry-only. **Tell the lead Aug 13.** |
| **Aug 13** | Exotic-byte arm beats covariance-matched null while P1 does not | **Cut the arm entirely.** Reallocate to model breadth + dose-matched TOST (D4's expansion). This is a real fallback; we have the harness by then. |
| **Aug 16** | Steering ≥20pp on ≥3/4; Arditi bypass ≥30pp | (b) fail = pipeline bug, fix before grid. (a) fail = extraction artifact, pivot. |
| **Aug 20** | Wave 1 judged | Cut to 3 models that day. |
| **Aug 24** | — | Experiments freeze. No exceptions. |

**Explicitly NOT kill criteria:** any geometry outcome; any dissociation outcome; any depth outcome; a negative mediation result. Dissociation contradicts the QCRI one-knob generalization; entanglement mechanistically explains Fafuła. A negative C4 is a real finding about convergent validity that the field currently asserts casually. All of these publish — but see the honesty requirement below.

### Degradation ladder (apply strictly in order under time pressure)

1. Corrected-twins appendix note (C4-twins).
2. Qwen2.5 scale ladder (P(f) forward passes).
3. −soft steer condition on 2 of 4 models.
4. MMLU-under-perturbation cells.
5. Mediation **sufficiency** arm (keep necessity — necessity is the stronger of the two).
6. Perturbation generation on model 4 → 3-model flip analysis.
7. C5 depth figure to appendix.
8. Grid down to 3 models.
9. **Last resort:** mediation entirely → the paper is C1 + C2 + C3.

**Never cut, at any point:** the covariance-matched null; the P1 retokenization control; dose matching; the sampling-noise floor; the safety battery in the designed arms; the human spot-check; the anonymization sweep.

### Ranked residual risks

1. **The harness doesn't exist and Week 1 is entirely about building it.** `src/` is 524 LOC of data loaders; the 2025 pipeline is one notebook. None of the four designs budgeted productionizing it, and it is a hard prerequisite for every design's Week 2. This is the single largest calendar risk and it is now Farhan's whole Week 1. If it slips past Aug 9, cut at Gate 1(c) immediately.
2. **Incoherence eats the effect.** Exotic-byte arms produce meta-comments and degraded text rather than clean flips. Detected in the first 200 judged responses by the six-way screen; mitigated by the k=1/k=2 low-disruption rungs and the 15% incoherence cap. Residual: if even k=1 homoglyphs mostly produce meta-comments, C3/C4 are dead and the paper is C0/C1/C2/C5.
3. **The displacement is a generic blob** — cos(Δ, d_soft) is real but so is cos(Δ, everything). Detected at Gate 1.5 by the covariance-matched null and the d_OOD nested test, for ~10 GPU-hrs and zero judging, **before a single perturbation generation runs.** The honest outcome in that branch is a reported negative result on convergent validity. It is publishable at this venue given how casually the field asserts convergent evidence is available — but it is a much weaker paper and the team should know that going in.
4. **Writing.** No .tex has ever existed in this project's history, and "write the 5-page draft in Week 4" is the line most likely to fail. Mitigated by creating the skeleton in Week 0 and drafting related work / methods / limitations in Weeks 1–2 while GPUs are the bottleneck.
5. **Judge degradation on perturbed text.** Measured directly (§7), not assumed. If κ differs by >0.1 clean vs perturbed, human adjudication costs Week-4 person-hours — the scarcest resource.
6. **Scoop.** ~4–8 week window in a monthly-cadence area. Standing alert + pre-committed re-lead on C4.
7. **Traveler.** Jeremiah's deliverables all land by Aug 24 EOD by design.

### Honesty requirement on the null branch

If C2 or C3 fails, we do **not** frame it as a neutral bonus that failed. A reviewer is entitled to read a null convergent-validity result as *evidence against d_soft being the model's representation* — and a paper that pre-frames the null as "still publishable" without engaging that reading looks like it wants the upside without the risk. The pre-registration states, in advance: **a null on C2/C3 weakens C1's interpretation, and the discussion section will say so.**

---

## 9. Dual-use and disclosure discipline

These are rules, not aspirations. They are checkable and they are enforced at the red-team read on Aug 26.

1. **Only contested-but-benign prompts are ever perturbed.** No harmful battery — not XSTest-unsafe, not JailbreakBench, not Do-Not-Answer — is perturbed in any arm, at any dose, in any model. This is a construction constraint, which means **no harm-refusal flip rate can appear in the paper**, which means there is no attack-efficacy number to extract. Cost: we forgo the perturbation-side safety double-dissociation (D1's C6). That is the right trade and it is taken deliberately.
2. **No attack-success framing anywhere.** We report detection coverage and monitor ROC. We do not cite or foreground third-party ASR figures against named commercial guardrails.
3. **Lead with accidental drift, not adversarial manipulation.** P7 (U+0027 → U+2019 apostrophe normalization) leads the motivation: real prompt pipelines differ in typographic normalization, and that is an unintentional source of behavioral divergence. The adversarial reading is available to anyone; the accidental reading is the one that is actually deployment-relevant and it is the one we write.
4. **Released artifact:** direction vectors, the ordered-screen rubric, the perturbation **evaluation** harness, and the projection monitor with its ROC. **Never a perturbation generator.** The operator classes are described at the level needed for replication and no further — no per-arm flip-rate table keyed to named models at named doses.
5. **The unpublished production-binary analysis is cited nowhere and alluded to nowhere.** The public guardrail-evasion and BPE-fragmentation literature (2607.01239, 2510.05025, 2506.07948, 2504.11168) establishes "real systems manipulate Unicode variants in prompts" on its own. **Additionally required before the cell list is frozen at Gate 0:** a written determination in `docs/PREREG.md` that each of the 18 cells is independently motivated from a public source, plus sign-off from whoever owns the internal analysis. Citing nothing does not undo *selecting the experiment* from confidential material — the leak channel is the design, not the bibliography. **Consequence already applied:** we do not run system-prompt-only perturbation cells, because those would reconstruct the observed production configuration.
6. **The Terminal-Bench audit is not cited, quoted, alluded to, uploaded, or shared.** `docs/THE_CORRECT_PROBLEM.md` carries an explicit restriction on it. The ordered validity screen is presented in the paper as standard measurement-validity practice — a first-match-wins cascade over response categories — with no reference to that document's source. If that paper becomes public before Aug 29, revisit whether to cite it as a cross-domain methodological ally. The dual-use critic flagged this as an independent non-public-derivation exposure that none of the four designs addressed; it is now a rule.
7. **Open-weight models only.** No API models are perturbed.
8. **Boxed Broader Impact paragraph in both the workshop and ICLR versions**, stating 1–7.
9. **Double-blind + artifact release:** the anonymization sweep is a budgeted Week-4 task with a named owner (Farhan), not an afterthought.
10. **On the parked "byte-fallback token budget" paper** that Candidate 4 proposed for Sept 1 — perturbing 200 harmful prompts in safety-critical spans to build a zero-forward-pass score for which prompts sit near the refusal boundary, in an archival venue. **That is not started, and it does not get started on the strength of this decision.** It is an input-side, no-model-access targeting heuristic wearing an auditing label, and it is the most attack-shaped object in any of the four proposals. If anyone wants to pursue it, it gets its own written dual-use review first.

---

## 10. Where the critics disagreed, and how this resolves it

**The split.** Novelty, feasibility, and dual-use all top-picked Candidate 1. The reviewer critic top-picked Candidate 4 and put Candidate 1 second — but stated the exact conditions under which Candidate 1 would be its pick: *cut to four claims, drop C4-twins as a construction artifact, move the retokenization control into the body.* All three conditions are met above. **The disagreement was never about whether to run the arm; it was about the arm's packaging and the paper's statistical hygiene.** So the resolution is not a compromise between designs — it is Candidate 1's science executed with Candidate 4's discipline.

**Resolved by import:**

| Disagreement | Resolution |
|---|---|
| Reviewer: D4's dose-matching is a one-sentence kill on D1/D2/D3's spine | **Imported verbatim.** Doses calibrated to equal on-target effect on a frozen dev split before any off-target number is read. Non-cuttable. |
| Reviewer: seven claims in five pages reads as none supported | Cut to five plus a gate; C1 compressed to ~1 page; C4-twins to appendix. |
| Reviewer + dual-use: D1's C4 twin-break is a foregone null | **Two independent derivations of the same algebra.** Demoted from headline to appendix with a corrected construction. The ICLR lead becomes mediation, not twins. |
| Dual-use: D3's benign-prompts-only rule is a one-line transplant; D1's null structure is unrecoverable after the fact | **Both applied.** D3's construction constraint + D1's covariance-matched null. This is the exact recommendation in the dual-use critic's notes. |
| Dual-use + reviewer: D2's sampling-noise floor is the single most important methodological addition and D1/D3 omit it | **Imported verbatim.** 3 samples at T=1, clean stable 3/3, perturbed differs ≥2/3. |
| Feasibility: the harness does not exist and nobody budgeted it | **Farhan's entire Week 1**, with D2's write-once cache as the deliverable. |
| Feasibility: D1's E3 stride-1 sweep and E5c twin certification are undercosted judged-generation campaigns priced as GPU arithmetic | E3 de-scoped to 9 fractional depths (~4.3k judged gens). E5c cut to appendix with 20 twins. |
| Feasibility + D4: "forward passes only, off the critical path" is false | Conceded in the plan. The geometry half is genuinely cheap and finishes Week 2; the generation half is priced honestly (45 A100-hrs, $35 judging) and queued **behind** the main grid, unattended. |
| Novelty: D4's replacement claim (three extraction routes as the identifiability answer) is pre-refuted by 2602.06801 and 2604.08524 | The hand-written-pair route is **kept** — but relabeled as what it actually is: **construct** validity, breaking the judge-defines-and-scores circularity. It is never described as answering non-identifiability. |
| Novelty: 2606.07696, 2602.02132, 2512.16602 are uncited by all four designs | All three cited explicitly in related work, with the saving distinctions stated in the body: 2606.07696 measures whether the *steering effect survives* perturbation (never the projection, never flip prediction, never identifiability); 2602.02132 and 2512.16602 own the dissociation spine, and our honest framing is "we test whether the shared-knob result extends to a category they did not include." |
| Novelty: fractional depth is method, not contribution, and 2606.29196 predicts against it | C5 demoted from headline to descriptive result; the Qwen2.5 within-family ladder is non-optional; 2606.29196 is cited as the anchor and the caveat. |
| Reviewer: the two-stage venue plan assumes workshop feedback that arrives Sept 29, after both ICLR deadlines | **Corrected.** ICLR version written in parallel from Aug 3; September adds no new experiments. |
| All four: no mediation test anywhere; the arm is purely correlational | **Added.** Necessity + sufficiency with random and d_OOD controls, ~15 A100-hrs. This is now the headline claim and the ICLR lead. |
| All four: no OOD/"weird-input" competitor direction | **Added** (d_OOD), with a nested-model requirement in C3. |
| Reviewer: nothing controls for d_soft being a hedging-register/verbosity direction — the 2025 failure one level up | **Added** (d_length), with a TOST requirement that it *fails* to reproduce on-target steering. |
| All four: multiplicity over layer sweeps unhandled | Max-statistic permutation nulls; l\* pre-registered from an independent sweep; Holm across headline tests; BH within the ladder. |
| All four: cross-model claims reported as vote counts | Replaced by per-model CIs plus a random-effects meta-estimate, with n=4 declared descriptive. |
| All four: overclaiming against non-identifiability | The paper's sentence is: *"perturbation-induced displacement is an out-of-band constraint that eliminates directions behavioral testing of the intervention cannot distinguish, reducing but not collapsing the admissible set."* Never "breaks the equivalence class." |

**One place I overrule a critic.** The dual-use critic ranked Candidate 4 (exclusion) above weak inclusions, on the grounds that running the arm against a bad null is worse than not running it. That reasoning is correct and it is why the covariance-matched null and Gate 1.5 are non-cuttable — but it argues for *better controls*, not exclusion, and the same critic's own analysis concedes the fix is cheap and available. Exclusion also deletes the only unpublished claim on the table and, per Candidate 4's own venue plan, relocates the harmful-prompt work into a standalone archival paper that is strictly more hazardous than anything we are proposing to run. **The arm is in, with the null structure that makes it mean something and the construction rule that makes it safe.**

**The one-line summary for the lead:** we are running the current plan's dissociation as a compressed one-page setup, and spending the surplus on a second, undesigned route to the same boundary — with the geometry figure finished before the crunch week, a causal mediation test nobody else proposed as the headline, and a Gate-1.5 kill switch on Aug 13 that costs ten GPU-hours and fires before we are committed.