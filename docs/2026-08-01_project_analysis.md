# Algoverse Mech Interp — Project Post-Mortem & Frontier Analysis
*Compiled 2026-08-01. Sources: repo forensics (602 commits, all branches), the Lovkush-Blondin paper outline (Downloads), and three parallel literature scans (steering frontier, latent-space geometry, bias/neutrality niche) covering 2025–2026 publications.*

---

## Part 1 — What the project actually was

**Repo:** github.com/oninvis/Algoverse_Mech_Interp · Algoverse Summer 2025 · README title: *"Revealing hidden biases by finding steering vectors for neutrality."*

**Thesis (never written down as prose, reconstructed from code + outline):** an LLM's tendency to be neutral vs. opinionated on controversial-but-not-harmful prompts ("soft refusal," as distinct from Arditi-style hard refusal of harmful requests) is mediated by a linear direction in the residual stream. Steering against it should surface suppressed opinions ("hidden biases"); steering toward it should enforce neutrality.

**Method:** a direct port of Arditi et al. 2024 (arXiv:2406.11717):
1. Generate model responses to synthetic forced-choice prompts ("Which is more creative: friends or dreams?").
2. GPT-4o-mini judges each response `neutral` / `opinionated` / `nonsense` (earlier: Gemini, gpt-3.5).
3. Cache `hook_resid_pre` per layer, mean-pool over tokens; vector = mean(opinionated resids) − mean(neutral resids). **Self-labeled contrast set** — the model's own outputs sorted by judge, not hand-written pairs.
4. Steer by adding `(coeff / n_layers) · vec[layer]` at every layer, every position, via TransformerLens hooks.

**Models:** Qwen1.5-{1.8B,7B,14B}-Chat, Yi-6B-Chat, gemma-{2b,7b}-it, Llama-2-{7b,13b}-chat, Llama-3-8B-Instruct (+ an abandoned Qwen2.5 0.5B→32B scaling ladder). **Datasets:** ~1,300 synthetic GPT-generated comparison prompts (primary), BBQ (10 categories), CrowS-Pairs, Do-Not-Answer (refusal thread), Political Compass prompts (unused).

**Contributors:** 6 nominal; ~95% of commits and all late-stage work by Farhan Kittur. Timeline: Jul 17 – Dec 14, 2025 (Oct = 386 commits, the crunch). No paper draft, .tex, or docs ever existed in the repo — the Overleaf outline in Downloads is the only writing.

### What worked
- **In-distribution bidirectional control is real and large.** Final `main` run (Batched_Gen.csv, 2025-10-30), ~100 test prompts per cell:

  | Model | Baseline Opin/Neut | →Opinion | →Neutral |
  |---|---|---|---|
  | Qwen1.5-1.8B | 30/66 | 75/21 | 2/94 |
  | Qwen1.5-7B | 48/48 | 78/18 | 3/93 |
  | Qwen1.5-14B | 62/34 | 86/10 | 17/79 |
  | gemma-2b-it | 82/14 | 96/0 | 21/75 |
  | Llama-3-8B-it | 6/90 | 24/72 | 12/84 |

  Near-zero `nonsense` — coherence preserved. Counts are out of **n = 96** per arm (not ~100), and the arrow-named CSV columns are **per-arm label marginals, not transitions** — `Init->Opin` means "the initial arm was judged opinionated." Verified to reproduce 7/7 from per-record artifacts (`scripts/verify_2025_results.py`).
  ⚠️ The zero-vector ablation "collapses to 99% nonsense" claim is **under review**: the audit found 2,032 case-insensitive `none` markers across 107 archived logs that are *judge-extraction failures*, not degeneration. Do not cite the ablation as a control until the `none` markers in that specific run are separated from genuine incoherence.
- Honest experimental hygiene by student standards: coefficient sweeps per model, cross-dataset transfer matrices, failure-labeled directories (`failed_opinion_refusal/`, `bad_bbq_tests/`).

### What failed (the three load-bearing failures)
1. **Transfer failure.** Vectors trained on synthetic comparison prompts largely stop working on CrowS-Pairs (gemma-7b-it: 78/18 → 77/19 steered toward neutral — no effect; Llama-2-7b likewise) and only semi-work on BBQ. The "revealing hidden biases" premise never survived contact with real bias benchmarks.
2. **Ablation failure.** Directional ablation didn't produce neutrality; the mentor-doc hypothesis was that neutrality and opinionation are *separate directions*, not one bipolar axis. Never tested.
3. **Refusal entanglement — RETRACTED 2026-08-07. The experiment was invalid, not null.**
   ~~Cross-applying Arditi's refusal vector and the opinion vector failed in both directions (Yi-6B: 7/92 → 3/94 — nothing).~~
   Two independently confirmed defects in Logs 210–214 (see `docs/REVIVAL_AUDIT.md`, `docs/VERIFICATION_2026-08-07.md`): **(a)** the model loop and the `vector_files` list were ordered differently, so every run loaded a *different model's* vector — payload SHA-256 matching recovers the exact rotation; **(b)** the archived refusal `.pt` files are **1-D tensors of hidden width** (`Qwen-1.5-1.8B: (2048,)`, `llama-2-7b: (4096,)`), yet the steering code indexes `steering_vector[layer]`, which on a 1-D tensor returns a **scalar** broadcast across the entire residual width — a DC offset, not a direction. The opinion vectors are `[n_layers, d_model]`, so the same line correctly yields a direction; that is why the headline opinion result is sound and this arm is not.
   **The soft-vs-hard-refusal relationship is therefore UNTESTED, not tested-and-null.** Anything downstream that treated this as evidence of separability (or of entanglement) must be re-derived.

**Construct validity flaw underneath all three:** the judge prompt scored anything taking "a clear stance, even if factual" as opinionated — conflating *decisiveness* with *bias*. The vector plausibly encodes **hedging style**, not bias suppression, which predicts exactly the observed transfer failure. Nobody tested this alternative hypothesis.

### Where it died
`main` ends Nov 5. The real tip is unmerged branch `farhan-synthetic-steering`: a December pivot to force-feeding synthetic ChatGPT-written neutral outputs and caching their residuals. Last commit, **Dec 14, 2025: "Patch up bugs in code, still unsuccessful synthetic steering."** The final logs are degenerate (everything collapses to neutral regardless of steering direction). The repo went quiet mid-pivot.

---

## Part 2 — Where the frontier moved (mid-2026)

### A. Activation steering: the 2024 recipe is now a baseline, not a method

- **AxBench** (Wu et al., ICML 2025, arXiv:2501.17148): prompting and finetuning beat every representation-level steering method head-to-head; difference-in-means is the best *cheap* intervention; SAE features near the bottom. Prompting is now the mandatory baseline.
- **Reliability literature** (Tan et al. NeurIPS 2024 arXiv:2407.12404; Da Silva et al. ACL 2025 arXiv:2504.04635 — 36 models, 14 families): huge per-example variance, many models show zero effect, single-family validation is disqualifying. 2026 adds *pre-intervention predictors* of steerability (arXiv:2602.17881, 2604.15557) and **non-identifiability** (arXiv:2602.06801): a vector that steers is not evidence you found "the" representation.
- **Recipe upgrades:** conditional activation steering (CAST, IBM, ICLR 2025 spotlight — steer only when a condition vector fires); affine concept editing (ACE, arXiv:2411.09003 — projection + addition with reference point); learned rank-1 interventions (ReFT-r1); hypernetwork-generated vectors matching prompting on AxBench (HyperSteer, arXiv:2506.03292); activation **capping** (clamp projection at 95th percentile) instead of constant addition.
- **Side-effect literature:** steering toward narrow behaviors can induce broad emergent misalignment (arXiv:2606.08682), silently weaken safety (arXiv:2603.24543), and removing the refusal direction shifts unrelated dispositions — abliterated models become measurably more optimistic and hedge less (Fafuła, arXiv:2607.17427, preregistered).
- **Production reality:** Anthropic industrialized the exact pipeline as **persona vectors** (arXiv:2507.21509 — automated trait→contrastive-prompts→DiM-vector, used for monitoring, preventative steering during finetuning, and data flagging) and the **Assistant Axis** (arXiv:2601.10387, deployed-grade activation capping). Google ships activation-based probes (Gemini probes, AMS scanner). Goodfire deprecated its self-serve steering API (Feb 2026) — monitoring with directions found product-market fit; consumer steering didn't.

### B. Latent-space geometry: the field went exactly where Edward's instinct pointed

- **Consensus shift: linear directions are local charts on curved, low-dimensional feature manifolds.** Anthropic's "When Models Manipulate Manifolds" (Gurnee et al., Jan 2026, arXiv:2601.04480): character-count lives on a helical manifold; SAE latents are place-cell-like discretizations of continua. Theory: cosine similarity ≈ on-manifold geodesic distance (arXiv:2505.18235).
- **Manifold steering** (Wurgaft et al., May 2026, arXiv:2605.05115 — the field's flagship this year): activation-manifold and behavior-manifold are approximately isometric; steering *along* the manifold beats linear steering (~2.8× lower intervention energy). This is the causal license for geometry-aware steering.
- **Refusal graduated from direction → cone → subspaces:** concept cones (Wollschläger et al., ICML 2025, arXiv:2502.17420); eleven distinct non-compliance directions that collapse onto one shared behavioral knob (QCRI, Feb 2026, arXiv:2602.02132); refusal-as-trajectory in reasoning models (CoT breaks static steering, arXiv:2605.26772).
- **SAE status: alive as infrastructure, demoted as ontology.** Gemma Scope 2 (64M+ latents), but GDM publicly pivoted from SAE basic science to probes/model biology after negative downstream results; SAEs provably "dilute" manifolds across redundant latents (arXiv:2604.28119) with geometry-walled scaling laws (arXiv:2605.09887). Circuit workhorse → transcoders/attribution graphs (open-sourced, Neuronpedia-hosted). Research bets → parameter decomposition (SPD, Goodfire) and OpenAI's weight-sparse transformers. Strongest SAE-alternative for steering: Mixture-of-Factor-Analyzers regions ("From Directions to Regions," arXiv:2602.02464 — ~96% interpretable features vs ~29% SAE, beats SAEs on steering).
- **Universality became operational:** vec2vec unpaired embedding translation (NeurIPS 2025, arXiv:2505.12540), linear-map alignability (mini-vec2vec), cross-architecture crosscoder diffing (arXiv:2602.11729), "Polymorphism Is Rotation" (arXiv:2605.24577). Community rallying doc: "The Future of Interpretability is Geometric" (LessWrong, Oct 2025).

### C. The bias/neutrality niche: crowded but the exact claim is unclaimed

- **Closest competitor:** *Refusal Steering* (Multiverse Computing, Dec 2025, arXiv:2512.16602) — DiM-family political-refusal removal on Qwen3-Next-80B (92%→24% political refusals, 99% JailbreakBench safety retained). Frames it as censorship, not opinionation; finds the signal *distributed*, not single-direction.
- **Political neutrality steering exists:** Nadeem et al. (arXiv:2508.08846) steer toward PCT-axis neutrality; multilingual extension CLAS (arXiv:2601.23001, 50 countries — ideology directions don't align across languages by default); continuous censorship dial (Cyberey & Evans, COLM 2025, arXiv:2504.17130).
- **The labs professionalized measurement:** Anthropic's open-sourced Paired Prompts even-handedness eval (Nov 2025 — Claude ~95%, and the stated ideal is *symmetric engagement, not refusal*, reframing soft refusal as a failure mode); OpenAI's five-axis political-bias framework with **"political refusal" as a named axis** (Oct 2025); CAS's consistency-training paper penalizes **"fence-sitting"** by name (arXiv:2605.22771).
- **Benchmarks moved:** IssueBench (TACL 2026, arXiv:2502.08395, 2.49M prompts, 212 issues) is the standard now; CrowS-Pairs is widely criticized; reviewers will ask why you used BBQ.
- **The term "soft refusal" is unclaimed as of Aug 2026.** No paper names an activation direction for it. But QCRI's one-knob finding, Multiverse's distributed-signal finding, and Fafuła's off-target disposition shifts all circle it from different sides.

---

## Part 3 — Gap analysis: the project's failures are now the frontier's questions

| 2025 failure | 2026 frontier answer | Revival experiment |
|---|---|---|
| Transfer failure (synthetic → BBQ/CrowS) | Steering vectors are brittle OOD (Tan et al.); linear accessibility predicts where they work; hedging-vs-bias construct conflation | Re-judge with a decisiveness-vs-bias-separated rubric; evaluate on IssueBench + Paired Prompts; predict transfer with geometric accessibility scores |
| Ablation failure ("neutrality and opinionation are separate directions") | Refusal is a cone/subspace, not a line; affine (ACE) decomposition explains why pure ablation fails | Extract the opinionation *subspace* (RFM-AGOP or MFA regions), test cone structure, apply ACE instead of raw ablation |
| Refusal entanglement (cross-application failed) | QCRI: 11 refusal flavors, distinct geometry, one shared behavioral knob; Fafuła: harm-refusal surgery shifts opinionation as a side effect | **The double dissociation** — the open question nobody has run |
| Per-model coefficient chaos | Activation capping (percentile clamp) replaces constant coefficients; conditional gating (CAST) replaces steer-everywhere | Cap the opinionation projection instead of sweeping coefficients |
| Judge = whole methodology | LLM-judge means hide bimodal effects; side-effect audits now expected | Per-example distributions, capability + safety off-target audits (MMLU, XSTest, JailbreakBench) |

## The 2026-grade revival (in descending order of value)

1. **The factorization claim (unclaimed):** cleanly separate *whether the model takes a side* (soft-refusal scalar) from *which side it takes* (ideology direction, per Nadeem et al.), and show you can move one without the other. This is the project's actual latent claim, still available, and now measurable with public lab evals.
2. **The double dissociation (requested by the literature):** extract the opinionation direction and Arditi's harm-refusal direction in the same models; report geometry (cosine, cone membership per Wollschläger, shared-latent-core per QCRI); demonstrate steering opinionation moves IssueBench/Paired-Prompts behavior with zero movement on JailbreakBench/XSTest, and vice versa. **Either outcome is publishable** — independence contradicts QCRI's one-knob finding; entanglement mechanistically explains Fafuła's off-target effects (his paper explicitly requests this).
3. **Geometry upgrade:** treat neutrality/opinionation as a region/manifold question (conceptors, MFA regions, manifold steering) rather than a direction — directly engages the 2026 methods wave and Edward's original instinct.
4. **Method hygiene floor (mandatory for any venue):** system-prompt baseline (AxBench), ≥3 model families, per-example distributions, affine/capped interventions, off-target audits, modern benchmarks.

**Branding note:** "soft refusal" is good, available terminology — OpenAI's "political refusal" axis and CAS's "fence-sitting" show the labs want a name for exactly this behavior.

---

## Reading list (priority order)

**Must-read core:**
1. QCRI, "There Is More to Refusal in LLMs than a Single Direction" — arXiv:2602.02132 (the direct threat/opportunity for the thesis)
2. Wollschläger et al., "The Geometry of Refusal: Concept Cones" — arXiv:2502.17420 (ICML 2025)
3. Wu et al., "AxBench" — arXiv:2501.17148 (the evaluation bar)
4. Multiverse Computing, "Refusal Steering" — arXiv:2512.16602 (closest competitor)
5. Wurgaft et al., "Manifold Steering" — arXiv:2605.05115 (geometry-aware steering license)

**Second ring:**
6. Anthropic, "Persona Vectors" — arXiv:2507.21509 · 7. Anthropic, "The Assistant Axis" — arXiv:2601.10387 · 8. Gurnee et al., "When Models Manipulate Manifolds" — arXiv:2601.04480 · 9. Fafuła, "Abliteration Is Not a Scalpel" — arXiv:2607.17427 · 10. Lee et al., "CAST" — arXiv:2409.05907 (ICLR 2025) · 11. Marshall et al., "Refusal is an Affine Function" — arXiv:2411.09003 · 12. Röttger et al., "IssueBench" — arXiv:2502.08395 · 13. Nadeem et al., "Steering Towards Fairness" — arXiv:2508.08846 · 14. Shafran et al., "From Directions to Regions" — arXiv:2602.02464 · 15. Tan et al., "Generalization and Reliability of Steering Vectors" — arXiv:2407.12404

**Evals/infra:** Anthropic political-neutrality-eval (github.com/anthropics/political-neutrality-eval) · IBM activation-steering library · EasyEdit2 · Neuronpedia (Gemma Scope 2, circuit tracer, assistant-axis demo)
