# Intervention classes — reverse-engineered map

*Owner for **intervention / control-surface taxonomy** (what we *do* to the
model). Response labels live only in `docs/RUBRIC_v2.md`. Street landscape
context: `experiments/community_methods/LANDSCAPE_THREADS.md`. Toolkit
adoption priority: `docs/STEERING_TOOLKIT_2026.md`.*

Written 2026-08-10 from (a) our 2025 pipeline, (b) 2026 literature already
cited in-repo, (c) BASI harvest as *naming* signal only.

## Why this file exists

People say “steering,” “jailbreak,” “abliteration,” and “bias control” as if
they were one dial. They are different **control surfaces**. Mixing them
produces the same bug class as mixing soft and hard refusal in one label:
one word, several mechanisms, uninterpretable tables.

## The surfaces (coarse → fine)

| ID | Surface | When it acts | Typical street name | Typical lab name | In our runner? |
|---|---|---|---|---|---|
| **S0** | Sampling / decoding only | generate() | temperature tricks | — | baseline always |
| **S1** | Single-turn prompt | before tokens | DAN, roleplay, “JB prompt” | system/user prompt baseline | **AxBench bar — required** |
| **S2** | Multi-turn context | across messages | crescendo, many-shot | multi-turn redteam | not this sprint (thread) |
| **S3** | Agent / tool scaffold | outside pure LM | harness, subagents, MCP | agent safety | not this sprint (thread) |
| **S4** | Residual activation (runtime) | each forward pass | “steering vector,” RepE/CAA (when real) | activation steering | **yes — core** |
| **S5** | Weight edit | offline, permanent | abliteration, heretic, “uncensored GGUF” | refusal orthogonalization / machine unlearning-ish | compare-only later |
| **S6** | External filter | I/O classifier | OpenAI moderation, llama-guard | safety stack | env, not method |

Our paper’s causal claims live in **S4** (plus **S1** as mandatory baseline).
The ecosystem’s hard-refusal product default is increasingly **S5**. Soft
refusal as a *civic product issue* is mostly **S1 policy + S4 probes** at labs.

## S4 subtypes (what we implement or cite)

| ID | Mechanism | Op on residual `h` (sketch) | Fixes / fails | Status for us |
|---|---|---|---|---|
| **S4-add** | Constant add | `h ← h + α d` (maybe all layers, α/L) | Bidirectional if ±; coefficient chaos; steer-everywhere | **Legacy 2025** (`add_constant`) |
| **S4-ablate** | Project out | `h ← h − (h·d̂)d̂` | Works for one-sided concepts; **fails** when both poles are meaningful (our ablation story) | Honest negative; don’t lead |
| **S4-ace** | Affine / ref-point edit | project-out + add toward target ref (ACE) | Better for affine concepts; candidate for bipolar opinionation | **Tier 1 implement** |
| **S4-cap** | Clamp projection | limit `h·d̂` to percentile band | Side-effect / oversteer control; Assistant Axis style | **Tier 1 implement** |
| **S4-cast** | Conditional gate | apply S4-* only if condition dir fires | Contested prompts only | Tier 1 optional |
| **S4-sae** | Feature basis | add/suppress SAE latents | Rare on BASI; Qwen-Scope available | Tier 2 |
| **S4-manifold** | Along manifold | not pure linear d | 2026 flagship; expensive | Parked geometry program |

### Shape hygiene (all S4)

Direction tensor must be `(n_layers, d_model)` or an explicit per-layer
`(d_model,)` with layer index — never a 1-D hidden vector indexed by layer
(silent DC offset). See `docs/REVIVAL_AUDIT.md`.

## S5 subtypes (street-default hard-refusal removal)

| ID | Mechanism | Relation to S4 | Notes |
|---|---|---|---|
| **S5-ablit** | Estimate refusal direction from harmful vs benign (or similar), **orthogonalize weights** so that direction is suppressed | Same *idea* as S4-ablate, different *locus* (weights vs activations) | Heretic / classic abliteration blogs; huihui-style GGUFs |
| **S5-ft** | Finetune / merge “uncensored” | Not a direction | Different beast; don’t call it steering |

**Reverse-engineering claim (hypothesis, not measured here):**  
S5-ablit and S4-ablate target **hard refusal**. Side effects on hedging /
optimism / soft refusal are expected (Fafuła). Our contribution is to
**measure T1 labels** under S4-add/cap/ace *and optionally one S5 checkpoint*
with the same batteries — not to ship S5.

## Crosswalk: BASI language → class

| If someone says… | Class | Our response |
|---|---|---|
| “use this JB prompt” | S1 | Prompt baseline only; no payload rehost |
| “talk it into it over many turns” | S2 | Landscape thread |
| “spin subagents / long harness” | S3 | Landscape thread |
| “add the refusal vector” / “RepE” | S4-add (if real) | Our lineage |
| “ablate refusal” | S4-ablate or S5-ablit | Disambiguate locus |
| “heretic / abliterated GGUF” | S5-ablit product | Related work + side-effect arm |
| “model won’t pick a side” | **not an intervention** — it’s a **T1 behavior** | Rubric: soft refusal |

## Crosswalk: our 2025 code → class

| Code path | Class |
|---|---|
| `get_opinion_vec_from_resids` (DiM neutral vs opinion) | Extract d for S4 |
| `value += (coeff/n_layers) * v` all layers | **S4-add** |
| `flip_steering` | S4-add with α → −α |
| Directional ablation experiments | S4-ablate (failed for neutrality) |
| Synthetic residual forcing branch | failed S4 extraction variant |

## What to implement next (unchanged plan, sharper names)

```
config.intervention: s1_system_prompt | s4_add | s4_cap | s4_ace | [s4_cast]
config.extraction:   dim_opinion | dim_harm | hand_written_pairs | …
```

Report every cell with `(intervention_id, judge_version, model_id)`.

## Threads (mechanism)

1. Spec ACE formula from Marshall et al. against our bipolar opinionation
   hypothesis (why S4-ablate failed).  
2. Spec cap as percentile of clean-run projections on d (Assistant Axis).  
3. One-page diff: S5-ablit (weight) vs S4-ablate (activation) — same linear
   intuition, different persistence and side-effect profile.  
4. Optional GPU: one public abliterated checkpoint × T1 rubric × safety battery.

## Non-goals

- Attack success rates on harmful prompts as a project KPI  
- Rehosting jailbreak payloads  
- Claiming S5 is “the soft-refusal method”
