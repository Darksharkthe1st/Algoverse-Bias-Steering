# Exp-1 reportable numbers — judge v2.1

Companion to `summary.md`, which is what the pipeline wrote and reports the **binary
`neutrality`** judge's verdicts. That file is left exactly as the run produced it (no
hand-edited conclusions, CLAUDE.md §7). This file holds the **reportable** numbers,
under pinned judge v2.1, and they point the other way.

| | |
|---|---|
| Run | `20260926-090514_laneA_exp1_prompt_vs_steer_issuebench_qwen3-8b` |
| Model | `Qwen/Qwen3-8B` @ `b968826d9c46` |
| Vector | forced-contrast opinion direction, `(36, 4096)` fp16, md5 `dab6cecca362db8378ebd31b20f34ae0` |
| Dose | c = 8 (fixed additive, `coeff/n_layers` per layer) |
| Items | 200 IssueBench `sample`, held out from the vector's fit set, artifact templates excluded |
| Generation | greedy, `enable_thinking=False`, `max_tokens=512` |
| Judge | v2.1 9-way, `gpt-4o-mini`, T=0, seed=0, rubric SHA-256 `fc28d3128dd41ac7…` (includes the 2026-09-26 identity-decline → `non-engagement` patch) |
| Judge extraction failures | 1 of 1000, excluded from denominators and reported, never folded into a class |
| Artifacts | `judged_v2.1.csv`, `judged_v2.1.json`, `coherence.csv`, `results.csv`, `logs/by_condition/*.txt` |

## Headline: under v2.1, prompting beats the direction on the opinion pole

`--positive stance` pools `stance-factual` + `stance-evaluative` (v2.1 splits "took a
side" in two).

| comparison | steer | prompt | margin | 90% CI | verdict |
|---|---|---|---|---|---|
| **opinion pole, all items** (n=199) | 0.618 | 0.935 | **−0.317** | [−0.377, −0.261] | prompt beats steer |
| **opinion pole, coherent in both arms** (n=127) | 0.787 | 0.969 | **−0.181** | [−0.236, −0.126] | prompt beats steer |
| neutral pole, all items (n=200) | 0.445 | 0.480 | −0.035 | [−0.105, +0.035] | inconclusive |
| neutral pole, coherent in both (n=169) | 0.509 | 0.521 | −0.012 | [−0.089, +0.065] | inconclusive |

Per-item 2×2, opinion pole, coherent in both arms (n=127):

| | prompt takes a side | prompt does not |
|---|---|---|
| **steer takes a side** | 100 | **0** |
| **steer does not** | 23 | 4 |

**There is not one item where steering takes a side and prompting fails.** The
direction's successes are a strict subset of the system prompt's. On the opinion pole
the fitted direction adds nothing over asking, and the conclusion survives restricting
to coherent completions — it is not an artifact of broken text.

The neutral pole is different and worth keeping: the margin is ~0 but with **64
discordant pairs** (31 steer-only, 33 prompt-only) out of 169. There the two methods are
genuinely **complementary** — each hedges on a different third of the items — which is a
distinct finding from "they behave the same", and the aggregate margin alone cannot tell
them apart.

## Why `summary.md` says the opposite

`summary.md` (binary judge): opinion pole steer 0.890 vs prompt 0.745, Δ **+0.145**
[+0.090, +0.200] — "steer beats prompt". That reverses under v2.1 because the binary
rubric counts anything decisive as "opinionated". Of the **178** `steered_pos`
completions it called opinionated, v2.1 says:

| v2.1 label | n | share |
|---|---|---|
| stance-evaluative | 94 | 52.8% |
| **incoherent** | 32 | 18.0% |
| stance-factual | 20 | 11.2% |
| **hard-refusal** | 16 | 9.0% |
| **non-engagement** | 11 | 6.2% |
| meta-comment | 3 | 1.7% |
| unclassifiable | 1 | 0.6% |
| soft-refusal | 1 | 0.6% |

**36% of them are not a stance at all.** This is the retired judge-v1 pathology
(CLAUDE.md §4: the v1 rubric scored factual decisiveness as opinionation) reproducing
itself on new data, and it is why the two judge versions must never share a table.

## Full v2.1 distribution by arm (raw 9-way, n=200 per arm)

| label | initial | prompt_pos | prompt_neg | steered_pos | steered_neg |
|---|---|---|---|---|---|
| stance-factual | 72 | 48 | 30 | 23 | 11 |
| stance-evaluative | 81 | 139 | 48 | 100 | 21 |
| soft-refusal | 7 | 0 | 96 | 1 | 89 |
| non-engagement | 9 | 3 | 16 | 15 | 43 |
| hard-refusal | 28 | 9 | 7 | 21 | 1 |
| incoherent | 1 | 1 | 0 | **35** | **25** |
| meta-comment | 2 | 0 | 2 | 3 | 7 |
| unclassifiable | 0 | 0 | 1 | 1 | 3 |
| nonsense (judge failure) | 0 | 0 | 0 | 1 | 0 |

Two things to read off it. **`incoherent` is 35/200 and 25/200 in the steered arms
against 1/200 unsteered** — the judge's own incoherence bucket independently corroborates
the judge-independent coherence gate below, which is the agreement you want between two
measures that share no machinery. And the unsteered arm already carries 28
`hard-refusal`: these IssueBench prompts ask for advocacy essays on contested issues, so
some refusal is the model's baseline behaviour, not a steering effect.

## Coherence (Exp-3 gate, judge-independent)

`scripts/coherence_gate.py`, reference LM `Qwen/Qwen2.5-1.5B`, ppl threshold P95 of the
unsteered arm = 15.24.

| arm | coherence-pass | vs baseline |
|---|---|---|
| initial | 190/200 = 0.950 | — (calibration arm) |
| prompt_pos | 194/200 = 0.970 | PASS |
| prompt_neg | 196/200 = 0.980 | PASS |
| steered_pos | 130/200 = **0.650** | BELOW BASELINE |
| steered_neg | 172/200 = 0.860 | BELOW BASELINE |

`think_fail` is 0 in every arm: the config guardrails held, so this is steering damage
and not the misconfiguration that invalidated the dose ladder. The failures are
lexical-diversity collapse (`distinct_3`), not truncation. **c = 8 does not clear the
handoff's own reportable-dose bar**, which is why every number above is given twice.

## Decision this triggers

The handoff's decision rule: *"Prompt matches/beats steer on modern prompts → reframe to
'adaptive-linear beats **both** fixed steer and prompt' (kill-criterion)."* Prompt beats
steer on the opinion pole, with the CI clear of 0 both ways it is cut. **The
kill-criterion is met for the fixed-add arm.** The "steering adds value over prompting"
clause does not stand at c=8 on IssueBench, and the reframe is now the live option, not a
contingency.

What would change this and is not yet ruled out: a coherent dose above 8 (the ladder
that was supposed to supply one is invalid — `runs/laneA_exp3_coherence_doseladder/`),
and the adaptive-linear schedule, which is the arm the reframe would rest on and which
has not been run under these guardrails.
