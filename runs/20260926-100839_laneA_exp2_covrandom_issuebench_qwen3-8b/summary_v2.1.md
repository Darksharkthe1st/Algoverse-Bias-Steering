# Exp-2 reportable numbers — covariance-matched random-direction control, judge v2.1

Companion to `summary.md` (the pipeline's own output, binary judge, left untouched).
Paired item-for-item with Exp-1
(`runs/20260926-090514_laneA_exp1_prompt_vs_steer_issuebench_qwen3-8b`): identical 200
items, identical dose, identical generation settings, and the only difference is which
vector is applied.

| | |
|---|---|
| Run | `20260926-100839_laneA_exp2_covrandom_issuebench_qwen3-8b` |
| Control vector | `runs/laneA_exp2_covrandom_vector_seed0/steering_vector.safetensors`, `(36, 4096)` fp16 |
| How built | `scripts/build_covrandom_vector.py --covariance within-pole --orthogonalise --seed 0` — per-layer draw from the within-pole residual covariance, opinion direction projected out, norm rematched per layer |
| Match quality | per-layer norms agree with the opinion vector to 5e-3 (fp16 rounding); max \|cos\| with it 1.8e-4 |
| Dose | c = 8, same as Exp-1 |
| Judge | v2.1, rubric SHA-256 `fc28d3128dd41ac7…`, 0 extraction failures in 600 |

## Result: it is the direction, not the dose

Stance rate (`stance-factual` + `stance-evaluative` pooled), each arm against **its own
run's** unsteered baseline, per-item paired, 90% CI, SE from the bootstrap SD:

| arm | vector | rate | vs own baseline | SE | verdict |
|---|---|---|---|---|---|
| `steered_pos` | **cov-random control** | 0.770 | **+0.010** [−0.025, +0.045] | 0.020 | 0.5 SE — moves nothing |
| `steered_neg` | **cov-random control** | 0.785 | **+0.025** [−0.010, +0.060] | 0.020 | 1.2 SE — moves nothing |
| `steered_pos` | opinion direction | 0.618 | −0.146 | 0.039 | 3.8 SE |
| `steered_neg` | opinion direction | 0.160 | **−0.605** | 0.036 | 16.8 SE |

Baselines: Exp-2 `initial` 0.760, Exp-1 `initial` 0.765 (same 200 items, byte-identical
completions — the 0.005 is judge drift, see below).

Direction minus control, paired on the same items:

| pole | margin | 90% CI | SE | gap |
|---|---|---|---|---|
| `steered_neg` | −0.625 | [−0.680, −0.565] | 0.0355 | **17.6 SE** |
| `steered_pos` | −0.151 | [−0.211, −0.090] | 0.0378 | **4.0 SE** |

**Exp-2 passes.** A vector with the same per-layer norm and the same residual-covariance
envelope, carrying no component along the opinion direction, moves the stance rate by
≈0 at the dose where the opinion direction moves it by 0.605. The handoff's bar (control
≈ baseline, direction exceeds it by ≥4 SE) is met with a large margin on the hedging
pole and exactly at the bar on the stance pole. "You just perturbed the residual stream
hard enough" is not what is happening.

Read the 2×2 for the hedging pole (n=200): both 30 · direction-only 2 · **control-only
127** · neither 41. On 127 of 200 items the control still takes a side and the opinion
direction does not — the effect is concentrated, not a diffuse shift in marginals.

## The sign, which matters for the paper

The opinion direction reduces stance-taking at **both** signs of the coefficient: hard
at −8 (0.765 → 0.160, as intended — that is the hedging pole) and also at +8
(0.765 → 0.618, which is *not* as intended: +8 is meant to push *toward* taking a side).
Under v2.1 the fitted direction is an effective **hedging** knob and a counterproductive
**stance** knob. That is consistent with Exp-1, where prompting beat it on the opinion
pole while the two were complementary on the neutral pole.

## Coherence: the damage is direction-specific too

`scripts/coherence_gate.py`, same reference LM and the same ppl threshold as Exp-1
(P95 of the unsteered arm = 15.24 — identical, because the unsteered completions are
byte-identical across the two runs).

| arm | Exp-2 control | Exp-1 opinion direction |
|---|---|---|
| initial | 190/200 = 0.950 | 190/200 = 0.950 |
| steered_pos | **188/200 = 0.940** | 130/200 = 0.650 |
| steered_neg | **192/200 = 0.960** | 172/200 = 0.860 |

The control, at the same norm and dose, leaves coherence at baseline. So the text
degradation Exp-3 found at c=8 is **not** "pushing the residual stream this hard breaks
generation" — it is specific to this direction. The judge's own `incoherent` count agrees:
2 of 600 completions here, against 60 of 1000 in Exp-1.

## A determinism and judge-reliability check, for free

The `initial` arm is re-run here deliberately; it should reproduce Exp-1's exactly.

- **Generation: exactly deterministic.** 0 of 200 completions differ in text.
- **The judge is not.** On those byte-identical completions the binary `neutrality`
  judge returned a **different verdict on 10 of 200 items (5%)** at `temperature=0`,
  `seed=0`, same model.

The flips nearly cancel, so the arm's *marginal* moved only 2 items (98→96 neutral) —
which is why a table of marginals understates per-item churn by ~5×, the same trap
CLAUDE.md §3 flags about the arrow-named historical columns. It is also a concrete
measurement of the drift the handoff estimates at ±1–2/100: on this data it is **5/100
per-item**, so any margin under ~5 points is inside judge noise and the item-bootstrap
CIs are not optional. (This is measured on the binary judge, because that is the one the
pipeline ran twice; v2.1's own test-retest would need a deliberate double-judge.)
