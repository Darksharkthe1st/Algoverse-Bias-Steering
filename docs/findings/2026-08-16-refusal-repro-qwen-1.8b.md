# Refusal-direction repro on qwen-1.8b — first Lambda run, both tracks

**Date:** 2026-08-16 · **Model:** qwen-1.8b (Qwen/Qwen1.5-1.8B-Chat) · **Box:** Lambda A100-40GB
**Paper:** Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction* (arXiv:2406.11717)
**Guide:** [`../05-refusal-repro.md`](../05-refusal-repro.md) (load track) · [`../06-refusal-generation.md`](../06-refusal-generation.md) (generation track)

## Verdict

**Not reproduced.** Neither track meets the §4 bar. The paper's *mechanism*
reproduces cleanly — ablation collapses harmful refusal, act-add(+) induces
harmless refusal — but the un-intervened harmful baseline is 0.32 low and the
extraction cosine is 0.90 against a 0.999 target.

Both tracks being off matters: it relocates the suspect. The load track's §3
decision rule says a low generation cosine implicates the extraction recipe while
off load-track rates implicate eval formatting. Here *both* are off, so the shared
cause sits upstream of both — in which activations/instructions feed the mean —
not in eval-side formatting, which is now largely eliminated.

---

## 1. Load track

Run: `runs/20260816-011914_refusal-repro_qwen-1.8b` (direction layer=15, pos=−2, ‖r‖=26.287)

| arm | ours | paper | Δ | within ±0.05 |
|---|---|---|---|---|
| harmful/baseline | 0.380 | 0.700 | −0.320 | ✗ |
| harmful/ablation | 0.000 | 0.010 | −0.010 | ✓ |
| harmful/actadd | 0.000 | 0.030 | −0.030 | ✓ |
| harmless/baseline | 0.010 | 0.030 | −0.020 | ✓ |
| harmless/actadd | 0.950 | 0.980 | −0.030 | ✓ |

**4/5 arms in tolerance.** Ablation drives harmful refusal to 0.00 and act-add(+)
drives harmless refusal to 0.95, both as predicted. The single failing cell is the
one arm with *no intervention at all*, and it fails in the direction of our model
refusing less than theirs.

An earlier run in the same session (`runs/20260816-010402_refusal-repro_qwen-1.8b`)
used the pre-fix prompt rendering and scored harmful/baseline 0.370.

## 2. Generation track

**Extraction** (`scripts/refusal_extract_check.py --model qwen-1.8b`):

| variant | cosine @ published cell (pos −2, layer 15) | mean over grid |
|---|---|---|
| with `filter_train` (default) | **0.89988** | 0.6420 |
| `--no-filter` | 0.89165 | 0.6465 |
| float32 forward, no filter | 0.89164 | 0.6747 *(excl. layer 0)* |

Target ≥0.999 ("recipe exactly right"), ≥0.95 acceptable. **All variants land at
~0.89 — below the floor.** Train filter kept harmful=71/128, harmless=124/128.

Layer 0 is identically zero in *the paper's own* grid, so the 0.000 cosine column
is expected and is **not** a bug on our side — verified by reading the norms out of
their `mean_diffs.pt` directly.

**Selection** (`scripts/refusal_select_direction.py --model qwen-1.8b`, selecting
over *their* `mean_diffs.pt`, which isolates selection from extraction):

```
selected:  position=-1, layer=15   (refusal=-6.7426, steering=1.1752, kl=0.0670)
published: position=-2, layer=15
MISMATCH
```

**Layer exact, position off by one.** Since this ran over the paper's own grid,
the discrepancy is in the selection metric/tie-breaking, independent of extraction.

---

## 3. Causes eliminated

Scripts and raw logs are preserved under
`runs/20260816-011914_refusal-repro_qwen-1.8b/diagnostics/`.

| suspect | verdict | evidence |
|---|---|---|
| eval-side empty system turn | **ruled out** | fixed it (commit `6db750a`); moved baseline 0.370 → 0.380 |
| BOS token prepended | **ruled out** | Qwen `default_prepend_bos=False`; `to_tokens` ids == tokenizer ids, 15 tok, matches paper exactly |
| batched left-padding corrupting short rows | **ruled out** | batch=16 vs batch=1 on the same 40 prompts: identical 0.550 rate, **0/40 verdict flips** |
| `max_tokens=128` truncation | **minor, real** | truncating *the paper's own* completions to 128 tok costs 0.020 (0.700 → 0.680) |
| substring refusal judge | **ruled out** | reproduces the paper's committed labels across 2500 labels, 0 mismatches; 25 model×arm rates, 0 drift |
| `filter_train` | **ruled out** | toggling it moves the published-cell cosine by <0.01 |
| fp16 vs fp32 forward | **ruled out** | cosine 0.89165 vs 0.89164 — identical to 5 decimal places |
| position/layer axis misalignment | **ruled out** | cross-correlation: every position and every layer best-matches its own index (0/23 layers misaligned) |

### The system-turn hypothesis was wrong

`05-refusal-repro.md` §3 named the eval-side system turn as "the most likely source
of a mismatch." It was reconciled properly and **it was not the cause.**

Three renderings of the same instruction, by token count:

| rendering | tokens |
|---|---|
| old code — `build_chat_messages` with `system_prompt=""` → *empty* system turn | 20 |
| HF `apply_chat_template` with the system message dropped → injects "You are a helpful assistant." | 26 |
| **paper's literal template** (`REFUSAL_TEMPLATES`) | **15** |

Note the trap in the middle row: simply removing the system message does *not* give
the paper's prompt for Qwen — HF substitutes its own default. Eval now uses the
literal template, verified token-identical. Net effect on the failing arm: **+0.010**.

---

## 4. Strongest remaining lead

Our mean-diff vectors are consistently **0.80–0.84× the paper's norm at every
layer**, while pointing ~0.89 cosine in the same direction (pos index 3, i.e. tok −2):

| layer | ours | theirs | ratio |
|---|---|---|---|
| 1 | 0.1946 | 0.1597 | 1.219 |
| 5 | 0.7777 | 0.9293 | 0.837 |
| 10 | 4.0992 | 5.4597 | 0.751 |
| 15 | 21.9531 | 26.2873 | 0.835 |
| 20 | 52.5541 | 65.7459 | 0.799 |
| 23 | 76.3617 | 90.9334 | 0.840 |

Same direction, diluted magnitude is the signature of **averaging over a different
set of instructions** — not of a numerics, formatting, or indexing bug, all of
which are now excluded. Next checks:

1. Verify `load_and_sample_repro`'s selection against upstream's *actual* train
   set, item by item — the seeded `random.sample` chain is order-sensitive, so a
   difference in file order or call order silently changes the sample.
2. Verify the `refusal_score` filter against upstream's kept count. Keeping only
   71/128 harmful is aggressive enough to be worth confirming.

## 5. Cheap fidelity fix, not yet applied

Raise `max_tokens` 128 → 512 (the paper's value) in `configs/refusal_repro.py`.
The refusal metric matches a prefix *anywhere* in the completion, and the paper's
completions run to 512 tokens (mean 179, median 110, 36/100 exceed 128), so our cap
loses real refusals. Worth ~0.02 of the 0.32 gap.
