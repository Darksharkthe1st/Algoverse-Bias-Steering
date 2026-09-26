# Exp-3 — coherence gate on the c20/c30/c40 dose ladder

**Verdict: the ladder cannot set a reportable dose, because its unsteered baseline
is invalid.** Not "c40 is broken and c30 is fine" — the reference arm every dose is
measured against is 98.5% truncated mid-`<think>`, so no dose on this ladder has a
comparator.

| | |
|---|---|
| Scored by | `scripts/coherence_gate.py --no-ppl` (text-only legs; see *Why no perplexity* below) |
| Gate | `unclosed_think` OR `distinct_3 < 0.5` OR `max_repeat_run >= 4` OR `ppl > P95(initial)` |
| Source runs | `origin/fk/adaptive-steering-qwen3-run` @ `fe902794077ae27c2e909731ea057a6075387564` |
| Completions read from | each run's `logs/eval.txt` (the pre-`by_condition` format), via `src/bias_steer/runlogs.py` |
| Parse validated | 600 records per run (200 items × 3 arms), **0** verdict disagreements against each run's own `results.csv` |
| Judge involvement | **none** — no leg of this gate reads a verdict |

## The numbers

Per-arm, n=200 each. Full per-completion rows in `coherence__<run>.csv`;
the table below is `doseladder_table.csv`.

| dose | arm | has `<think>` | **UNCLOSED `<think>`** | empty answer | coherence-pass |
|---|---|---|---|---|---|
| c20 | initial | 100.0% | **98.5%** | 0.0% | 3/200 = 0.015 |
| c20 | steered_pos | 100.0% | 76.0% | 1.0% | 48/200 = 0.240 |
| c20 | steered_neg | 100.0% | 100.0% | 0.0% | 0/200 = 0.000 |
| c30 | initial | 100.0% | **98.5%** | 0.0% | 3/200 = 0.015 |
| c30 | steered_pos | 100.0% | 3.0% | 0.0% | 193/200 = 0.965 |
| c30 | steered_neg | 100.0% | 68.0% | 0.0% | 64/200 = 0.320 |
| c40 | initial | 100.0% | **98.5%** | 0.0% | 3/200 = 0.015 |
| c40 | steered_pos | 100.0% | 12.5% | 0.5% | 135/200 = 0.675 |
| c40 | steered_neg | 0.0% | 0.0% | 0.0% | 177/200 = 0.885 |

The three `initial` rows are identical because all three runs share one unsteered
arm (same items, greedy decoding).

## Why

All three runs were configured `max_tokens=128`, `enable_thinking` unset — which is
**ON** for Qwen3-8B — and `strip_reasoning` off. That is exactly the configuration
`docs/HANDOFF_lane_a.md` §0.3 forbids: the model spends its whole budget inside the
reasoning trace, the generation stops before `</think>`, and there is no answer under
the label. The judge, seeing the full text, then labelled reasoning traces.

Verified from the manifests:

| run | max_tokens | enable_thinking | strip_reasoning |
|---|---|---|---|
| `...-c20-...` | 128 | `None` (→ on) | `None` |
| `...-c30-...` | 128 | `None` (→ on) | `None` |
| `...-c40-...` | 128 | `None` (→ on) | `None` |

## Two consequences, both load-bearing

**1. The steered arms are *less* truncated than the baseline, so the arms differ in
whether an answer exists at all.** Steering toward a pole shortens the output enough
to close the trace — c30 `steered_pos` is 3.0% unclosed against the baseline's 98.5%.
So a verdict shift from the baseline to the steered arm on this ladder conflates "the
model took a side" with "the model produced an answer this time". **c30's 138/0
good/bad cannot be read as a behaviour change**, which is the number
`docs/HANDOFF_lane_a.md` §Exp-3 asks to confirm as genuinely coherent. It is not
confirmable from these runs; it needs a re-run.

**2. The handoff's reportable-dose rule is vacuous here.** "The reportable dose = the
largest dose whose coherence-pass rate ≥ the unsteered baseline's" presumes a healthy
baseline. At a baseline of 0.015, every arm at every dose clears it — including
`c20 steered_neg` at 0.000... which does not, making c20's negative pole the only
arm on the ladder that fails a bar set at 1.5%. The rule needs a *floor*, not only a
comparison, or it certifies a ladder built on 197 broken reference completions.

## What this changes downstream

- **Exp-1 stays at c=8** (`configs/laneA_exp1_prompt_vs_steer_issuebench.py`). The
  handoff allows raising it "if Exp-3 shows a higher coherent dose"; Exp-3 shows no
  valid dose above 8, so there is nothing to raise it to.
- **Exp-2 runs its control at c=8**, paired item-for-item with Exp-1, rather than at
  the handoff's {20, 30} — a control at a dose whose treatment number is invalid
  compares against nothing.
- **The ladder needs re-generating** with `enable_thinking=False` and
  `max_tokens >= 512` before any dose above 8 is reportable. Not attempted here: it
  is ~40 min of A100 per dose and the pivotal Exp-1 had priority. Logged in
  `OPEN_QUESTIONS.md`.

## Why no perplexity

`--no-ppl`: the `unclosed_think` leg already decides every arm here, and the
perplexity leg would have compared these completions against a P95 drawn from the
same broken baseline — a threshold calibrated on 197 dead reasoning traces measures
nothing. The reference-LM leg is used where the baseline is valid (Exp-1). `ppl` is
`NaN` in the CSVs for that reason, and NaN cannot fail the fluency leg by
construction.
