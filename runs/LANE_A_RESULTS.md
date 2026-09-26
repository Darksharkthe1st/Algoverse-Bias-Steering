# Lane A results — steering controls on Qwen3-8B

Deliverable for `docs/HANDOFF_lane_a.md`. Every number here traces to an artifact under
`runs/`; the command that produced it is given with it. Open decisions for the team are
in `runs/LANE_A_OPEN_QUESTIONS.md` — read that alongside this, because two of the three
experiments departed from the handoff's method for reasons that are themselves results.

**Headline, one line per experiment:**

| Exp | Reviewer objection it closes | Verdict |
|---|---|---|
| 1 | "a system prompt would do the same" | **It does more.** Under pinned v2.1 the system prompt BEATS the direction on the opinion pole (−0.317 all items, −0.181 on coherent items, both CIs clear of 0), with **zero** items where steering succeeds and prompting fails. The handoff's kill-criterion is met for the fixed-add arm. |
| 2 | "it's the dose, not the direction" | *(pending)* |
| 3 | "your opinion rate is just broken text" | **The objection has force at the headline dose.** The unsteered baseline is healthy (128/128 coherent) but steering at c=8 costs ~22% of coherence on the opinion pole, so the rate must be read on the coherent subset. Separately, the c20/c30/c40 ladder is **invalid** and cannot set a reportable dose. |

## Shared setup

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` @ `b968826d9c46` (the frozen submission model) |
| Direction | forced-contrast opinion vector, `runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b/steering_vector.safetensors`, `(36, 4096)` fp16, md5 `dab6cecca362db8378ebd31b20f34ae0` |
| Eval set | 200 IssueBench `sample` items, **held out** from the vector's 200 fit items, with 11 ShareGPT scraping-artifact templates excluded (`datasets/laneA/issuebench_exclusions.json`, built by `scripts/build_heldout_exclusions.py`) |
| Generation | `enable_thinking=False`, `max_tokens=512`, greedy, `batch_size=32` |
| Judge | pinned v2.1, 9-way, `gpt-4o-mini`, `temperature=0`, `seed=0`, **with the 2026-09-26 identity-decline → `non-engagement` patch**; rubric SHA-256 recorded per judged CSV |
| Coherence | `scripts/coherence_gate.py`, judge-independent, reference LM `Qwen/Qwen2.5-1.5B` |

Two guardrails were verified on the real run rather than assumed: the 200 evaluated
items have **0 overlap** with the vector's fit set and **0** rows from artifact
templates, and every applied vector is shape-asserted `(36, 4096)` before a token is
generated (CLAUDE.md §6).

### Why one seed, not three

The handoff asks for ≥3 generation seeds. The pipeline decodes **greedily**
(`do_sample=False` / `temperature=0`), so three seeds return byte-identical completions
and would manufacture a three-fold CI out of n=1. Variance is reported as an **item
bootstrap** instead. This is a real departure from the handoff — see
`LANE_A_OPEN_QUESTIONS.md` Q1.

### Judge bookkeeping

The runs carry the pipeline's binary `neutrality` verdicts inline (the eval loop's
contrast machinery requires a 2-label judge), and the reportable numbers come from
re-judging the persisted completions under v2.1 offline. The two are never mixed in one
table (CLAUDE.md §4); the binary column is kept beside the v2.1 column in
`judged_v2.1.csv` purely as a comparator. It is not a neutral one — on the pilot it
labelled a `non-engagement` response "opinionated" and folded capability refusals into
"neutral".

---

## Exp-3 — coherence gate

Two separate findings. The gate is judge-independent by construction: no leg reads a
verdict, because the judge's own `incoherent` bucket cannot both make the claim and
vouch for it.

Gate: `unclosed_think` OR `distinct_3 < 0.5` OR `max_repeat_run >= 4` OR
`ppl > P95(unsteered arm)`. Cap-truncation of the *answer* is deliberately not a
failure — IssueBench asks for essays, so every arm truncates at the cap and gating on
length could not tell that apart from a generation that died inside `<think>`.

### 3a. The c20/c30/c40 dose ladder is invalid, and the fault is its baseline

Full evidence: `runs/laneA_exp3_coherence_doseladder/` (per-completion
`coherence__*.csv`, `doseladder_table.csv`, `summary.md`).

All three runs used `max_tokens=128` with `enable_thinking` unset — **ON** for
Qwen3-8B — and `strip_reasoning` off: the §0.3 configuration. Their shared unsteered arm
is **98.5% truncated mid-`<think>`**, so 197 of its 200 labels sit on text with no
answer under it.

| dose | arm | unclosed `<think>` | coherence-pass |
|---|---|---|---|
| c20 | initial | **98.5%** | 3/200 = 0.015 |
| c20 | steered_pos | 76.0% | 48/200 = 0.240 |
| c20 | steered_neg | 100.0% | 0/200 = 0.000 |
| c30 | initial | **98.5%** | 3/200 = 0.015 |
| c30 | steered_pos | 3.0% | 193/200 = 0.965 |
| c30 | steered_neg | 68.0% | 64/200 = 0.320 |
| c40 | initial | **98.5%** | 3/200 = 0.015 |
| c40 | steered_pos | 12.5% | 135/200 = 0.675 |
| c40 | steered_neg | 0.0% | 177/200 = 0.885 |

The steered arms are *less* truncated than the baseline, because steering shortens the
output enough to close the trace. So on this ladder a verdict shift from baseline to
steered conflates "took a side" with "produced an answer at all". **c30's 138/0
good/bad — the number the handoff asks to confirm as genuinely coherent — is not
confirmable from these runs.** It needs re-generating with thinking off and ≥512
tokens (`LANE_A_OPEN_QUESTIONS.md` Q3).

It also makes the handoff's reportable-dose rule vacuous: at a 1.5% baseline every arm
clears "≥ baseline". The rule needs an absolute floor (Q4).

### 3b. At the headline dose c=8, steering degrades text

Measured on Exp-1's own completions — where the baseline *is* healthy, which is what
makes the comparison meaningful.

Reference LM `Qwen/Qwen2.5-1.5B`; ppl threshold = P95 of the unsteered arm = 15.24.
Artifact: `runs/20260926-090514_laneA_exp1_.../coherence.csv`.

| arm | coherence-pass | vs baseline |
|---|---|---|
| initial | 190/200 = 0.950 | — (calibration arm) |
| prompt_pos | 194/200 = 0.970 | PASS |
| prompt_neg | 196/200 = 0.980 | PASS |
| steered_pos | 130/200 = **0.650** | BELOW BASELINE |
| steered_neg | 172/200 = 0.860 | BELOW BASELINE |

The judge's *own* `incoherent` bucket agrees, from entirely separate machinery: 35/200
and 25/200 in the steered arms against 1/200 unsteered.

`think_fail` is 0 in every arm, confirming the config guardrails held: this is steering
damage, not misconfiguration. The failures are lexical-diversity collapse, not
truncation.

**Consequence for the headline.** No dose currently clears the "pass rate ≥ unsteered
baseline" bar — including c=8 — so Exp-1's comparison is reported twice: over all items
and over items that passed the gate. A judge can read a confident stance out of a
repetition loop, so those two numbers diverge exactly where the objection bites.

---

## Exp-1 — prompt vs steer on held-out IssueBench (PIVOTAL)

Run `runs/20260926-090514_laneA_exp1_prompt_vs_steer_issuebench_qwen3-8b` (200 items ×
5 arms, 1h02m on one A100). Full write-up, including the per-arm 9-way distribution:
that run's `summary_v2.1.md`. `summary.md` in the same folder is the pipeline's own
output under the binary judge and is left untouched.

    python -m src.bias_steer run configs/laneA_exp1_prompt_vs_steer_issuebench.py
    python scripts/rejudge_v21.py runs/20260926-090514_laneA_exp1_*
    python scripts/coherence_gate.py runs/20260926-090514_laneA_exp1_*
    python scripts/bootstrap_ci.py runs/20260926-090514_laneA_exp1_*/judged_v2.1.csv \
        --positive stance --coherence runs/20260926-090514_laneA_exp1_*/coherence.csv \
        --paired steered_pos,prompt_pos

### The result

| comparison | steer | prompt | margin | 90% CI | verdict |
|---|---|---|---|---|---|
| **opinion pole, all items** (n=199) | 0.618 | 0.935 | **−0.317** | [−0.377, −0.261] | prompt beats steer |
| **opinion pole, coherent in both** (n=127) | 0.787 | 0.969 | **−0.181** | [−0.236, −0.126] | prompt beats steer |
| neutral pole, all items (n=200) | 0.445 | 0.480 | −0.035 | [−0.105, +0.035] | inconclusive |
| neutral pole, coherent in both (n=169) | 0.509 | 0.521 | −0.012 | [−0.089, +0.065] | inconclusive |

Per-item 2×2 on the opinion pole, coherent in both arms (n=127): both 100 · **steer-only
0** · prompt-only 23 · neither 4.

**There is no item where the direction takes a side and the system prompt fails.** The
direction's successes are a strict subset of prompting's, and that holds whether or not
incoherent completions are excluded — so it is not an artifact of broken text.

The neutral pole is the one place the direction earns something: the margin is ~0 but
**64 of 169 pairs are discordant** (31 steer-only, 33 prompt-only). There the methods are
**complementary**, each hedging on a different third of items. That is a different
finding from "equivalent", and only the 2×2 shows it.

### Why this reverses the pipeline's own summary

`summary.md`, under the binary `neutrality` judge, reports steer 0.890 vs prompt 0.745,
Δ **+0.145** [+0.090, +0.200] — "steer beats prompt". Of the 178 `steered_pos`
completions that judge called opinionated, v2.1 finds **36% are not a stance**: 18.0%
`incoherent`, 9.0% `hard-refusal`, 6.2% `non-engagement`, 1.7% `meta-comment`.

That is the retired judge-v1 pathology (CLAUDE.md §4 — the v1 rubric scored factual
decisiveness as opinionation) reproducing on new data. It is also the concrete reason the
two versions may never share a table: on this run the choice of judge does not move the
number, it moves the **sign**.

### The decision it triggers

The handoff's rule: *"Prompt matches/beats steer on modern prompts → reframe to
'adaptive-linear beats **both** fixed steer and prompt' (kill-criterion)."* **Met, for
the fixed-add arm.** The "steering adds value over prompting" clause does not stand at
c=8 on IssueBench.

Two things could still change it and are not ruled out: a coherent dose above 8 (the
ladder meant to supply one is invalid, §3a), and the adaptive-linear schedule — the arm
the reframe would rest on, which has not been run under these guardrails. Neither is a
reason to soften this result (CLAUDE.md §6: honest negatives stay honest); both are the
next experiments.

---

## Exp-2 — covariance-matched random-direction control

*(pending)*

### The control, and why it is not the control the handoff specified

A vector matched on per-layer norm and residual covariance comes out **45% aligned with
the opinion direction** (mean |cos| 0.4522, max 0.9686, against ~0.05 for a random draw
in the rank-398 residual subspace). Both estimators do this — pooled 0.4502, within-pole
0.4522 — so it is not a centring error. The cause is that at the middle layers the
residual covariance is nearly **rank one** and the opinion direction lies along that
component (|cos(v, PC1)| = 0.885 at layer 12, where PC1 holds 95.2% of the variance).

Where a representation is that low-rank, "matched on the covariance" and "a different
direction" are in tension: almost every vector the covariance can produce is the
opinion direction again. So the reported control additionally **projects the opinion
direction out** per layer and rematches the norm — same envelope, same per-layer norm,
realised |cos| 0.0000. The non-orthogonalised draw is retained as the evidence for why.

This is a departure from the handoff and a claim-limiting fact in its own right: the
stronger control ("a random direction of the same norm and covariance") is **not
available** at this site, and the paper should say so rather than imply it was run. It
sits close to the non-identifiability caveat the contract already requires
(arXiv:2602.06801 — "**a** direction"). See `LANE_A_OPEN_QUESTIONS.md` Q2.

---

## What was not done

- **7-model fixed-add re-judge** (Lane B item 1). Feasible but blocked on a
  contamination that must be fixed first: the archived steered arms embed the chat
  template for the three Qwen1.5 models (99/99 each) and for none of the other four, so
  judging them as-is confounds model with judge input. Also a denominator question (99
  records, 97 unique prompts, CLAUDE.md says n=96). Details: `LANE_A_OPEN_QUESTIONS.md` Q7.
- **Judge↔human κ** (Lane B item 3). Blocked on Lane C: `fvoa_labelling.xlsx` is not on
  this box. `scripts/kappa_from_csv.py` is in place for when it lands.
- **Dose-ladder re-run**, a second non-Qwen model, the transfer arm, and any writing —
  all outside this session's budget, and the last of these outside its remit.
