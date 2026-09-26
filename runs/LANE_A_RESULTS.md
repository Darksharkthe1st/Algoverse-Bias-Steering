# Lane A results — steering controls on Qwen3-8B

Deliverable for `docs/HANDOFF_lane_a.md`. Every number here traces to an artifact under
`runs/`; the command that produced it is given with it. Open decisions for the team are
in `runs/LANE_A_OPEN_QUESTIONS.md` — read that alongside this, because two of the three
experiments departed from the handoff's method for reasons that are themselves results.

**Headline, one line per experiment:**

| Exp | Reviewer objection it closes | Verdict |
|---|---|---|
| 1 | "a system prompt would do the same" | *(pending — run completing)* |
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

| arm | coherence-pass | `distinct_3` fails | mean `distinct_3` |
|---|---|---|---|
| initial | *(pending)* | | |
| prompt_pos | | | |
| prompt_neg | | | |
| steered_pos | | | |
| steered_neg | | | |

`think_fail` is 0 in every arm, confirming the config guardrails held: this is steering
damage, not misconfiguration. The failures are lexical-diversity collapse, not
truncation.

**Consequence for the headline.** No dose currently clears the "pass rate ≥ unsteered
baseline" bar — including c=8 — so Exp-1's comparison is reported twice: over all items
and over items that passed the gate. A judge can read a confident stance out of a
repetition loop, so those two numbers diverge exactly where the objection bites.

---

## Exp-1 — prompt vs steer on held-out IssueBench (PIVOTAL)

*(pending)*

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
