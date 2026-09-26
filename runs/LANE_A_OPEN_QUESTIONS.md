# Lane A — open questions for the team

`docs/HANDOFF_lane_a.md`: *"If a step needs a decision not covered here, stop and
write the question into `runs/<id>/OPEN_QUESTIONS.md` rather than guessing."* These
span the three experiments rather than sitting inside one run, so they are collected
here, next to `LANE_A_RESULTS.md`.

Each entry says what I did in the meantime, so nothing is blocked waiting on an
answer — but a different answer changes a reported number in the way stated.

---

## Q1 — "≥3 seeds" is not achievable as written: the pipeline decodes greedily

**The problem.** The handoff requires ≥3 generation seeds on every reported rate.
This pipeline generates greedily — `models.generate` passes `do_sample=False`, and
`models.generate_with_hooks` passes `temperature=0` (TL's `sample_logits`
early-returns argmax at 0.0). Three seeds therefore return **byte-identical**
completions, and averaging them would report a three-fold CI that is really n=1.
`SampleSpec.seed` changes *which items* are drawn, not how they decode.

**What I did.** One seed, and the variance estimate is the **item bootstrap** —
`metrics.beat_rate` inline, `scripts/bootstrap_ci.py` on the judged CSV. This is
consistent with `PROJECT_STATE.md`, which already notes k=1 is "forced by greedy
decoding".

**The decision.** Three options, and they are not equivalent:
1. **Item bootstrap only** (what is reported now). Honest, but it does not measure
   decoding variance, because there is none to measure.
2. **Turn on sampling** (e.g. `temperature=0.7`, k=3). This is a *method change*: it
   changes what every historical number means and would need its own baseline. Not a
   decision a config should make quietly.
3. **Re-draw items** (`sample.seed` ∈ {0,1,2}, three runs). Measures item-selection
   variance — which the item bootstrap already estimates — at 3× the GPU cost.

My recommendation is (1) plus an explicit sentence in the paper that k=1 is forced by
greedy decoding, so the CI is over items and the reader is not left to assume
otherwise. If the team wants (2), every arm needs re-running, not just the reported
ones.

---

## Q2 — a covariance-matched random direction is not a random direction here

**The problem.** Exp-2's control is specified as "matched on norm AND residual
covariance". Built that way, it comes out **45% aligned with the opinion direction**
(mean |cos| 0.4522, max 0.9686, where a random draw in the rank-398 residual subspace
would give ~0.05). Both plausible estimators do this — pooled covariance gives 0.4502,
within-pole 0.4522 — so it is not a centring mistake. The spectrum shows why:

| layer | \|cos(v_opinion, PC1)\| | PC1 share of within-pole variance |
|---|---|---|
| 12 | 0.885 | 0.952 |
| 18 | 0.748 | 0.916 |
| 24 | 0.522 | 0.684 |
| 35 | 0.648 | 0.129 |

At the middle layers the residual covariance is nearly **rank one**, and the opinion
direction lies along that one component. So "matched on the covariance" and "a
different direction" are in direct tension: nearly every vector the covariance can
produce *is* the opinion direction again. Such a control would move the opinion rate,
and Exp-2 would report the direction as non-specific having shown only that its
control was not random.

**What I did.** The reported control adds `--orthogonalise`: project the opinion
direction out of each layer's draw, then rematch the norm. Same per-layer norm, same
anisotropic envelope, realised mean |cos| 0.0000. The non-orthogonalised draw is kept
(`--covariance within-pole` without the flag) as the evidence above.

**The decision.** Is the orthogonalised draw the control Exp-2 reports? I think it
must be, because it is the only version that isolates direction from magnitude. But
note what it concedes: with the covariance this low-rank, "a random direction of the
same norm and covariance" is not available as a control, and the paper should say so
rather than imply the stronger control was run. This is also a point *about* the
representation — a near-rank-one residual covariance aligned with the fitted direction
is worth a sentence in its own right, and it sits close to the non-identifiability
caveat the contract already requires (arXiv:2602.06801: "**a** direction").

---

## Q3 — the dose ladder needs re-generating before any dose above 8 is reportable

**The problem.** The `adaptive-add-linear-c20/c30/c40` runs are invalid as behaviour
measurements: `max_tokens=128` with `enable_thinking` unset (ON for Qwen3-8B) and
`strip_reasoning` off, giving an unsteered arm that is **98.5% truncated
mid-`<think>`**. Full evidence and per-arm table:
`runs/laneA_exp3_coherence_doseladder/summary.md`. This is the configuration
HANDOFF §0.3 names as fatal.

**What I did.** Exp-1 stays at c=8 and Exp-2's control runs at c=8, so both sit at the
one dose with a valid treatment number. I did not re-run the ladder: ~40 min of A100
per dose, and the pivotal Exp-1 had priority.

**The decision.** Who re-runs c20/c30/c40 with `enable_thinking=False` and
`max_tokens ≥ 512`, and does the paper need doses above 8 at all? If the headline is
"the direction beats prompting at a coherent dose", c=8 with a coherence gate may be
sufficient and the ladder becomes a robustness appendix.

---

## Q4 — the reportable-dose rule needs a floor, not just a comparison

"Reportable dose = the largest dose whose coherence-pass rate ≥ the unsteered
baseline's" presumes a healthy baseline. On the ladder the baseline passes 0.015, so
every arm at every dose clears it — the rule certifies doses measured against 197
broken reference completions. Suggest: **a dose is reportable only if the unsteered
baseline itself passes at some absolute floor** (0.90 is the natural candidate, and
Exp-1's baseline should be checked against it) **and** the dose's pass rate is within
noise of that baseline. I have not adopted a floor unilaterally, so
`scripts/coherence_gate.py` reports the comparison and leaves the floor to the team.

---

## Q5 — the rubric patch means "judge v2.1" no longer names one judge

The 2026-09-26 team decision (capability/identity declines → `non-engagement`) changes
the rubric string that v2.1 runs on. Any number already labelled "v2.1" was produced
by the *pre-patch* rubric and is not the same judge, even though the version name did
not move. `scripts/rejudge_v21.py` therefore records the **SHA-256 of the exact rubric
string** with every judged CSV, so two tables can be checked for agreement rather than
assumed to agree.

**The decision.** Either bump the version name (v2.2) so the name tracks the rubric,
or accept "v2.1 + rubric sha" as the citation unit. Also: any previously judged v2.1
table needs re-judging before it can share a table with Lane A's numbers (CLAUDE.md
§4). I do not know which tables those are; whoever produced them should check the
rubric hash.

---

## Q6 — control-plane conflict: which paper is this branch for?

`docs/HANDOFF_lane_a.md` says: *"This paper is the **positive steering** result
(Farhan's line), not the frozen θ shared-mechanism paper."* But `PROJECT_STATE.md` and
`RESEARCH_CONTRACT.md` — the two files CLAUDE.md makes canonical — describe the frozen
θ paper, and CLAUDE.md's standing rules say a handoff *"may not redefine the paper, an
experiment, a metric, a deadline, a model set, a rubric, a claim, or a definition of
done."*

I executed the handoff as written, because it is this branch's task and the work it
asks for (a prompt baseline, a random-direction control, a judge-independent coherence
gate, CIs) is methodological hardening that either paper needs and neither is harmed
by. **No framing text was written and no claim was made** — the deliverables are
measurements and their provenance.

But the conflict is real and only the team can resolve it: if the positive-steering
line is a second paper, it needs its own entry in the control plane
(`docs/SOURCES_OF_TRUTH.md` plus a contract or an amendment under §12), because right
now a reader of the canonical files would not know these runs exist, and a reader of
the handoff would not know the contract governs. Flagging rather than resolving, since
"do not expand scope" cuts against my inventing a second contract.
