# How we work — humans, agents, and what counts as evidence

*For the whole team. Peer-level. No hierarchy theater.*

## Roles (peers, different loads)

| Person | Center of gravity | Trust |
|---|---|---|
| **Farhan** | Pipeline, GPU runs, refactor — original builder | High autonomy on engineering; still bound by framing + claim ledger |
| **Jeremiah** | Measurement validity, fault study, geometry foundations, labeling | Design authority on his workstreams; paper lead when writing |
| **Edward** | Evals, toolkit, Gate 1 harness, landscape / stakes notes, team kit | Owns harnesses and sources-of-truth hygiene; not sole decider on science |

Disagreement on framing or claims → PR the owner file (`PAPER_FRAMING.md`,
`RUBRIC_v2.md`, etc.). Don’t “fix quietly” in a chat summary.

## Agents are tools, not co-authors of truth

| Use agents for | Don’t use agents for |
|---|---|
| Code, refactors, plots, log parsing | Replacing human gold labels for Gate 1 κ |
| Drafting docs you’ll edit | Inventing numbers or “recalling” archive stats |
| Computing κ / recounts from text logs | Unpickling untrusted blobs without need |
| Literature search with citations | Restating the rubric in a fifth file |

If an agent wrote it, a human still owns whether it’s **true for this repo**.

## What must be human (Gate 1)

**Assigning behavior labels** on the calibration and scored sheets that we use
for Cohen’s κ must be **independent humans** who have read `docs/RUBRIC_v2.md`.

**Source text:** 2026 model-set models only (`docs/MODEL_SET_2026-08-07.md`).
The 2025 archive is not a valid gold pool for current-system claims.

Why: the project exists because automated judgment mixed causes. Making the
gate “LLM judges the LLM” is circular. Scripts may prepare CSVs and score
agreement; they may not fill the official gold labels.

Practice labeling with a model for *discussion* is fine if you mark it as
practice and don’t put it in the κ numerator.

## Epistemic minimum for any steering claim

Before a number is “ours” in a slide or paper:

1. **Judge version** (commit hash of rubric / prompt) attached  
2. **Denominator** stated (archive: n = 96 per arm unless re-run says otherwise)  
3. **System-prompt baseline** on the same prompts when claiming an activation
   intervention  
4. **Per-example distributions** (not only means)  
5. **Tensor shape** asserted — no silent broadcast class of bug  
6. Claim ledger updated if we change what we assert  

## Concepts map (shared vocabulary)

| Phrase | Means here |
|---|---|
| Soft refusal | Contested-but-benign: engages but won’t take a side |
| Hard refusal | Safety/harm decline (Arditi lineage) |
| Opinionation | Whether a side is taken (scalar), not which ideology |
| Direction d | Difference-in-means handle in residual space — *a* direction |
| S4 / S5 | Residual runtime edit vs weight abliteration (`INTERVENTION_CLASSES.md`) |
| Fog gate | Soft refusal as access-like asymmetry (stakes note — not yet paper doctrine) |

## If you’re lost

1. `docs/THE_CORRECT_PROBLEM.md`  
2. Your `RUNBOOK_<NAME>.md`  
3. `docs/SOURCES_OF_TRUTH.md`  
4. Ask with tried / expected / saw  

That’s enough.
