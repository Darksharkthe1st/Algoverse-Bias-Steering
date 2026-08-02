# HANDOFF — Farhan (research lead · codebase owner)

> Read `PAPER_FRAMING.md` before writing any paper/summary text. Disagree → PR
> that file, don't fork the narrative.

You own the institutional memory: you ran all 244 logged experiments and built
the pipeline. The revival's first jobs that only you can do:

## Week 1

1. **Sign off (or red-line) `PAPER_FRAMING.md`.** It encodes the post-mortem's
   read of what worked/failed — you know where that read is wrong. The claims
   section is a working draft for the Tue thesis-lock discussion.
2. **Pipeline walkthrough (Tue meeting).** The logging/resume system in
   `experiments/farhan-experimentation.ipynb` is the team's main asset — 30
   minutes on: how a run starts, where pickles land, how `Log_N_*` dirs map to
   CSVs, and the coefficient-notation `(a:b)` convention.
3. **Recover the lost artifacts.** The Slack archive (graphs, analysis
   screenshots) once Kevin's export lands; the more-complete Overleaf draft if
   it exists anywhere; any local data from the old laptop.
4. **Judge v2 input.** You've read more model outputs than anyone — list the
   response patterns that fooled the v1 judge (both directions). That list is
   the core of the v2 rubric; your dormant `farhan-opinion-spectrum` branch
   (5-point scale + CoT) is the starting point.

5. **The week-1 technical task: unified re-extraction** (sprint plan §3).
   Port the opinionation extraction to Arditi conventions (post-instruction
   token positions — the 2025 mean-pool-over-all-tokens is a confound for any
   geometry comparison) and extract harm-refusal directions with the same
   pipeline, on all 4 sprint models. Your archived vectors
   (`experiments/best_vecs/`, `experiments/past_vecs/calculated_refusal_vecs/`)
   become sanity cross-checks. ~20 GPU-hrs.

## Then (sprint plan §3, gates in §5)

- Week 2: Gate 2 — reproduce bidirectional steering + Arditi bypass/induction
  with the re-extracted vectors.
- Week 3: the 4×5×2 cross-steering grid (~30k generations) — your Batched_Gen
  loop, with activation capping instead of constant coefficients. Degradation
  order is pre-committed in the plan; never shrink the safety battery.
- Lambda credits (~$377 ≈ 290 A100-hrs) are the whole GPU budget: ~200
  planned + 60 reserve. Judge costs are cash (GPT-4o-mini, ~$75–125 total).
