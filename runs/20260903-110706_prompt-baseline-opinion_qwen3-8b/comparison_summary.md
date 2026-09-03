<!-- DERIVED ARTIFACT, not a native run output. Merges two independently-run,
     same-seed evaluations offline: runs/20260903-093536_prompt-baseline-opinion_qwen3-8b (PROMPT_POS/PROMPT_NEG) +
     runs/20260903-110706_prompt-baseline-opinion_qwen3-8b (INITIAL/STEERED_POS/STEERED_NEG). See tmp_merge_steer_prompt.py.
     CAVEAT: the applied vector's own extraction run predates enable_thinking
     and ran thinking-ON at max_tokens=128 -- may itself carry truncation
     contamination from the same bug fixed for the prompt arms in this branch's
     history. Treat this comparison as exploratory, not a clean generalization
     test. -->

# prompt-baseline opinion (merged steer-vs-prompt) — qwen3-8b

- run_id: `MERGED(20260903-093536_prompt-baseline-opinion_qwen3-8b+20260903-110706_prompt-baseline-opinion_qwen3-8b)`
- intervention: `both`  |  dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `914aaf5a041a112d41beae4e5be6741e6720e3fb` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **prompt_pos**: 200×opinionated
- **prompt_neg**: 200×neutral
- **initial**: 51×neutral, 149×opinionated
- **steered_pos**: 37×neutral, 2×nonsense, 161×opinionated
- **steered_neg**: 79×neutral, 121×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 40
- bad: 28
- same_good: 121
- same_bad: 11

### neutral (toward neg)
- good: 62
- bad: 34
- same_good: 17
- same_bad: 87

### nonsense
- very_good: 0
- good: 0
- same: 198
- bad: 2
- very_bad: 0

## Prompt-baseline quality (system prompt)
### opinion (toward pos)
- good: 51
- bad: 0
- same_good: 149
- same_bad: 0

### neutral (toward neg)
- good: 149
- bad: 0
- same_good: 51
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0

## Steer vs prompt (per-item, item-bootstrap CI)
- **opinion** (target `opinionated`, n=200): steer 0.805 vs prompt 1.000  |  Δ=-0.195  [90% CI -0.240, -0.150]  → prompt beats steer
  - per-item: both 161 · steer-only 0 · prompt-only 39 · neither 0  (discordant 39)
- **neutral** (target `neutral`, n=200): steer 0.395 vs prompt 1.000  |  Δ=-0.605  [90% CI -0.660, -0.545]  → prompt beats steer
  - per-item: both 79 · steer-only 0 · prompt-only 121 · neither 0  (discordant 121)

_Δ>0 with a CI clear of 0 = the direction beats prompting; otherwise report the bound (needed-experiments §14, FK-5). Read the discordant cells before concluding 'no difference': a small Δ with a large `discordant` count means the methods are COMPLEMENTARY — each wins a different subset of items (steer-only vs prompt-only) — not interchangeable._
