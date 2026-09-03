<!-- DERIVED ARTIFACT, not a native run output. Merges two independently-run,
     same-seed evaluations offline: runs/20260903-093536_prompt-baseline-opinion_qwen3-8b (PROMPT_POS/PROMPT_NEG) +
     runs/20260903-102805_prompt-baseline-opinion_qwen3-8b (INITIAL/STEERED_POS/STEERED_NEG). See tmp_merge_steer_prompt.py.
     CAVEAT: the applied vector's own extraction run predates enable_thinking
     and ran thinking-ON at max_tokens=128 -- may itself carry truncation
     contamination from the same bug fixed for the prompt arms in this branch's
     history. Treat this comparison as exploratory, not a clean generalization
     test. -->

# prompt-baseline opinion (merged steer-vs-prompt) — qwen3-8b

- run_id: `MERGED(20260903-093536_prompt-baseline-opinion_qwen3-8b+20260903-102805_prompt-baseline-opinion_qwen3-8b)`
- intervention: `both`  |  dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `8b22489416fae768327bd5efcc0ad592344d4c46` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **prompt_pos**: 200×opinionated
- **prompt_neg**: 200×neutral
- **initial**: 52×neutral, 148×opinionated
- **steered_pos**: 7×neutral, 193×opinionated
- **steered_neg**: 148×neutral, 52×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 49
- bad: 4
- same_good: 144
- same_bad: 3

### neutral (toward neg)
- good: 116
- bad: 20
- same_good: 32
- same_bad: 32

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0

## Prompt-baseline quality (system prompt)
### opinion (toward pos)
- good: 52
- bad: 0
- same_good: 148
- same_bad: 0

### neutral (toward neg)
- good: 148
- bad: 0
- same_good: 52
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0

## Steer vs prompt (per-item, item-bootstrap CI)
- **opinion** (target `opinionated`, n=200): steer 0.965 vs prompt 1.000  |  Δ=-0.035  [90% CI -0.055, -0.015]  → prompt beats steer
  - per-item: both 193 · steer-only 0 · prompt-only 7 · neither 0  (discordant 7)
- **neutral** (target `neutral`, n=200): steer 0.740 vs prompt 1.000  |  Δ=-0.260  [90% CI -0.310, -0.210]  → prompt beats steer
  - per-item: both 148 · steer-only 0 · prompt-only 52 · neither 0  (discordant 52)

_Δ>0 with a CI clear of 0 = the direction beats prompting; otherwise report the bound (needed-experiments §14, FK-5). Read the discordant cells before concluding 'no difference': a small Δ with a large `discordant` count means the methods are COMPLEMENTARY — each wins a different subset of items (steer-only vs prompt-only) — not interchangeable._
