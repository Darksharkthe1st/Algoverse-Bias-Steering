# prompt-baseline opinion — qwen3-8b

- run_id: `20260923-030712_prompt-baseline-opinion_qwen3-8b`
- intervention: `both`  |  dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `6c32c02a51b5b53088a187755a11c0171efcdf35` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 50×neutral, 150×opinionated
- **steered_pos**: 38×neutral, 1×nonsense, 161×opinionated
- **steered_neg**: 78×neutral, 122×opinionated
- **prompt_pos**: 200×opinionated
- **prompt_neg**: 200×neutral

## Steering quality (vector)
### opinion (toward pos)
- good: 39
- bad: 28
- same_good: 122
- same_bad: 11

### neutral (toward neg)
- good: 62
- bad: 34
- same_good: 16
- same_bad: 88

### nonsense
- very_good: 0
- good: 0
- same: 199
- bad: 1
- very_bad: 0

## Prompt-baseline quality (system prompt)
### opinion (toward pos)
- good: 50
- bad: 0
- same_good: 150
- same_bad: 0

### neutral (toward neg)
- good: 150
- bad: 0
- same_good: 50
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
- **neutral** (target `neutral`, n=200): steer 0.390 vs prompt 1.000  |  Δ=-0.610  [90% CI -0.665, -0.555]  → prompt beats steer
  - per-item: both 78 · steer-only 0 · prompt-only 122 · neither 0  (discordant 122)

_Δ>0 with a CI clear of 0 = the direction beats prompting; otherwise report the bound (needed-experiments §14, FK-5). Read the discordant cells before concluding 'no difference': a small Δ with a large `discordant` count means the methods are COMPLEMENTARY — each wins a different subset of items (steer-only vs prompt-only) — not interchangeable._
