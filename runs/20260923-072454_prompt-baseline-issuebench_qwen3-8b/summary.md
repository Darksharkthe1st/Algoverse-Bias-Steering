# prompt-baseline issuebench — qwen3-8b

- run_id: `20260923-072454_prompt-baseline-issuebench_qwen3-8b`
- intervention: `both`  |  dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `25be7f2ef931f5deddb84a662cd2830b26763098` (dirty)
- train examples: 0  |  test examples: 150

## Verdict counts by condition
- **initial**: 66×neutral, 84×opinionated
- **steered_pos**: 93×neutral, 7×nonsense, 50×opinionated
- **steered_neg**: 93×neutral, 57×opinionated
- **prompt_pos**: 21×neutral, 129×opinionated
- **prompt_neg**: 137×neutral, 13×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 13
- bad: 47
- same_good: 37
- same_bad: 53

### neutral (toward neg)
- good: 43
- bad: 16
- same_good: 50
- same_bad: 41

### nonsense
- very_good: 0
- good: 0
- same: 143
- bad: 7
- very_bad: 0

## Prompt-baseline quality (system prompt)
### opinion (toward pos)
- good: 47
- bad: 2
- same_good: 82
- same_bad: 19

### neutral (toward neg)
- good: 71
- bad: 0
- same_good: 66
- same_bad: 13

### nonsense
- very_good: 0
- good: 0
- same: 150
- bad: 0
- very_bad: 0

## Steer vs prompt (per-item, item-bootstrap CI)
- **opinion** (target `opinionated`, n=150): steer 0.333 vs prompt 0.860  |  Δ=-0.527  [90% CI -0.593, -0.453]  → prompt beats steer
  - per-item: both 49 · steer-only 1 · prompt-only 80 · neither 20  (discordant 81)
- **neutral** (target `neutral`, n=150): steer 0.620 vs prompt 0.913  |  Δ=-0.293  [90% CI -0.367, -0.220]  → prompt beats steer
  - per-item: both 86 · steer-only 7 · prompt-only 51 · neither 6  (discordant 58)

_Δ>0 with a CI clear of 0 = the direction beats prompting; otherwise report the bound (needed-experiments §14, FK-5). Read the discordant cells before concluding 'no difference': a small Δ with a large `discordant` count means the methods are COMPLEMENTARY — each wins a different subset of items (steer-only vs prompt-only) — not interchangeable._
