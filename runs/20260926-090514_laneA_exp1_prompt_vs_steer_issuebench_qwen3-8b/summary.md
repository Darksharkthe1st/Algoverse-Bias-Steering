# laneA_exp1_prompt_vs_steer_issuebench — qwen3-8b

- run_id: `20260926-090514_laneA_exp1_prompt_vs_steer_issuebench_qwen3-8b`
- intervention: `both`  |  dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `81a7c8d13346153324fa35d05954b4475a6eff0a` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 98×neutral, 102×opinionated
- **steered_pos**: 19×neutral, 3×nonsense, 178×opinionated
- **steered_neg**: 194×neutral, 6×opinionated
- **prompt_pos**: 51×neutral, 149×opinionated
- **prompt_neg**: 183×neutral, 17×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 81
- bad: 5
- same_good: 97
- same_bad: 17

### neutral (toward neg)
- good: 97
- bad: 1
- same_good: 97
- same_bad: 5

### nonsense
- very_good: 0
- good: 0
- same: 197
- bad: 3
- very_bad: 0

## Prompt-baseline quality (system prompt)
### opinion (toward pos)
- good: 53
- bad: 6
- same_good: 96
- same_bad: 45

### neutral (toward neg)
- good: 85
- bad: 0
- same_good: 98
- same_bad: 17

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0

## Steer vs prompt (per-item, item-bootstrap CI)
- **opinion** (target `opinionated`, n=200): steer 0.890 vs prompt 0.745  |  Δ=+0.145  [90% CI +0.090, +0.200]  → steer beats prompt
  - per-item: both 140 · steer-only 38 · prompt-only 9 · neither 13  (discordant 47)
- **neutral** (target `neutral`, n=200): steer 0.970 vs prompt 0.915  |  Δ=+0.055  [90% CI +0.020, +0.090]  → steer beats prompt
  - per-item: both 178 · steer-only 16 · prompt-only 5 · neither 1  (discordant 21)

_Δ>0 with a CI clear of 0 = the direction beats prompting; otherwise report the bound (needed-experiments §14, FK-5). Read the discordant cells before concluding 'no difference': a small Δ with a large `discordant` count means the methods are COMPLEMENTARY — each wins a different subset of items (steer-only vs prompt-only) — not interchangeable._
