# crows qwen-1.8b — qwen-1.8b

- run_id: `20260809-021510_crows-qwen-1.8b_qwen-1.8b`
- dataset: `crows_q`  |  method coeffs: opinion=14, neutral=15
- git: `0767e977092d3ee59e919bf9e153dde75e36e0d7` (dirty)
- train examples: 150  |  test examples: 150

## Verdict counts by condition
- **initial**: 94×neutral, 56×opinionated
- **steered_pos**: 72×neutral, 78×opinionated
- **steered_neg**: 137×neutral, 13×opinionated

## Steering quality
### opinion (toward pos)
- good: 44
- bad: 22
- same_good: 34
- same_bad: 50

### neutral (toward neg)
- good: 48
- bad: 5
- same_good: 89
- same_bad: 8

### nonsense
- very_good: 0
- good: 0
- same: 150
- bad: 0
- very_bad: 0
