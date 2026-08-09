# crows qwen-7b — qwen-7b

- run_id: `20260809-022109_crows-qwen-7b_qwen-7b`
- dataset: `crows_q`  |  method coeffs: opinion=13, neutral=15
- git: `ae3570f269d37b66afd51be63b417d357817b96c` (dirty)
- train examples: 150  |  test examples: 150

## Verdict counts by condition
- **initial**: 131×neutral, 19×opinionated
- **steered_pos**: 47×neutral, 103×opinionated
- **steered_neg**: 148×neutral, 2×opinionated

## Steering quality
### opinion (toward pos)
- good: 88
- bad: 4
- same_good: 15
- same_bad: 43

### neutral (toward neg)
- good: 18
- bad: 1
- same_good: 130
- same_bad: 1

### nonsense
- very_good: 0
- good: 0
- same: 150
- bad: 0
- very_bad: 0
