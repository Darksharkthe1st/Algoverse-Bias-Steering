# extract axbench opinion — qwen3-8b

- run_id: `20260901-085117_extract-axbench-opinion_qwen3-8b`
- dataset: `axbench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `59b4d37544556239bd92c2bfaa638d6c5b70fd85` (dirty)
- train examples: 200  |  test examples: 200

## Verdict counts by condition
- **initial**: 168×neutral, 32×opinionated
- **steered_pos**: 141×neutral, 59×opinionated
- **steered_neg**: 186×neutral, 14×opinionated

## Steering quality
### opinion (toward pos)
- good: 43
- bad: 16
- same_good: 16
- same_bad: 125

### neutral (toward neg)
- good: 28
- bad: 10
- same_good: 158
- same_bad: 4

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
