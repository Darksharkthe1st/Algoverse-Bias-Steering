# forced-adaptive-linear-c16 qwen3-8b — qwen3-8b

- run_id: `20260923-114535_forced-adaptive-linear-c16-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=16.0, neutral=16.0
- git: `6323bc2cf9aa233bc9103d4c2b6f1515aa4d31a4` (dirty)
- train examples: 0  |  test examples: 100

## Verdict counts by condition
- **initial**: 44×neutral, 56×opinionated
- **steered_pos**: 99×neutral, 1×nonsense
- **steered_neg**: 77×neutral, 23×opinionated

## Steering quality
### opinion (toward pos)
- good: 0
- bad: 56
- same_good: 0
- same_bad: 44

### neutral (toward neg)
- good: 33
- bad: 0
- same_good: 44
- same_bad: 23

### nonsense
- very_good: 0
- good: 0
- same: 99
- bad: 1
- very_bad: 0
