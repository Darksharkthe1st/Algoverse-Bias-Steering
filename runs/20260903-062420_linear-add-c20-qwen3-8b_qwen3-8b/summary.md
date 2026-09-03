# linear-add-c20 qwen3-8b — qwen3-8b

- run_id: `20260903-062420_linear-add-c20-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=20.0, neutral=20.0
- git: `1d614b425261b53b3acfba9096933013b1873649` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 152×neutral, 48×opinionated
- **steered_pos**: 198×neutral, 2×opinionated
- **steered_neg**: 199×neutral, 1×opinionated

## Steering quality
### opinion (toward pos)
- good: 2
- bad: 48
- same_good: 0
- same_bad: 150

### neutral (toward neg)
- good: 48
- bad: 1
- same_good: 151
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
