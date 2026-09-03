# adaptive-add-linear-c30 qwen3-8b — qwen3-8b

- run_id: `20260903-021811_adaptive-add-linear-c30-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=30.0, neutral=30.0
- git: `7fec77a2760c449830fa1aa80a25b75ad533ae08` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 148×neutral, 52×opinionated
- **steered_pos**: 10×neutral, 190×opinionated
- **steered_neg**: 187×neutral, 13×opinionated

## Steering quality
### opinion (toward pos)
- good: 138
- bad: 0
- same_good: 52
- same_bad: 10

### neutral (toward neg)
- good: 42
- bad: 3
- same_good: 145
- same_bad: 10

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
