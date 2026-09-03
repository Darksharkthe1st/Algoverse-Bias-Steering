# adaptive-add-linear-c8 qwen3-8b — qwen3-8b

- run_id: `20260903-011119_adaptive-add-linear-c8-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `130bf259cd6c9fca1b327b98065a11692af34819` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 149×neutral, 51×opinionated
- **steered_pos**: 117×neutral, 83×opinionated
- **steered_neg**: 170×neutral, 30×opinionated

## Steering quality
### opinion (toward pos)
- good: 35
- bad: 3
- same_good: 48
- same_bad: 114

### neutral (toward neg)
- good: 30
- bad: 9
- same_good: 140
- same_bad: 21

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
