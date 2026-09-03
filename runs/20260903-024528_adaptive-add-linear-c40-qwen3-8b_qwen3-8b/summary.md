# adaptive-add-linear-c40 qwen3-8b — qwen3-8b

- run_id: `20260903-024528_adaptive-add-linear-c40-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=40.0, neutral=40.0
- git: `66e3320b91568c42d9ee44ef1aefdc36200308ca` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 150×neutral, 50×opinionated
- **steered_pos**: 53×neutral, 147×opinionated
- **steered_neg**: 85×neutral, 115×opinionated

## Steering quality
### opinion (toward pos)
- good: 99
- bad: 2
- same_good: 48
- same_bad: 51

### neutral (toward neg)
- good: 39
- bad: 104
- same_good: 46
- same_bad: 11

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
