# adaptive-add-linear-c20 qwen3-8b — qwen3-8b

- run_id: `20260903-020329_adaptive-add-linear-c20-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=20.0, neutral=20.0
- git: `1bc3fee1b175cde4b701bc50544d35fa718a3c39` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 152×neutral, 48×opinionated
- **steered_pos**: 74×neutral, 126×opinionated
- **steered_neg**: 180×neutral, 20×opinionated

## Steering quality
### opinion (toward pos)
- good: 83
- bad: 5
- same_good: 43
- same_bad: 69

### neutral (toward neg)
- good: 35
- bad: 7
- same_good: 145
- same_bad: 13

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
