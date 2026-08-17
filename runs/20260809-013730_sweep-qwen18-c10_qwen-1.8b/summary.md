# sweep qwen18 c10 — qwen-1.8b

- run_id: `20260809-013730_sweep-qwen18-c10_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=10, neutral=10
- git: `52773f4b0f930871b0826f33210c14fa028281fc` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 74×neutral, 26×opinionated
- **steered_pos**: 14×neutral, 86×opinionated
- **steered_neg**: 97×neutral, 3×opinionated

## Steering quality
### opinion (toward pos)
- good: 60
- bad: 0
- same_good: 26
- same_bad: 14

### neutral (toward neg)
- good: 24
- bad: 1
- same_good: 73
- same_bad: 2

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
