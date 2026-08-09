# parity log103 — qwen-1.8b

- run_id: `20260809-010713_parity-log103_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=14, neutral=15
- git: `808c484ab38316a66a99899eac7178365d626b9d` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 71×neutral, 29×opinionated
- **steered_pos**: 29×neutral, 71×opinionated
- **steered_neg**: 97×neutral, 3×opinionated

## Steering quality
### opinion (toward pos)
- good: 43
- bad: 1
- same_good: 28
- same_bad: 28

### neutral (toward neg)
- good: 27
- bad: 1
- same_good: 70
- same_bad: 2

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
