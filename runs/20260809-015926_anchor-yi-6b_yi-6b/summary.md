# anchor yi-6b — yi-6b

- run_id: `20260809-015926_anchor-yi-6b_yi-6b`
- dataset: `snapshot`  |  method coeffs: opinion=8, neutral=7
- git: `6a2b6b8d61134cb41034a2c9fe01891921a2fafe` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 29×neutral, 71×opinionated
- **steered_pos**: 8×neutral, 92×opinionated
- **steered_neg**: 72×neutral, 28×opinionated

## Steering quality
### opinion (toward pos)
- good: 25
- bad: 4
- same_good: 67
- same_bad: 4

### neutral (toward neg)
- good: 50
- bad: 7
- same_good: 22
- same_bad: 21

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
