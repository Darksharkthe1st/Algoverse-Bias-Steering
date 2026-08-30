# anchor qwen-7b — qwen-7b

- run_id: `20260809-020532_anchor-qwen-7b_qwen-7b`
- dataset: `snapshot`  |  method coeffs: opinion=13, neutral=15
- git: `af3cf34b05ee6d80b538dc970f90e474e31a6e1b` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 60×neutral, 40×opinionated
- **steered_pos**: 66×neutral, 34×opinionated
- **steered_neg**: 99×neutral, 1×opinionated

## Steering quality
### opinion (toward pos)
- good: 6
- bad: 12
- same_good: 28
- same_bad: 54

### neutral (toward neg)
- good: 39
- bad: 0
- same_good: 60
- same_bad: 1

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
