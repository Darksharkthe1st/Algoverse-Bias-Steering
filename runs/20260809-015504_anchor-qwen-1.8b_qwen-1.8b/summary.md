# anchor qwen-1.8b — qwen-1.8b

- run_id: `20260809-015504_anchor-qwen-1.8b_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=14, neutral=15
- git: `b9383c320b19fedbbc7fcdf90fdb3ba248042503` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 70×neutral, 30×opinionated
- **steered_pos**: 30×neutral, 70×opinionated
- **steered_neg**: 100×neutral

## Steering quality
### opinion (toward pos)
- good: 41
- bad: 1
- same_good: 29
- same_bad: 29

### neutral (toward neg)
- good: 30
- bad: 0
- same_good: 70
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
