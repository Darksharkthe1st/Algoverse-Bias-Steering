# sweep qwen18 c2 — qwen-1.8b

- run_id: `20260809-012833_sweep-qwen18-c2_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=2, neutral=2
- git: `59e6e9687a69310c3361bfa31ef8bdcc395cb058` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 73×neutral, 27×opinionated
- **steered_pos**: 56×neutral, 44×opinionated
- **steered_neg**: 86×neutral, 14×opinionated

## Steering quality
### opinion (toward pos)
- good: 18
- bad: 1
- same_good: 26
- same_bad: 55

### neutral (toward neg)
- good: 16
- bad: 3
- same_good: 70
- same_bad: 11

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
