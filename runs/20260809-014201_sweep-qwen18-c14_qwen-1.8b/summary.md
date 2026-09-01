# sweep qwen18 c14 — qwen-1.8b

- run_id: `20260809-014201_sweep-qwen18-c14_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=14, neutral=14
- git: `4031ed85564f44bd0d8c8c6fc7772d42838995b8` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 73×neutral, 27×opinionated
- **steered_pos**: 22×neutral, 78×opinionated
- **steered_neg**: 99×neutral, 1×opinionated

## Steering quality
### opinion (toward pos)
- good: 51
- bad: 0
- same_good: 27
- same_bad: 22

### neutral (toward neg)
- good: 26
- bad: 0
- same_good: 73
- same_bad: 1

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
