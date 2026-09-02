# adaptive-add-target8 qwen3-8b — qwen3-8b

- run_id: `20260902-100757_adaptive-add-target8-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `b7c53eed19c06b8342b12b11a3a26ce1a720b4c9` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 150×neutral, 50×opinionated
- **steered_pos**: 200×neutral
- **steered_neg**: 200×neutral

## Steering quality
### opinion (toward pos)
- good: 0
- bad: 50
- same_good: 0
- same_bad: 150

### neutral (toward neg)
- good: 50
- bad: 0
- same_good: 150
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
