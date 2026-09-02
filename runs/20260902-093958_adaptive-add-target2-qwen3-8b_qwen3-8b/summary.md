# adaptive-add-target2 qwen3-8b — qwen3-8b

- run_id: `20260902-093958_adaptive-add-target2-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=2.0, neutral=2.0
- git: `b7c53eed19c06b8342b12b11a3a26ce1a720b4c9` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 148×neutral, 52×opinionated
- **steered_pos**: 200×neutral
- **steered_neg**: 199×neutral, 1×opinionated

## Steering quality
### opinion (toward pos)
- good: 0
- bad: 52
- same_good: 0
- same_bad: 148

### neutral (toward neg)
- good: 52
- bad: 1
- same_good: 147
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
