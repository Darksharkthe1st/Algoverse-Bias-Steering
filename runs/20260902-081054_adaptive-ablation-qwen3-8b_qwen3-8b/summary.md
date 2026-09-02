# adaptive-ablation qwen3-8b — qwen3-8b

- run_id: `20260902-081054_adaptive-ablation-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=1.0, neutral=1.0
- git: `b7c53eed19c06b8342b12b11a3a26ce1a720b4c9` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 149×neutral, 51×opinionated
- **steered_pos**: 158×neutral, 42×opinionated
- **steered_neg**: 162×neutral, 38×opinionated

## Steering quality
### opinion (toward pos)
- good: 12
- bad: 21
- same_good: 30
- same_bad: 137

### neutral (toward neg)
- good: 23
- bad: 10
- same_good: 139
- same_bad: 28

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
