# fixed-add qwen3-8b — qwen3-8b

- run_id: `20260902-082522_fixed-add-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `b7c53eed19c06b8342b12b11a3a26ce1a720b4c9` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 146×neutral, 54×opinionated
- **steered_pos**: 81×neutral, 119×opinionated
- **steered_neg**: 189×neutral, 11×opinionated

## Steering quality
### opinion (toward pos)
- good: 66
- bad: 1
- same_good: 53
- same_bad: 80

### neutral (toward neg)
- good: 45
- bad: 2
- same_good: 144
- same_bad: 9

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
