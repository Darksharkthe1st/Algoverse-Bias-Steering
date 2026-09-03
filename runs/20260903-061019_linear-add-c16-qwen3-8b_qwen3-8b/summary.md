# linear-add-c16 qwen3-8b — qwen3-8b

- run_id: `20260903-061019_linear-add-c16-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=16.0, neutral=16.0
- git: `1bf05daccd2baad92efc7ef5de68ac69e035f5cf` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 144×neutral, 56×opinionated
- **steered_pos**: 74×neutral, 126×opinionated
- **steered_neg**: 193×neutral, 7×opinionated

## Steering quality
### opinion (toward pos)
- good: 76
- bad: 6
- same_good: 50
- same_bad: 68

### neutral (toward neg)
- good: 56
- bad: 7
- same_good: 137
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
