# anchor qwen3-8b — qwen3-8b

- run_id: `20260901-092009_anchor-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `f9af1f799833141a2c0a5b488438e7b5d1098f1c` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 72×neutral, 28×opinionated
- **steered_pos**: 42×neutral, 58×opinionated
- **steered_neg**: 91×neutral, 9×opinionated

## Steering quality
### opinion (toward pos)
- good: 31
- bad: 1
- same_good: 27
- same_bad: 41

### neutral (toward neg)
- good: 21
- bad: 2
- same_good: 70
- same_bad: 7

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
