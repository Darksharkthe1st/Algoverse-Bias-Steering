# adaptive-add-linear-c16 qwen3-8b — qwen3-8b

- run_id: `20260903-014921_adaptive-add-linear-c16-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=16.0, neutral=16.0
- git: `ed51159d0c9b5d34a7fa19b8748f2c24ce02ddf6` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 150×neutral, 50×opinionated
- **steered_pos**: 86×neutral, 114×opinionated
- **steered_neg**: 181×neutral, 19×opinionated

## Steering quality
### opinion (toward pos)
- good: 68
- bad: 4
- same_good: 46
- same_bad: 82

### neutral (toward neg)
- good: 36
- bad: 5
- same_good: 145
- same_bad: 14

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
