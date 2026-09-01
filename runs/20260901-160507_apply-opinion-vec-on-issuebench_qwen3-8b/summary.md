# apply opinion vec on issuebench — qwen3-8b

- run_id: `20260901-160507_apply-opinion-vec-on-issuebench_qwen3-8b`
- dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `52ae1abba50813858d840f3d8074d8efa97f4adb` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 167×neutral, 33×opinionated
- **steered_pos**: 116×neutral, 84×opinionated
- **steered_neg**: 178×neutral, 22×opinionated

## Steering quality
### opinion (toward pos)
- good: 61
- bad: 10
- same_good: 23
- same_bad: 106

### neutral (toward neg)
- good: 23
- bad: 12
- same_good: 155
- same_bad: 10

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
