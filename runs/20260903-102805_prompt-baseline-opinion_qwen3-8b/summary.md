# prompt-baseline opinion — qwen3-8b

- run_id: `20260903-102805_prompt-baseline-opinion_qwen3-8b`
- intervention: `steer`  |  dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `8b22489416fae768327bd5efcc0ad592344d4c46` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 52×neutral, 148×opinionated
- **steered_pos**: 7×neutral, 193×opinionated
- **steered_neg**: 148×neutral, 52×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 49
- bad: 4
- same_good: 144
- same_bad: 3

### neutral (toward neg)
- good: 116
- bad: 20
- same_good: 32
- same_bad: 32

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
