# extract issuebench opinion — qwen3-8b

- run_id: `20260923-092656_extract-issuebench-opinion_qwen3-8b`
- intervention: `steer`  |  dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `1d16f8cbe427420820d7fec1dff4b5979fce2af8` (dirty)
- train examples: 200  |  test examples: 100

## Verdict counts by condition
- **initial**: 42×neutral, 58×opinionated
- **steered_pos**: 8×neutral, 92×opinionated
- **steered_neg**: 74×neutral, 1×nonsense, 25×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 35
- bad: 1
- same_good: 57
- same_bad: 7

### neutral (toward neg)
- good: 35
- bad: 3
- same_good: 39
- same_bad: 23

### nonsense
- very_good: 0
- good: 0
- same: 99
- bad: 1
- very_bad: 0
