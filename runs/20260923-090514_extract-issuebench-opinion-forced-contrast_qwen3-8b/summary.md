# extract issuebench opinion (forced-contrast) — qwen3-8b

- run_id: `20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b`
- intervention: `steer`  |  dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `1d16f8cbe427420820d7fec1dff4b5979fce2af8` (dirty)
- train examples: 200  |  test examples: 100

## Verdict counts by condition
- **initial**: 44×neutral, 56×opinionated
- **steered_pos**: 11×neutral, 89×opinionated
- **steered_neg**: 91×neutral, 9×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 36
- bad: 3
- same_good: 53
- same_bad: 8

### neutral (toward neg)
- good: 48
- bad: 1
- same_good: 43
- same_bad: 8

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
