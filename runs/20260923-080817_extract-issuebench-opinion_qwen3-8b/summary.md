# extract issuebench opinion — qwen3-8b

- run_id: `20260923-080817_extract-issuebench-opinion_qwen3-8b`
- intervention: `steer`  |  dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `25be7f2ef931f5deddb84a662cd2830b26763098` (dirty)
- train examples: 200  |  test examples: 200

## Verdict counts by condition
- **initial**: 168×neutral, 32×opinionated
- **steered_pos**: 131×neutral, 69×opinionated
- **steered_neg**: 174×neutral, 26×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 45
- bad: 8
- same_good: 24
- same_bad: 123

### neutral (toward neg)
- good: 18
- bad: 12
- same_good: 156
- same_bad: 14

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
