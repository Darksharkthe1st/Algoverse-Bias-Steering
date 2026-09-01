# extract issuebench opinion — qwen3-8b

- run_id: `20260901-091212_extract-issuebench-opinion_qwen3-8b`
- dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `59b4d37544556239bd92c2bfaa638d6c5b70fd85` (dirty)
- train examples: 200  |  test examples: 200

## Verdict counts by condition
- **initial**: 173×neutral, 27×opinionated
- **steered_pos**: 124×neutral, 76×opinionated
- **steered_neg**: 181×neutral, 19×opinionated

## Steering quality
### opinion (toward pos)
- good: 55
- bad: 6
- same_good: 21
- same_bad: 118

### neutral (toward neg)
- good: 16
- bad: 8
- same_good: 165
- same_bad: 11

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
