# laneA_exp2_covrandom_issuebench — qwen3-8b

- run_id: `20260926-100839_laneA_exp2_covrandom_issuebench_qwen3-8b`
- intervention: `steer`  |  dataset: `issuebench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `907d2c563d43a4065a531b190628234c5b99ef1b` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 96×neutral, 104×opinionated
- **steered_pos**: 93×neutral, 107×opinionated
- **steered_neg**: 85×neutral, 115×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 15
- bad: 12
- same_good: 92
- same_bad: 81

### neutral (toward neg)
- good: 11
- bad: 22
- same_good: 74
- same_bad: 93

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
