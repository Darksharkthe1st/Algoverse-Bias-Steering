# linear-add-c30 qwen3-8b — qwen3-8b

- run_id: `20260903-063807_linear-add-c30-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=30.0, neutral=30.0
- git: `80a7f98854d1ce343e621ee2db3d054f4334ffe5` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 151×neutral, 49×opinionated
- **steered_pos**: 200×neutral
- **steered_neg**: 200×neutral

## Steering quality
### opinion (toward pos)
- good: 0
- bad: 49
- same_good: 0
- same_bad: 151

### neutral (toward neg)
- good: 49
- bad: 0
- same_good: 151
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
