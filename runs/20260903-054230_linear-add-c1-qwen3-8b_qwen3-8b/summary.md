# linear-add-c1 qwen3-8b — qwen3-8b

- run_id: `20260903-054230_linear-add-c1-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=1.0, neutral=1.0
- git: `0416b9470c5ca32c9bd03d48927e53ce4675ae11` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 149×neutral, 51×opinionated
- **steered_pos**: 143×neutral, 57×opinionated
- **steered_neg**: 164×neutral, 36×opinionated

## Steering quality
### opinion (toward pos)
- good: 19
- bad: 13
- same_good: 38
- same_bad: 130

### neutral (toward neg)
- good: 27
- bad: 12
- same_good: 137
- same_bad: 24

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
