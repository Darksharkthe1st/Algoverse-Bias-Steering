# linear-add-c8 qwen3-8b — qwen3-8b

- run_id: `20260903-055631_linear-add-c8-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `87ffffd432a16634eb05f7a2f68de7621910ae87` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 152×neutral, 48×opinionated
- **steered_pos**: 22×neutral, 178×opinionated
- **steered_neg**: 174×neutral, 26×opinionated

## Steering quality
### opinion (toward pos)
- good: 130
- bad: 0
- same_good: 48
- same_bad: 22

### neutral (toward neg)
- good: 42
- bad: 20
- same_good: 132
- same_bad: 6

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
