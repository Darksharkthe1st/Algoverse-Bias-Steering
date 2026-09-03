# adaptive-add-linear qwen3-8b — qwen3-8b

- run_id: `20260903-004434_adaptive-add-linear-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=1.0, neutral=1.0
- git: `8800fe33a5503953930111204a48315843405d34` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 146×neutral, 54×opinionated
- **steered_pos**: 123×neutral, 77×opinionated
- **steered_neg**: 178×neutral, 22×opinionated

## Steering quality
### opinion (toward pos)
- good: 31
- bad: 8
- same_good: 46
- same_bad: 115

### neutral (toward neg)
- good: 36
- bad: 4
- same_good: 142
- same_bad: 18

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
