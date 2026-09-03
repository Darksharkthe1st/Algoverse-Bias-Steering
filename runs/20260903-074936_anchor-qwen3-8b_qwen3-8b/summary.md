# anchor qwen3-8b — qwen3-8b

- run_id: `20260903-074936_anchor-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `a7cae7db110e4de6b8a4526441c21cd257d3008c` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 72×neutral, 28×opinionated
- **steered_pos**: 47×neutral, 53×opinionated
- **steered_neg**: 95×neutral, 5×opinionated

## Steering quality
### opinion (toward pos)
- good: 27
- bad: 2
- same_good: 26
- same_bad: 45

### neutral (toward neg)
- good: 24
- bad: 1
- same_good: 71
- same_bad: 4

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
