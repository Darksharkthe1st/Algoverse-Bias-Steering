# anchor qwen3-8b — qwen3-8b

- run_id: `20260903-105600_anchor-qwen3-8b_qwen3-8b`
- intervention: `steer`  |  dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `914aaf5a041a112d41beae4e5be6741e6720e3fb` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 30×neutral, 70×opinionated
- **steered_pos**: 10×neutral, 90×opinionated
- **steered_neg**: 31×neutral, 69×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 26
- bad: 6
- same_good: 64
- same_bad: 4

### neutral (toward neg)
- good: 27
- bad: 26
- same_good: 4
- same_bad: 43

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
