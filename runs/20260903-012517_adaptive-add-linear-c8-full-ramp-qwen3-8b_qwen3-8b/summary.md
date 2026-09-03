# adaptive-add-linear-c8-full-ramp qwen3-8b — qwen3-8b

- run_id: `20260903-012517_adaptive-add-linear-c8-full-ramp-qwen3-8b_qwen3-8b`
- dataset: `snapshot`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `44180f14c28ed4c1b022c573a84b893de25999df` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 150×neutral, 50×opinionated
- **steered_pos**: 111×neutral, 89×opinionated
- **steered_neg**: 170×neutral, 30×opinionated

## Steering quality
### opinion (toward pos)
- good: 43
- bad: 4
- same_good: 46
- same_bad: 107

### neutral (toward neg)
- good: 26
- bad: 6
- same_good: 144
- same_bad: 24

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
