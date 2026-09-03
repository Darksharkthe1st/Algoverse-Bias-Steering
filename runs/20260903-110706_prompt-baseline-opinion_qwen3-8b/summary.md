# prompt-baseline opinion — qwen3-8b

- run_id: `20260903-110706_prompt-baseline-opinion_qwen3-8b`
- intervention: `steer`  |  dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `914aaf5a041a112d41beae4e5be6741e6720e3fb` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 51×neutral, 149×opinionated
- **steered_pos**: 37×neutral, 2×nonsense, 161×opinionated
- **steered_neg**: 79×neutral, 121×opinionated

## Steering quality (vector)
### opinion (toward pos)
- good: 40
- bad: 28
- same_good: 121
- same_bad: 11

### neutral (toward neg)
- good: 62
- bad: 34
- same_good: 17
- same_bad: 87

### nonsense
- very_good: 0
- good: 0
- same: 198
- bad: 2
- very_bad: 0
