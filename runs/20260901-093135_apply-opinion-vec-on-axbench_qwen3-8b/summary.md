# apply opinion vec on axbench — qwen3-8b

- run_id: `20260901-093135_apply-opinion-vec-on-axbench_qwen3-8b`
- dataset: `axbench`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `e801c2a45f60f9f10892feb8697618befaf2599e` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 169×neutral, 31×opinionated
- **steered_pos**: 145×neutral, 55×opinionated
- **steered_neg**: 183×neutral, 17×opinionated

## Steering quality
### opinion (toward pos)
- good: 36
- bad: 12
- same_good: 19
- same_bad: 133

### neutral (toward neg)
- good: 25
- bad: 11
- same_good: 158
- same_bad: 6

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
