# sweep qwen18 c6 — qwen-1.8b

- run_id: `20260809-013300_sweep-qwen18-c6_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=6, neutral=6
- git: `f6b9cd9f61a37fa7ab4ef74a64c945b820677601` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 74×neutral, 26×opinionated
- **steered_pos**: 24×neutral, 76×opinionated
- **steered_neg**: 96×neutral, 4×opinionated

## Steering quality
### opinion (toward pos)
- good: 50
- bad: 0
- same_good: 26
- same_bad: 24

### neutral (toward neg)
- good: 23
- bad: 1
- same_good: 73
- same_bad: 3

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
