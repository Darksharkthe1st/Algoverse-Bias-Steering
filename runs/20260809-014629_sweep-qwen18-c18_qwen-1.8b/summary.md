# sweep qwen18 c18 — qwen-1.8b

- run_id: `20260809-014629_sweep-qwen18-c18_qwen-1.8b`
- dataset: `snapshot`  |  method coeffs: opinion=18, neutral=18
- git: `8a56fb822087b4e483f52b1c13605182bf41c4c4` (dirty)
- train examples: 100  |  test examples: 100

## Verdict counts by condition
- **initial**: 72×neutral, 28×opinionated
- **steered_pos**: 57×neutral, 43×opinionated
- **steered_neg**: 99×neutral, 1×opinionated

## Steering quality
### opinion (toward pos)
- good: 24
- bad: 9
- same_good: 19
- same_bad: 48

### neutral (toward neg)
- good: 28
- bad: 1
- same_good: 71
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 100
- bad: 0
- very_bad: 0
