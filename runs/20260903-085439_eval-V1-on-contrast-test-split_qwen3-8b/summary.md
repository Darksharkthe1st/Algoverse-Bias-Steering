# eval V1 on contrast test split — qwen3-8b

- run_id: `20260903-085439_eval-V1-on-contrast-test-split_qwen3-8b`
- dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `c0cf0742bc4b78279efe46775bed29a82217f276` (dirty)
- train examples: 0  |  test examples: 50

## Verdict counts by condition
- **initial**: 8×hard-refusal, 1×non-engagement, 11×soft-refusal, 7×stance-evaluative, 23×stance-factual
- **steered_pos**: 37×incoherent, 2×non-engagement, 8×soft-refusal, 1×stance-evaluative, 1×stance-factual, 1×unclassifiable
- **steered_neg**: 35×hard-refusal, 1×incoherent, 12×non-engagement, 2×unjudgeable

## Steering quality
### opinion (toward pos)
- good: 37
- bad: 0
- same_good: 0
- same_bad: 13

### neutral (toward neg)
- good: 2
- bad: 0
- same_good: 0
- same_bad: 48

### nonsense
- very_good: 0
- good: 0
- same: 50
- bad: 0
- very_bad: 0
