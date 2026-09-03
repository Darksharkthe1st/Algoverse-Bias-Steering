# eval V2 on contrast test split — qwen3-8b

- run_id: `20260903-092039_eval-V2-on-contrast-test-split_qwen3-8b`
- dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `c0cf0742bc4b78279efe46775bed29a82217f276` (dirty)
- train examples: 0  |  test examples: 50

## Verdict counts by condition
- **initial**: 8×hard-refusal, 1×non-engagement, 11×soft-refusal, 8×stance-evaluative, 22×stance-factual
- **steered_pos**: 8×hard-refusal, 1×incoherent, 4×soft-refusal, 5×stance-evaluative, 32×stance-factual
- **steered_neg**: 2×hard-refusal, 2×incoherent, 7×non-engagement, 30×soft-refusal, 2×stance-evaluative, 7×stance-factual

## Steering quality
### opinion (toward pos)
- good: 1
- bad: 0
- same_good: 0
- same_bad: 49

### neutral (toward neg)
- good: 0
- bad: 0
- same_good: 0
- same_bad: 50

### nonsense
- very_good: 0
- good: 0
- same: 50
- bad: 0
- very_bad: 0
