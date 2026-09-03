# prompt-baseline opinion — qwen3-8b

- run_id: `20260903-093536_prompt-baseline-opinion_qwen3-8b`
- intervention: `prompt`  |  dataset: `plain`  |  method coeffs: opinion=8.0, neutral=8.0
- git: `300325f8f2f4df0803796d1edb80796ce5d814d7` (dirty)
- train examples: 0  |  test examples: 200

## Verdict counts by condition
- **initial**: 51×neutral, 149×opinionated
- **prompt_pos**: 200×opinionated
- **prompt_neg**: 200×neutral

## Prompt-baseline quality (system prompt)
### opinion (toward pos)
- good: 51
- bad: 0
- same_good: 149
- same_bad: 0

### neutral (toward neg)
- good: 149
- bad: 0
- same_good: 51
- same_bad: 0

### nonsense
- very_good: 0
- good: 0
- same: 200
- bad: 0
- very_bad: 0
