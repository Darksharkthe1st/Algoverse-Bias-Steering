# refusal repro — qwen-1.8b (refusal-direction repro)

- run_id: `20260816-011914_refusal-repro_qwen-1.8b`
- direction: layer=15, pos=-2, ‖r‖=26.287  |  act-add coeff magnitude=1.0
- git: `aaaa4554277894f6cd4d0e6a3596e16006b562b1` (dirty)

## Refusal rate by condition
- **harmful/baseline**: refusal 38/100 = 0.380  (success 0.620)
- **harmful/ablation**: refusal 0/100 = 0.000  (success 1.000)
- **harmful/actadd**: refusal 0/100 = 0.000  (success 1.000)
- **harmless/baseline**: refusal 1/100 = 0.010  (success 0.990)
- **harmless/actadd**: refusal 95/100 = 0.950  (success 0.050)

_Interpretation: ablation should DROP harmful refusal; act-add(+) should RAISE harmless refusal (arXiv:2406.11717)._

## vs. paper (refusal rate)

| arm | ours | paper | Δ | within ±0.05 |
|---|---|---|---|---|
| harmful/baseline | 0.380 | 0.700 | -0.320 | ✗ |
| harmful/ablation | 0.000 | 0.010 | -0.010 | ✓ |
| harmful/actadd | 0.000 | 0.030 | -0.030 | ✓ |
| harmless/baseline | 0.010 | 0.030 | -0.020 | ✓ |
| harmless/actadd | 0.950 | 0.980 | -0.030 | ✓ |

_4/5 arms within ±0.05._
