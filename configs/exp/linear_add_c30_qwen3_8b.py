"""Linear-schedule UNCONDITIONAL additive steering, coeff=30 — confound-isolation
counterpart to `adaptive_add_linear_c30_qwen3_8b.py` (default denom). THE
LOAD-BEARING POINT of this sweep.

Tests `linear_add` (`steering.apply_linear_add_perlayer`): ALWAYS adds
`increment_L = coeff * L / n_layers` along r̂_L, no clamp. Same target formula /
unit direction / default denom (n_layers=36 -> deepest-layer increment = 30.0)
as `adaptive_add_linear` at this coeff; only the one-sided clamp is removed.

CLEAN (B)-ISOLATION PAIR — the whole reason this branch exists: compare against
`runs/20260903-021811_adaptive-add-linear-c30-qwen3-8b_...`, where
`adaptive_add_linear` hit POS 138/148 (93%, near-saturation), NEG 42/52 (81%)
BUT showed a qualitative coherence shift — many STEERED+ responses emitted an
empty `<think>\n</think>` (reasoning skipped) then a blunt, sometimes
non-sequitur answer (not repetition-loop degeneracy; every sentence is real
English — a real quality caveat, GPU_RUN_LOG.md). The question this run answers:
does the coeff=30 coherence caveat reproduce under the UNCONDITIONAL linear ramp
(=> the linear schedule / raw dose magnitude drives it), or does it vanish
without the clamp (=> the state-dependent one-sidedness drives it)?

>>> MANUAL COHERENCE CHECK REQUIRED on this run's logs/eval.txt <<< — read the
actual STEERED+ generations, do not trust the summary counts alone (a big
judged effect can hide skipped-reasoning / non-sequitur text). Same manual
review the user did for the adaptive coeff=30 run; no automated heuristic.

Third arm: `fixed_add` (`mean_diff` coeff=8: POS 66/146, NEG 45/54).
See docs/SCOPE_linear_scaling_isolation.md.

    python -m src.bias_steer run configs/exp/linear_add_c30_qwen3_8b.py \
        --vector runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors
"""

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
)

config = ExperimentConfig(
    label="linear-add-c30 qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path="datasets/Snapshots/log_103_comparison_200.json",
        train_split=0.5, shuffle=False,
    ),
    sample=SampleSpec(),
    judge=JudgeSpec(name="neutrality"),
    # Default denom=n_layers -> increment_L = 30*L/36, reaching 30.0 at the last layer.
    coeffs=Coeffs(opinion=30.0, neutral=30.0),
    method="linear_add",
    max_tokens=128,
    batch_size=16,
)

config.vector_path = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"
