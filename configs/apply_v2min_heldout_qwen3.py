"""Held-out eval of the minimized V2 vector (stance <- soft-refusal) on qwen3-8b.

Phase-4 companion to configs/extract_v2min_issuebench_qwen3.py. Applies the V2
vector to the EXACT held-out prompts that extraction set aside (`test_split.json`),
generating INITIAL / STEERED_POS (+opinion dose) / STEERED_NEG (-neutral dose) and
labelling each with the FULL 9-way judge v2.1 — so the eval keeps the complete
behaviour breakdown, not a binary collapse.

Fill VECTOR_RUN below with the extraction run folder, then on the GPU box:

    python -m src.bias_steer run configs/apply_v2min_heldout_qwen3.py \
        --vector runs/<extract-id>/V2.safetensors

--vector overrides extraction: the snapshot's TRAIN half is folded into the eval set
(a supplied vector needs no fit), so ALL held-out prompts are scored.

NOTE: summary.md's "steering quality" line is meaningless here — it assumes a 2-label
judge, but this run uses the 9-way judge. Read results.csv instead, via
`python scripts/analyze_v2min_eval.py runs/<this-eval-id>/results.csv`, which
collapses+pools to the stance / soft-refusal / other view and prints the
initial->steered confusion (the per-example distributions the 2026 bar wants).

Doses: opinion/neutral coeffs start at 8 (the historical opinion-vector dose). Sweep
later if the frontier needs it.
"""

from src.bias_steer.config import ExperimentConfig, DatasetSpec, Coeffs, SampleSpec
from src.bias_steer.judges.v2 import judge_v2_spec

# The extraction run whose V2.safetensors + test_split.json this eval consumes.
# Point at the folder printed by `vectors`; --vector on the CLI names the .safetensors.
VECTOR_RUN = "runs/REPLACE_WITH_EXTRACT_RUN_ID"

config = ExperimentConfig(
    label="v2min eval V2 heldout qwen3-8b",
    models=["qwen3-8b"],
    dataset=DatasetSpec(
        name="snapshot",
        path=f"{VECTOR_RUN}/test_split.json",     # the exact held-out prompts
        train_split=0.5,                          # ignored: --vector -> all prompts evaluated
        shuffle=False,                            # preserve the recorded held-out order
    ),
    sample=SampleSpec(seed=0),
    judge=judge_v2_spec(model="gpt-4o-mini"),     # FULL 9-way v2.1 judge
    coeffs=Coeffs(opinion=8.0, neutral=8.0),
    method="mean_diff",
    max_tokens=512,
    batch_size=16,
    strip_reasoning=True,
    enable_thinking=False,
)
