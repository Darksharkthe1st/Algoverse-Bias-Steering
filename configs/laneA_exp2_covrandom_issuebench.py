"""Lane A Exp-2 — the covariance-matched random-direction control on IssueBench.

`docs/HANDOFF_lane_a.md` §Exp-2. Answers "it's the dose, not the direction": if a
vector matched to the opinion direction on per-layer norm and residual covariance
moves the opinion rate just as much, then the result is about how hard the residual
stream was pushed, not about a representation.

Identical to `configs/laneA_exp1_prompt_vs_steer_issuebench.py` in every respect that
could confound the comparison — same 200 held-out IssueBench items (same
`sample.seed`, same exclusion list), same dose, same `max_tokens`, same
`enable_thinking`, same judge — and differs in exactly one thing: `vector_path`
points at the control instead of the opinion direction. That makes the two runs
PAIRED per item, which is stronger than the handoff's design of separate dose sweeps,
because the comparison is then within-item rather than between-samples.

`intervention="steer"` (3 arms, not 5): the prompt arms would be byte-identical to
Exp-1's, since greedy decoding and an unsteered generation do not depend on which
vector sits unused on the GPU. Re-running them would spend a third of the GPU budget
reproducing rows we already have. The INITIAL arm IS re-run, deliberately: it must
come out identical to Exp-1's, which makes it a free end-to-end determinism check on
the whole harness.

## The dose

c=8, matching Exp-1's headline dose, so control and treatment are compared where the
treatment was actually measured. The handoff asks for doses {20, 30}; those come from
the `adaptive-add-linear-c20/c30/c40` ladder, and that ladder cannot supply a
reportable dose — all three runs used `max_tokens=128` with `enable_thinking` unset
(so ON for Qwen3-8B) and `strip_reasoning` off, and the coherence gate finds their
unsteered arm 98.5% truncated mid-`<think>`, i.e. no answer under the label at all
(HANDOFF §0.3; see `runs/LANE_A_RESULTS.md` Exp-3). Running a control at a dose whose
treatment number is invalid would produce a comparison to nothing. Recorded in
`OPEN_QUESTIONS.md` alongside the re-run the ladder needs.

## The control vector

Built by `scripts/build_covrandom_vector.py` (read its docstring — a covariance match
alone turned out NOT to give a random direction here, because the within-pole residual
covariance is nearly rank one at the middle layers and the opinion direction lies
along that one component). The vector applied here is the **orthogonalised**
covariance-matched draw: same per-layer norm, same anisotropic envelope, and exactly
zero component along the opinion direction (realised mean |cos| 0.0000).

    python scripts/build_covrandom_vector.py \
        --extract-run runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b \
        --out runs/laneA_exp2_covrandom_vector_seed0 --seed 0 --orthogonalise
    python -m src.bias_steer run configs/laneA_exp2_covrandom_issuebench.py

One seed is one draw from the control distribution. The provenance JSON records the
seed; a second draw is `--seed 1` and another run.
"""

import json

from src.bias_steer.config import (
    ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs, SampleSpec,
)
from src.utils import get_repo_root

_REPO = get_repo_root()

# The SAME exclusion artifact Exp-1 uses, so the two runs score the identical 200
# items. Read, never restated (one fact, one owner).
_EXCLUSIONS = json.loads(
    (_REPO / "datasets" / "laneA" / "issuebench_exclusions.json").read_text()
)

# The covariance-matched, opinion-orthogonal control. Shape-asserted (36, 4096) at
# build time AND again by `_load_provided_vector` before generation: a control that
# silently collapsed to a scalar would "pass" this experiment by doing nothing, which
# is the most flattering possible failure and therefore the one to guard hardest.
VECTOR = "runs/laneA_exp2_covrandom_vector_seed0/steering_vector.safetensors"

config = ExperimentConfig(
    label="laneA_exp2_covrandom_issuebench",
    models=["qwen3-8b"],
    dataset=DatasetSpec(name="issuebench", path="sample", train_split=0.5),
    judge=JudgeSpec(name="neutrality"),
    coeffs=Coeffs(opinion=8.0, neutral=8.0),      # Exp-1's dose, exactly
    sample=SampleSpec(
        limit=200,
        seed=0,                                   # same draw as Exp-1 => paired items
        exclude_ids=tuple(_EXCLUSIONS["heldout"]["exclude_ids"]),
        exclude={"template_id": _EXCLUSIONS["artifacts"]["exclude_template_ids"]},
    ),
    max_tokens=512,
    batch_size=32,
)
config.enable_thinking = False
config.intervention = "steer"    # INITIAL + STEERED_POS/NEG; prompt arms come from Exp-1
config.vector_path = VECTOR
