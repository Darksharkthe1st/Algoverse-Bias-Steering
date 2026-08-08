"""bias_steer — bias-steering experiment pipeline.

Phase 0 (foundations): the shared data contract, typed config, component
registries, and run tracking. See docs/02-architecture-roadmap.md.

Later phases add the science (datasets, models, steering, judge), the run
wiring, and the batch coordinator.
"""

from .schema import Example, Result, CONDITIONS, INITIAL, STEERED_POS, STEERED_NEG
from .config import (
    ExperimentConfig,
    ModelSpec,
    DatasetSpec,
    SampleSpec,
    JudgeSpec,
    Coeffs,
    from_dict,
    DEFAULT_SYS,
    DEFAULT_JUDGE_RUBRIC,
)
from . import registry
from . import tracking

__all__ = [
    # schema
    "Example", "Result", "CONDITIONS", "INITIAL", "STEERED_POS", "STEERED_NEG",
    # config
    "ExperimentConfig", "ModelSpec", "DatasetSpec", "SampleSpec", "JudgeSpec",
    "Coeffs", "from_dict", "DEFAULT_SYS", "DEFAULT_JUDGE_RUBRIC",
    # modules
    "registry", "tracking",
]
