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

# Phase 1 science. Importing these populates the registries (DATASETS / MODELS /
# METHODS / JUDGES) as a side effect. They lazy-import torch/openai, so this stays
# safe on a machine without the ML stack.
from . import datasets
from . import models
from . import steering
from . import judge
from .datasets import sample
from .steering import SteeringMethod
from .models import LoadedModel, load_model
from .judge import parse_verdict

# Refusal-direction repro (arXiv:2406.11717): loaders for the paper's published
# steering vectors. Import-safe without torch (lazy-imported at load time).
from . import refusal
from .refusal import RefusalDirection, load_refusal_direction

# Phase 2: wiring + persistence.
from . import artifacts
from . import logs
from . import metrics
from . import experiment
from . import experiment_refusal
from . import cli
from . import coordinator
from .experiment import run, Backend, RunResult
from .experiment_refusal import run_refusal, RefusalBackend, RefusalRunResult
from .coordinator import Coordinator, RouteEntry, GitOps

__all__ = [
    # schema
    "Example", "Result", "CONDITIONS", "INITIAL", "STEERED_POS", "STEERED_NEG",
    # config
    "ExperimentConfig", "ModelSpec", "DatasetSpec", "SampleSpec", "JudgeSpec",
    "Coeffs", "from_dict", "DEFAULT_SYS", "DEFAULT_JUDGE_RUBRIC",
    # science
    "datasets", "models", "steering", "judge",
    "sample", "SteeringMethod", "LoadedModel", "load_model", "parse_verdict",
    "refusal", "RefusalDirection", "load_refusal_direction",
    # wiring + persistence
    "artifacts", "logs", "metrics", "experiment", "cli",
    "run", "Backend", "RunResult",
    # refusal-direction repro (arXiv:2406.11717)
    "experiment_refusal", "run_refusal", "RefusalBackend", "RefusalRunResult",
    # batch coordinator
    "coordinator", "Coordinator", "RouteEntry", "GitOps",
    # infra modules
    "registry", "tracking",
]
