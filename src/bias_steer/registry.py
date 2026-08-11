"""Name -> component registries (arch roadmap §3).

Plain dicts, not a class hierarchy. Adding a dataset / model / steering method /
judge is one function (or spec) plus one `register` line, with zero downstream
change. `validate` fails fast when a config names something unregistered.

Registries are populated by later phases:
- DATASETS / MODELS / METHODS / JUDGES fill up as datasets.py, models.py,
  steering.py, judge.py are built (Phase 1). In Phase 0 they start empty.
"""

from typing import Callable

from .config import ModelSpec, ExperimentConfig

# A dataset loader: DatasetSpec -> list[Example].
DATASETS: dict[str, Callable] = {}
# A model declared as data (§3.4).
MODELS: dict[str, ModelSpec] = {}
# A steering technique. Its concrete type lands in Phase 1 (steering.py); kept as
# a bare object here so Phase 0 has no dependency on it.
METHODS: dict[str, object] = {}
# A judge: (responses, examples, JudgeSpec) -> list[label].
JUDGES: dict[str, Callable] = {}


def register(registry: dict, name: str, value=None):
    """Register `value` under `name`, or return a decorator if `value` is omitted.

    Direct (for specs)::

        register(MODELS, "qwen-7b", ModelSpec(...))

    Decorator (for functions)::

        @register(DATASETS, "bbq")
        def load_bbq(spec): ...
    """

    def _set(v):
        if name in registry:
            raise ValueError(f"{name!r} is already registered")
        registry[name] = v
        return v

    return _set if value is None else _set(value)


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    """Check every component a config names is actually registered.

    This is the "fail fast on desync" guard: it turns a typo or a forgotten
    registration into a clear error up front rather than a confusing failure
    deep in a run. Returns the config for chaining.
    """
    missing: list[str] = []
    for m in cfg.models:
        if m not in MODELS:
            missing.append(f"model {m!r}")
    if cfg.dataset.name not in DATASETS:
        missing.append(f"dataset {cfg.dataset.name!r}")
    if cfg.method not in METHODS:
        missing.append(f"method {cfg.method!r}")
    if cfg.judge.name not in JUDGES:
        missing.append(f"judge {cfg.judge.name!r}")

    if missing:
        raise KeyError(
            "Unregistered components: "
            + "; ".join(missing)
            + f". Registered: models={sorted(MODELS)}, datasets={sorted(DATASETS)}, "
            + f"methods={sorted(METHODS)}, judges={sorted(JUDGES)}"
        )
    return cfg
