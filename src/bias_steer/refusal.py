"""Loading the paper's pre-computed refusal directions (Arditi et al., 2024).

We test the *published* refusal vectors inside this framework rather than
re-deriving them. This module reads the third-party artifacts fetched by
`scripts/fetch_refusal_artifacts.py` into `third_party/refusal_direction/runs/`:
each model ships a raw `direction.pt` of shape `(d_model,)` plus a
`direction_metadata.json` giving the `(layer, pos)` it was extracted from.

Conventions matching the rest of the package:
- torch is lazy-imported inside `load_refusal_direction`, so importing this
  module stays safe without the ML stack.
- Paths resolve *package-relative* (not via `utils.get_repo_root`, which looks
  for a `.git` directory and would escape a git worktree, where `.git` is a file).

The direction is returned **raw** (un-normalized): directional ablation
normalizes it internally (`r̂`), while activation-addition uses the raw vector at
`coeff=±1` (its natural norm is the dose). See the ablation/act-add methods in
`steering.py` and the paper's `pipeline/utils/hook_utils.py`.
"""

from dataclasses import dataclass
from pathlib import Path
import json

# Upstream run-dir name -> this repo's MODEL_CATALOG key (see models.py).
RUN_DIR_TO_MODEL = {
    "qwen-1_8b-chat": "qwen-1.8b",
    "gemma-2b-it": "gemma-2b",
    "yi-6b-chat": "yi-6b",
    "meta-llama-3-8b-instruct": "llama3-8b",
    "llama-2-7b-chat-hf": "llama-2-7b",
}
MODEL_TO_RUN_DIR = {v: k for k, v in RUN_DIR_TO_MODEL.items()}

# third_party/refusal_direction/runs/ lives at the repo/worktree root, two levels
# up from this file (src/bias_steer/refusal.py -> src -> root).
_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "refusal_direction" / "runs"


@dataclass
class RefusalDirection:
    """One model's published refusal direction plus where it came from."""

    model_key: str      # key into MODELS (e.g. "qwen-1.8b")
    run_dir: str        # upstream run-dir name (e.g. "qwen-1_8b-chat")
    layer: int          # source layer the direction was extracted at
    pos: int            # source token position (negative index into the prompt)
    direction: object   # torch.Tensor, shape (d_model,), float32, raw (un-normalized)

    @property
    def d_model(self) -> int:
        return int(self.direction.shape[-1])


def artifact_dir(run_dir: str) -> Path:
    """Local directory holding one model's fetched artifacts."""
    return _ARTIFACT_ROOT / run_dir


def _resolve(model: str) -> tuple[str, str]:
    """Accept either a catalog key or an upstream run-dir name -> (run_dir, model_key)."""
    if model in RUN_DIR_TO_MODEL:
        return model, RUN_DIR_TO_MODEL[model]
    if model in MODEL_TO_RUN_DIR:
        return MODEL_TO_RUN_DIR[model], model
    known = sorted(RUN_DIR_TO_MODEL) + sorted(MODEL_TO_RUN_DIR)
    raise KeyError(f"unknown refusal model {model!r}; known keys: {known}")


def available_run_dirs() -> list[str]:
    """Run dirs whose `direction.pt` has been fetched locally (may be empty)."""
    if not _ARTIFACT_ROOT.exists():
        return []
    return sorted(
        d.name for d in _ARTIFACT_ROOT.iterdir()
        if d.is_dir() and (d / "direction.pt").exists()
    )


def load_refusal_direction(model: str) -> RefusalDirection:
    """Load one direction by catalog key ("qwen-1.8b") or run-dir ("qwen-1_8b-chat").

    Raises FileNotFoundError (with the fetch command) if the artifact is absent.
    """
    import torch

    run_dir, model_key = _resolve(model)
    d = artifact_dir(run_dir)
    dpt = d / "direction.pt"
    if not dpt.exists():
        raise FileNotFoundError(
            f"missing {dpt}\nFetch it first:\n"
            f"    python scripts/fetch_refusal_artifacts.py --model {run_dir}"
        )
    meta = json.loads((d / "direction_metadata.json").read_text())
    # The artifacts are plain tensors; weights_only=True is safe and avoids
    # executing pickle. Fall back for torch versions predating the kwarg.
    try:
        tensor = torch.load(dpt, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(dpt, map_location="cpu")
    direction = tensor.to(torch.float32).flatten()
    return RefusalDirection(
        model_key=model_key,
        run_dir=run_dir,
        layer=int(meta["layer"]),
        pos=int(meta["pos"]),
        direction=direction,
    )


def load_all() -> list[RefusalDirection]:
    """Load every locally-fetched direction (skips models not yet fetched)."""
    return [load_refusal_direction(rd) for rd in available_run_dirs()]
