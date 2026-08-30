"""Run identity, manifest, and the run index (arch roadmap §6).

The traceability answer to "which experiment was this, and what code produced
it": a per-run `manifest.json` (full config + git SHA) and a browsable
`runs/index.csv`. Run IDs are readable slugs (no hashing) — we optimize for human
scanning, since resumability is a non-goal (§9).
"""

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..utils import get_current_time_str, get_repo_root
from .config import ExperimentConfig

# Columns of runs/index.csv. Headline metrics are appended by later phases; a
# Phase-0 row records the run's coordinates + status only.
INDEX_COLUMNS = [
    "run_id", "label", "model", "dataset", "method",
    "opin_coeff", "neut_coeff", "git_sha", "dirty", "timestamp", "status",
    # headline metrics filled in by a completed run (Phase 2); blank on creation.
    "n_train", "n_test", "opin_good", "neut_good",
]


@dataclass
class RunHandle:
    """Everything a caller needs after a run directory is opened."""

    run_id: str
    dir: Path
    model: str


def _slug(s: str) -> str:
    """Filesystem-safe slug; drops an org/ prefix (e.g. 'Qwen/Qwen1.5' -> 'Qwen1.5')."""
    s = s.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def make_run_id(label: str, model: str, when: str | None = None) -> str:
    """`<YYYYMMDD-HHMMSS>_<label>_<model>` — the readable run identity (§6)."""
    when = when or get_current_time_str()
    return f"{when}_{_slug(label)}_{_slug(model)}"


def git_sha(repo_dir=None) -> tuple[str, bool]:
    """Return (commit_sha, dirty). `dirty` is True if the working tree has
    uncommitted changes (§6.1). Falls back to ('unknown', False) off-git."""
    repo_dir = str(repo_dir or get_repo_root())
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", False


def open_run(
    cfg: ExperimentConfig,
    model: str,
    runs_dir="runs",
    when: str | None = None,
    repo_dir=None,
) -> RunHandle:
    """Create `runs/<run_id>/` (+ a `logs/` subdir) and write its manifest.

    The manifest captures the full config, the model this run targets, the git
    SHA + dirty flag, and the timestamp — the complete traceable record. Writing
    is atomic-ish via a single `write_text`; no partial-JSON concerns here since
    manifests are tiny and written once.
    """
    when = when or get_current_time_str()
    run_id = make_run_id(cfg.label, model, when)
    run_dir = Path(runs_dir) / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    sha, dirty = git_sha(repo_dir)
    # Resolve the short handle to what will actually be loaded. `model` on its
    # own is a repo-local nickname; PREREG §3b requires the hf_id and the
    # immutable revision to be in the record, or the run does not count as
    # evidence. Tolerant of unregistered handles so test doubles still open.
    from .registry import MODELS
    _spec = MODELS.get(model)
    manifest = {
        "run_id": run_id,
        "label": cfg.label,
        "model": model,
        "model_spec": {
            "name": getattr(_spec, "name", model),
            "hf_id": getattr(_spec, "hf_id", None),
            "revision": getattr(_spec, "revision", "") or None,
        },
        "timestamp": when,
        "git": {"sha": sha, "dirty": dirty},
        "config": cfg.to_dict(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return RunHandle(run_id=run_id, dir=run_dir, model=model)


def index_row(
    cfg: ExperimentConfig,
    model: str,
    run_id: str,
    sha: str,
    dirty: bool,
    when: str,
    status: str = "created",
) -> dict:
    """Build one `runs/index.csv` row from a config + run coordinates."""
    return {
        "run_id": run_id,
        "label": cfg.label,
        "model": model,
        "dataset": cfg.dataset.name,
        "method": cfg.method,
        "opin_coeff": cfg.coeffs.opinion,
        "neut_coeff": cfg.coeffs.neutral,
        "git_sha": sha,
        "dirty": dirty,
        "timestamp": when,
        "status": status,
    }


def append_index(index_path, row: dict) -> None:
    """Append `row` to `index.csv`, writing the header if the file is new.

    Uses a fixed column set (`INDEX_COLUMNS`); unknown keys are ignored and
    missing ones written blank, so callers never corrupt the header.
    """
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in INDEX_COLUMNS})
