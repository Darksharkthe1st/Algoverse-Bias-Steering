"""Batch coordinator — run experiments back-to-back across branches (arch roadmap §10).

Committed to git and intended to be **frozen** (stable-by-convention): all branches
carry identical bytes, so an in-place `git checkout` never meaningfully changes it,
and the running process holds its code in memory regardless. Its *state* lives in
the gitignored `_coordinator/` directory.

Model:
- `route.json` is the lever: an ordered list of `{branch, configs, push}` entries.
- For each entry: check out the branch, then run each config **one at a time** as a
  subprocess (so an OOM/segfault kills only that run — soft-land, continue).
- The coordinator is the **sole git writer**: it commits/pushes at each phase the
  run reports (via stdout sentinels) and finalizes after each run. Push is
  best-effort — a failure is logged, never fatal (local commits already persist).
- Batch-level restart: configs whose done-marker exists in `_coordinator/queue/`
  are skipped, so re-running the coordinator resumes where it left off.
- `control.json` (stop/skip) and `status.json` are the file-based control surface a
  supervising LLM drives without a live session.

This module imports no torch/openai and does not import the run engine — it only
launches `python -m src.bias_steer run <config>` as a subprocess.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import get_repo_root

# run() prints these on stdout at each persistence boundary (see cli.py `run`).
_PHASE_RE = re.compile(r"::bias-steer:phase:([^:]+):(.+)")


@dataclass
class RouteEntry:
    branch: str
    configs: list = field(default_factory=list)
    push: bool = True


# --------------------------------------------------------------------------- git

class GitOps:
    """Thin git wrapper scoped to one repo. Mutating ops that can legitimately fail
    (nothing-to-commit, no remote) return a bool rather than raising, so the
    coordinator can soft-land them."""

    def __init__(self, repo_dir):
        self.repo = str(repo_dir)

    def _git(self, *args, check=False):
        return subprocess.run(["git", *args], cwd=self.repo,
                              capture_output=True, text=True, check=check)

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def checkout(self, branch: str) -> None:
        r = self._git("checkout", branch)
        if r.returncode != 0:
            raise RuntimeError(f"git checkout {branch} failed: {r.stderr.strip()}")

    def add_commit(self, pathspec: str, message: str) -> bool:
        """Stage `pathspec` and commit. Returns False if there was nothing to
        commit (not an error)."""
        self._git("add", "-A", "--", pathspec)
        r = self._git("commit", "-m", message)
        return r.returncode == 0

    def push(self, branch: str) -> bool:
        """Best-effort push; returns success. Never raises (network/auth/no-remote
        failures are expected and soft-landed)."""
        return self._git("push", "origin", branch).returncode == 0


# ------------------------------------------------------------------------ runner

def _subprocess_runner(config, runs_dir, on_phase, repo) -> int:
    """Launch `python -m src.bias_steer run <config>` and stream its stdout,
    invoking `on_phase(name, run_id)` on each phase sentinel. Returns the exit code.
    A crash (OOM/segfault) kills only this subprocess."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.bias_steer", "run", str(config), "--runs-dir", str(runs_dir)],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        sys.stdout.write(line)  # echo so humans watching the coordinator see progress
        m = _PHASE_RE.match(line.strip())
        if m:
            on_phase(m.group(1), m.group(2))
    proc.wait()
    return proc.returncode


# ------------------------------------------------------------------- coordinator

class Coordinator:
    def __init__(self, repo_dir=None, coord_dir=None, runner=None, git=None, runs_dir="runs"):
        self.repo = Path(repo_dir or get_repo_root())
        self.coord = Path(coord_dir) if coord_dir else self.repo / "_coordinator"
        self.runner = runner or _subprocess_runner
        self.git = git or GitOps(self.repo)
        self.runs_dir = runs_dir
        self.queue_done = self.coord / "queue" / "done"
        self.queue_failed = self.coord / "queue" / "failed"
        self.queue_done.mkdir(parents=True, exist_ok=True)
        self.queue_failed.mkdir(parents=True, exist_ok=True)

    # -- control plane ------------------------------------------------------
    def load_route(self) -> list[RouteEntry]:
        data = json.loads((self.coord / "route.json").read_text())
        return [RouteEntry(branch=e["branch"], configs=list(e.get("configs", [])),
                           push=e.get("push", True)) for e in data]

    def _control(self):
        path = self.coord / "control.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text()).get("command")
        except (json.JSONDecodeError, OSError):
            return None

    def _clear_control(self):
        (self.coord / "control.json").write_text(json.dumps({"command": None}))

    def _status(self, **fields):
        (self.coord / "status.json").write_text(json.dumps(fields, indent=2))

    @staticmethod
    def _key(branch: str, config: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{branch}__{config}").strip("-")

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        route = self.load_route()
        total = sum(len(entry.configs) for entry in route)  # queue size across all branches
        pos = 0
        for entry in route:
            if self._control() == "stop":
                self._status(state="stopped")
                return

            # Commit any stragglers on the current branch, then switch. The
            # gitignored _coordinator/ state rides across the checkout untouched.
            self.git.add_commit(self.runs_dir, "checkpoint before branch switch")
            try:
                self.git.checkout(entry.branch)
            except RuntimeError as e:
                self._status(state="error", branch=entry.branch, error=str(e))
                continue

            for config in entry.configs:
                pos += 1  # 1-based position in the whole queue (skipped items still count)
                key = self._key(entry.branch, config)
                if (self.queue_done / key).exists():
                    continue  # batch-restart: already completed

                cmd = self._control()
                if cmd == "stop":
                    self._status(state="stopped", branch=entry.branch)
                    return
                if cmd == "skip":
                    self._clear_control()
                    continue

                self._run_config(entry, config, key, pos=pos, total=total)

    def _run_config(self, entry: RouteEntry, config: str, key: str, *,
                    pos: int = 1, total: int = 1) -> None:
        # One-time run-level banner so it's obvious which queue item is running
        # ("3 of 5"). Queue position is a per-run fact, not a per-line one — a banner
        # answers "which experiment am I on" without prefixing every echoed line (which
        # would fight tqdm's \r waterfall and risk breaking phase-sentinel detection).
        print(f"=== [{pos}/{total}] {entry.branch} / {config} ===", flush=True)
        self._status(state="running", branch=entry.branch, config=config, phase="start",
                     queue_pos=pos, queue_total=total)

        def on_phase(phase, run_id):
            self.git.add_commit(self.runs_dir, f"{config} - {phase} ({run_id})")
            pushed = self.git.push(entry.branch) if entry.push else None
            self._status(state="running", branch=entry.branch, config=config,
                         phase=phase, run_id=run_id, pushed=pushed,
                         queue_pos=pos, queue_total=total)

        try:
            code = self.runner(config, self.runs_dir, on_phase, self.repo)
        except Exception as e:  # noqa: BLE001 - a launch failure must not kill the batch
            self._status(state="error", branch=entry.branch, config=config, error=str(e),
                         queue_pos=pos, queue_total=total)
            code = 1

        # Finalize: capture any straggler files so the tree is clean before the next
        # checkout, and record done/failed for batch-restart.
        self.git.add_commit(self.runs_dir, f"{config} - finalize")
        if entry.push:
            self.git.push(entry.branch)
        marker = self.queue_done if code == 0 else self.queue_failed
        (marker / key).write_text(str(code))
        self._status(state=("done" if code == 0 else "failed"),
                     branch=entry.branch, config=config, exit_code=code,
                     queue_pos=pos, queue_total=total)
