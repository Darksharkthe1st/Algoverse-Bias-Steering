"""A queue runner that cannot hide a failure — notes/14 §3.

Run 1's `run()` returned 0 unconditionally and blanket-`pkill`ed after every
step, so a killed step and a successful one were indistinguishable.  The four
required properties, and where each is implemented:

  1. Real exit codes, written to a machine-readable manifest    `Step.exit_code`
  2. No blanket pkill; only the step's own PID is signalled      `_run_subprocess`
  3. Declared expected outputs, checked on completion            `Step.produces`
  4. A post-run verifier over the manifest                       `verifier.py`

The pilot's steps are in-process callables; the real run's are subprocesses.
Both go through the same `Step`, so the manifest the verifier reads is produced
by the same code in both cases.  A pilot that exercised a different runner than
the real run would not be a pilot.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field


@dataclass
class Step:
    """One unit of work, with the files it is REQUIRED to produce.

    `produces` is not documentation.  A step that exits 0 without writing its
    declared outputs is marked INCOMPLETE and fails the queue — which is the
    specific failure run 1 could not detect.
    """
    name: str
    produces: list = field(default_factory=list)
    fn: object = None            # callable() -> None, for in-process steps
    argv: list = field(default_factory=list)   # for subprocess steps

    # filled in by the runner
    status: str = "PENDING"
    exit_code: int | None = None
    started_at: float | None = None
    duration_s: float | None = None
    error: str | None = None
    missing_outputs: list = field(default_factory=list)


def _run_subprocess(step: Step, cwd: str) -> int:
    """Run one step as a child process, tracking only its own PID.

    `subprocess.Popen` + `wait()` on that single handle is the whole fix for
    run 1's blanket `pkill`: nothing here can signal a process it did not start.
    """
    p = subprocess.Popen(step.argv, cwd=cwd)
    try:
        return p.wait()
    except BaseException:
        p.kill()          # only this PID, never a pattern match
        p.wait()
        raise


def run_queue(steps: list, *, out_dir: str, cwd: str = ".",
              manifest_name: str = "queue_manifest.json",
              stop_on_failure: bool = True) -> dict:
    """Execute steps in order, writing the manifest after EVERY step.

    Written after every step, not at the end, so a queue that dies mid-run still
    leaves an accurate record of what completed — the same reasoning as the
    10-minute continuous sync in notes/14 §1.3.  State that exists in exactly one
    volatile place is the root cause behind S5.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, manifest_name)
    started = time.time()

    def _write():
        payload = {
            "started_at": started,
            "elapsed_s": time.time() - started,
            "n_steps": len(steps),
            "steps": [{
                "name": s.name, "status": s.status, "exit_code": s.exit_code,
                "duration_s": s.duration_s, "error": s.error,
                "produces": s.produces, "missing_outputs": s.missing_outputs,
            } for s in steps],
            "all_ok": all(s.status == "OK" for s in steps),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    _write()
    for step in steps:
        step.started_at = time.time()
        try:
            if step.fn is not None:
                step.fn()
                step.exit_code = 0
            else:
                step.exit_code = _run_subprocess(step, cwd)
        except Exception:
            step.exit_code = 1
            step.error = traceback.format_exc(limit=6)
        step.duration_s = time.time() - step.started_at

        # Declared outputs are checked IMMEDIATELY, before the queue advances.
        step.missing_outputs = [p for p in step.produces if not os.path.exists(p)]
        if step.exit_code != 0:
            step.status = "FAILED"
        elif step.missing_outputs:
            step.status = "INCOMPLETE"
        else:
            step.status = "OK"

        _write()
        if step.status != "OK" and stop_on_failure:
            break

    return _write()
