# `tests/test_phase2.py` starts a real coordinator campaign on the Lambda box

**Date:** 2026-08-16 · **Box:** Lambda A100-40GB · **Status:** diagnosed, **not fixed in code**

## What happens

`tests/test_phase2.py::test_cli_queue_requires_a_route_file` calls:

```python
assert cli.main(["run", "--queue"]) == 2
```

Its comment states the precondition: *"The repo has no route file by default."*
**That precondition is false on the Lambda box**, which carries a machine-local,
gitignored `_coordinator/route.json` from the campaign queued 2026-08-09.

So on this box the test does not exercise an error path — it **drains the real
queue**. Observed on 2026-08-16:

1. The coordinator committed the in-progress working tree as `aaaa455`
   *"checkpoint before branch switch"*.
2. It checked out away from the working branch:
   `fk/init-refusal-rewrite` → `exp/coeff-sweep` → `exp/anchors`.
3. It started a **Qwen-14B** run on the GPU (`runs/20260816-011401_anchor-qwen-14b_qwen-14b/`).
4. The test returned 1 instead of 2, so the assertion failed.
5. The campaign died when the test process exited, leaving
   `_coordinator/status.json` stuck at `state: "running"`.

A second invocation reproduced it and hung for the full 2-minute command timeout.

## Why it matters

It is silent and it is destructive of *context*, not data: a test run mid-task
moves you to a different branch and burns GPU on an unrelated 14B model. The
failure mode looks like "someone else is using the repo."

## Recovery (verified)

Nothing was lost, but the recovery is non-obvious:

- **Uncommitted work survives.** Git carries modified files across the checkout;
  `git stash` also holds them. Recover the branch from `git reflog`.
- **Queue markers are untouched** — `queue/done/` and `queue/failed/` kept their
  2026-08-09 timestamps. The qwen-14b entry was already in `failed/` from the
  original campaign, so the queue stayed consistent.
- **Only `status.json` needs correcting.** It was reset to a truthful terminal
  state with a note explaining the interruption.
- The aborted 14B run dir is left untracked for the repo owner to triage.

## Fix

Point the test at a temp dir instead of the repo's `_coordinator/`, so the
"no route file" precondition is created rather than assumed. Until then:

> **Do not run `tests/test_phase2.py` on the Lambda box.** Run the other suites
> individually: `test_phase0`, `test_phase1`, `test_phase3`, `test_phase4`,
> `test_refusal`, `test_refusal_extract`, `test_refusal_datasets`,
> `test_refusal_grid_provenance` (84 tests, all passing as of this date).
