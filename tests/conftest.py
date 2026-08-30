"""Cross-test state guard.

The component registries in `src/bias_steer/registry.py` are process-global
dicts, populated once at import time. A test that mutates them without
restoring does not fail — it silently changes what *every later test* sees.

That is not hypothetical. `test_phase0.test_registry_register_and_validate`
cleared all four registries and never restored them, so the suite passed
file-by-file and failed 17 of 110 under a normal `pytest -q`. The dashboard ran
pytest once per file and therefore published a green suite the repo could not
reproduce.

This fixture does two things, and the split matters:

* **Every test is rolled back to the registries it started with.** Cross-test
  contamination becomes structurally impossible rather than merely detected, so
  the suite's result no longer depends on collection order. Tests that register
  a fake component (`p3model`, `faketest`, ...) are a normal, harmless pattern
  under that guarantee, and are rolled back without complaint.

* **Removing or rebinding a component that was already registered fails the
  test**, naming the keys. That is the destructive pattern that caused this bug,
  and it is never a legitimate way to get isolation — snapshot and restore in a
  `finally` instead. Rolling that case back silently would hide exactly the
  contamination we are trying to surface.
"""

import pytest

from src.bias_steer import registry

_REGISTRIES = ("DATASETS", "MODELS", "METHODS", "JUDGES")


@pytest.fixture(autouse=True)
def _registries_are_not_leaked():
    before = {name: dict(getattr(registry, name)) for name in _REGISTRIES}
    yield
    clobbered = []
    for name in _REGISTRIES:
        reg = getattr(registry, name)
        was, now = before[name], dict(reg)
        removed = sorted(set(was) - set(now))
        rebound = sorted(k for k in set(was) & set(now) if was[k] is not now[k])
        if removed or rebound:
            clobbered.append(f"{name}: removed={removed} rebound={rebound}")
        # Roll back unconditionally, so no test's registrations — legitimate or
        # not — can change what a later test sees.
        reg.clear()
        reg.update(was)

    if clobbered:
        pytest.fail(
            "test removed or rebound already-registered components instead of "
            "snapshotting and restoring them in a finally. This is the pattern "
            "that made the suite pass per-file and fail under normal collection. "
            "Clobbered -> " + "; ".join(clobbered),
            pytrace=False,
        )
