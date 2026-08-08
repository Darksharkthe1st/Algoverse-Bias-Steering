"""Phase 0 verification: schema, config, registry, tracking.

Runs with plain Python (no pytest required):

    python3 tests/test_phase0.py

...and is also collectible by pytest if it's installed. Stdlib only.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Make the repo root importable (so `import src.bias_steer` works when this
# script is run directly, whatever the cwd).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer import (  # noqa: E402
    Example, Result, CONDITIONS, STEERED_POS,
    ExperimentConfig, ModelSpec, DatasetSpec, SampleSpec, JudgeSpec, Coeffs,
    from_dict, registry, tracking,
)


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        label="crows neutrality",
        models=["qwen-7b"],
        dataset=DatasetSpec(name="crows", path="datasets/Crows_Pairs/x.csv", train_split=0.5),
        judge=JudgeSpec(name="neutrality"),
        coeffs=Coeffs(opinion=13, neutral=15),
        sample=SampleSpec(filter={"category": ["Age"]}, per_group=("category", 50), seed=7),
    )


def test_schema_defaults():
    ex = Example(id="e1", prompt="hello")
    assert ex.metadata == {} and ex.prompt == "hello"
    r = Result(example_id="e1", condition=STEERED_POS, response="yes", verdict="opinionated")
    assert r.condition in CONDITIONS
    # metadata dicts are independent instances, not a shared default
    Example(id="a", prompt="p").metadata["k"] = 1
    assert Example(id="b", prompt="p").metadata == {}


def test_config_roundtrip_through_json():
    cfg = _make_config()
    restored = from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored == cfg, "config did not survive a JSON round-trip"
    # the tuple field specifically (JSON turns it into a list)
    assert restored.sample.per_group == ("category", 50)
    assert isinstance(restored.sample.per_group, tuple)


def test_config_structural_validation():
    _make_config().validate()  # valid -> no raise

    bad_split = _make_config()
    bad_split.dataset.train_split = 1.5
    _expect(ValueError, bad_split.validate)

    no_models = _make_config()
    no_models.models = []
    _expect(ValueError, no_models.validate)


def test_registry_register_and_validate():
    # isolate the module-level registries for this test
    for reg in (registry.DATASETS, registry.MODELS, registry.METHODS, registry.JUDGES):
        reg.clear()

    cfg = _make_config()
    # nothing registered yet -> validate lists every missing component
    err = _expect(KeyError, lambda: registry.validate(cfg))
    for token in ("model 'qwen-7b'", "dataset 'crows'", "method 'mean_diff'", "judge 'neutrality'"):
        assert token in str(err), f"missing component not reported: {token}"

    # decorator form
    @registry.register(registry.DATASETS, "crows")
    def _load(spec):
        return []

    # direct form (for specs)
    registry.register(registry.MODELS, "qwen-7b", ModelSpec("qwen-7b", "Qwen/Qwen1.5-7B-Chat", True, "7B"))
    registry.register(registry.METHODS, "mean_diff", object())
    registry.register(registry.JUDGES, "neutrality", lambda resp, ex, spec: [])

    registry.validate(cfg)  # now fully registered -> no raise

    # double registration is rejected
    _expect(ValueError, lambda: registry.register(registry.DATASETS, "crows", _load))


def test_make_run_id_is_readable_and_deterministic():
    rid = tracking.make_run_id("crows neutrality", "Qwen/Qwen1.5-7B-Chat", when="20260101-120000")
    assert rid == "20260101-120000_crows-neutrality_Qwen1.5-7B-Chat", rid
    # deterministic given a fixed timestamp
    assert rid == tracking.make_run_id("crows neutrality", "Qwen/Qwen1.5-7B-Chat", when="20260101-120000")


def test_git_sha_in_this_repo():
    sha, dirty = tracking.git_sha()
    assert isinstance(sha, str) and len(sha) >= 7 and sha != "unknown", sha
    assert isinstance(dirty, bool)


def test_open_run_writes_manifest():
    cfg = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        handle = tracking.open_run(cfg, model="qwen-7b", runs_dir=tmp, when="20260101-120000")
        assert handle.dir.is_dir()
        assert (handle.dir / "logs").is_dir()
        manifest = json.loads((handle.dir / "manifest.json").read_text())
        assert manifest["run_id"] == handle.run_id
        assert manifest["model"] == "qwen-7b"
        assert "sha" in manifest["git"] and "dirty" in manifest["git"]
        # the manifest's config round-trips back to an equal ExperimentConfig
        assert from_dict(manifest["config"]) == cfg


def test_append_index_creates_header_then_appends():
    cfg = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index.csv"
        sha, dirty = "abc123", True
        for i in range(2):
            row = tracking.index_row(cfg, "qwen-7b", f"run{i}", sha, dirty, "20260101-120000")
            tracking.append_index(index, row)
        lines = index.read_text().strip().splitlines()
        assert lines[0].split(",") == tracking.INDEX_COLUMNS  # header once
        assert len(lines) == 3                                # header + 2 rows
        assert "run0" in lines[1] and "run1" in lines[2]


def _expect(exc_type, fn):
    """Assert `fn()` raises `exc_type`; return the exception for inspection."""
    try:
        fn()
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001 - test runner surfaces all failures
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
