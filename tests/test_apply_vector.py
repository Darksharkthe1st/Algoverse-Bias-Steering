"""run() with a supplied vector — apply an existing direction instead of extracting.

    python3 tests/test_apply_vector.py

There is ONE run path. `run(config, vector_path=...)` skips extraction and
evaluates the supplied vector; `run(config)` extracts from TRAIN as usual. Both
share the eval+persist tail. Torch-free: a fake Backend (incl. `load_vector` and
`generate_with_cache`) + fake method/judge/dataset/model.
"""

import csv
import json
import os
import sys
import tempfile
import types
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402,F401
from src.bias_steer import registry, experiment  # noqa: E402
from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs, ModelSpec  # noqa: E402
from src.bias_steer.schema import Example, INITIAL, STEERED_POS, STEERED_NEG  # noqa: E402


class _StubVector:
    """(n_layers, d_model)-shaped stand-in: passes assert_steering_shape, .to() no-op."""
    shape = (2, 4)
    def to(self, device):
        return self


class _FakeMethod:
    name = "applyfake"
    def capture(self, cache, n_layers): return ("resid", n_layers)
    def build(self, resids_by_label, contrast): return _StubVector()
    def apply(self, model, vector, coeff): return [("sign", 1 if coeff >= 0 else -1)]
    def names(self, n_layers): return [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]


def _register_fakes():
    if "applyfake" not in registry.DATASETS:
        registry.register(
            registry.DATASETS, "applyfake",
            lambda spec: [Example(f"e{i}", f"prompt {i}", {"category": "X"}) for i in range(6)],
        )
    if "applyfake" not in registry.MODELS:
        registry.register(registry.MODELS, "applyfake", ModelSpec("applyfake", "fake/model", True, "S"))
    if "applyfake" not in registry.METHODS:
        registry.register(registry.METHODS, "applyfake", _FakeMethod())
    if "applyfake" not in registry.JUDGES:
        registry.register(registry.JUDGES, "applyfake", lambda responses, examples, spec: list(responses))


def _fake_backend(calls):
    def load(spec):
        model = types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2, d_model=4))
        return types.SimpleNamespace(model=model, tokenizer=None, spec=spec, device="cpu")

    def generate(loaded, prompts, mnt, sysp):
        return ["opinionated"] * len(prompts)

    def generate_with_hooks(loaded, prompts, hooks, mnt, sysp):
        sign = hooks[0][1] if hooks else 1
        return [("opinionated" if sign > 0 else "neutral")] * len(prompts)

    def generate_with_cache(loaded, prompts, mnt, sysp, capture_names=None):
        calls["generate_with_cache"] += 1
        # alternate so both contrast buckets fill during extraction
        return (["opinionated" if i % 2 == 0 else "neutral" for i in range(len(prompts))],
                [None] * len(prompts))

    def save_vector(path, vector, *, n_layers, d_model):
        Path(path).write_text("fake-vector")

    def save_residuals(path, resids, *, n_layers, d_model):
        calls["save_residuals"] += 1
        Path(path).write_text("fake-residuals")

    def load_vector(path):
        calls["load_vector"] += 1
        return _StubVector()

    return experiment.Backend(
        load=load, generate=generate, generate_with_hooks=generate_with_hooks,
        generate_with_cache=generate_with_cache, save_vector=save_vector,
        save_residuals=save_residuals, load_vector=load_vector,
    )


def _cfg():
    return ExperimentConfig(
        label="apply fake",
        models=["applyfake"],
        dataset=DatasetSpec(name="applyfake", path="ignored"),  # 6 examples, split 3/3
        judge=JudgeSpec(name="applyfake", labels=["neutral", "opinionated"]),
        coeffs=Coeffs(opinion=8, neutral=8),
        sample=SampleSpec(seed=0),
        method="applyfake",
    )


def test_supplied_vector_skips_extraction_and_evals_all():
    _register_fakes()
    calls = {"generate_with_cache": 0, "save_residuals": 0, "load_vector": 0}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_cfg(), vector_path="some/vector.safetensors",
                           backend=_fake_backend(calls), runs_dir=tmp)[0]
        # extraction was skipped, vector was loaded
        assert calls["load_vector"] == 1
        assert calls["generate_with_cache"] == 0
        assert calls["save_residuals"] == 0
        # apply provenance present; residuals absent
        assert (r.dir / "applied_vector.json").is_file()
        assert not (r.dir / "residuals.safetensors").exists()
        prov = json.loads((r.dir / "applied_vector.json").read_text())
        assert prov["source_vector_path"] == "some/vector.safetensors"

        # whole sampled set evaluated (train folded in): 6 examples * 3 conditions
        with (r.dir / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 18
        assert {row["condition"] for row in rows} == {INITIAL, STEERED_POS, STEERED_NEG}
        assert r.counts[STEERED_POS] == {"opinionated": 6}
        assert r.counts[STEERED_NEG] == {"neutral": 6}

        with (Path(tmp) / "index.csv").open() as f:
            idx = list(csv.DictReader(f))
        assert idx[0]["status"] == "done" and idx[0]["n_test"] == "6"
        assert idx[0]["n_train"] in ("0", "")


def test_no_vector_extracts_as_usual():
    _register_fakes()
    calls = {"generate_with_cache": 0, "save_residuals": 0, "load_vector": 0}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_cfg(), backend=_fake_backend(calls), runs_dir=tmp)[0]
        # extraction ran; no vector loaded
        assert calls["generate_with_cache"] >= 1 and calls["save_residuals"] == 1
        assert calls["load_vector"] == 0
        assert (r.dir / "residuals.safetensors").is_file()
        assert not (r.dir / "applied_vector.json").exists()
        # eval on TEST only (3 examples) * 3 conditions
        with (r.dir / "results.csv").open() as f:
            assert len(list(csv.DictReader(f))) == 9


def test_vector_path_from_config():
    _register_fakes()
    cfg = _cfg()
    cfg.vector_path = "cfg/vec.safetensors"
    calls = {"generate_with_cache": 0, "save_residuals": 0, "load_vector": 0}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(cfg, backend=_fake_backend(calls), runs_dir=tmp)[0]
        assert calls["load_vector"] == 1  # config.vector_path honored
        prov = json.loads((r.dir / "applied_vector.json").read_text())
        assert prov["source_vector_path"] == "cfg/vec.safetensors"


def test_loud_warning_when_train_nonempty_and_vector_supplied():
    _register_fakes()
    calls = {"generate_with_cache": 0, "save_residuals": 0, "load_vector": 0}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_cfg(), vector_path="v.safetensors",
                           backend=_fake_backend(calls), runs_dir=tmp)[0]
        # train split (3 examples) is non-empty -> a loud WARNING is logged, not an error
        log_txt = (r.dir / "logs" / "run.log").read_text()
        assert "WARNING" in log_txt and "not used to fit" in log_txt


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
