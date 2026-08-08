"""Phase 2 verification: run() end-to-end wiring, metrics, logs, CLI helpers.

The end-to-end test injects a FAKE Backend + registers a fake method / judge /
dataset / model, so the entire orchestration runs with no torch / no OpenAI. The
numeric correctness of capture/build is covered separately by the (torch-gated)
Phase 1 steering tests.

    python3 tests/test_phase2.py
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

import src.bias_steer as bs  # noqa: E402
from src.bias_steer import registry, metrics, experiment, cli  # noqa: E402
from src.bias_steer.config import (  # noqa: E402
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs, ModelSpec,
)
from src.bias_steer.schema import Example, Result, INITIAL, STEERED_POS, STEERED_NEG  # noqa: E402


# ------------------------------------------------------------------ fakes

def _register_fakes():
    """Register fake components (idempotent) so run() has something to resolve."""
    if "faketest" not in registry.DATASETS:
        # 10 examples, one category, so train/test split = 5/5
        registry.register(
            registry.DATASETS, "faketest",
            lambda spec: [Example(f"e{i}", f"prompt {i}", {"category": "X"}) for i in range(10)],
        )
    if "faketest" not in registry.MODELS:
        registry.register(registry.MODELS, "faketest", ModelSpec("faketest", "fake/model", True, "S"))
    if "faketest" not in registry.METHODS:
        registry.register(registry.METHODS, "faketest", _FakeMethod())
    if "faketest" not in registry.JUDGES:
        # the fake generators emit the verdict string directly; judge is identity
        registry.register(registry.JUDGES, "faketest", lambda responses, examples, spec: list(responses))


class _FakeMethod:
    name = "faketest"

    def capture(self, cache, n_layers):
        return ("resid", n_layers)                       # opaque; never serialized for real

    def build(self, resids_by_label, contrast):
        return "VECTOR"                                  # opaque steering vector

    def apply(self, model, vector, coeff):
        return [("sign", 1 if coeff >= 0 else -1)]       # encode direction for the fake generator


def _fake_backend():
    def load(spec):
        model = types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2))
        return types.SimpleNamespace(model=model, tokenizer=None, spec=spec, device="cpu")

    def generate_with_cache(loaded, prompts, max_new_tokens, system_prompt):
        # alternate verdicts so both contrast buckets fill during training
        responses = ["opinionated" if i % 2 == 0 else "neutral" for i in range(len(prompts))]
        return responses, [None] * len(prompts)

    def generate(loaded, prompts, max_new_tokens, system_prompt):
        return ["opinionated"] * len(prompts)            # every INITIAL is opinionated

    def generate_with_hooks(loaded, prompts, fwd_hooks, max_new_tokens, system_prompt):
        sign = fwd_hooks[0][1] if fwd_hooks else 1
        label = "opinionated" if sign > 0 else "neutral"  # +coeff -> opinion, -coeff -> neutral
        return [label] * len(prompts)

    def save_vector(path, vector):
        Path(path).write_text("fake-vector")

    def save_residuals(path, resids_by_label):
        Path(path).write_text("fake-residuals")

    return experiment.Backend(
        load=load, generate=generate, generate_with_cache=generate_with_cache,
        generate_with_hooks=generate_with_hooks, save_vector=save_vector,
        save_residuals=save_residuals,
    )


def _fake_config():
    return ExperimentConfig(
        label="fake run",
        models=["faketest"],
        dataset=DatasetSpec(name="faketest", path="ignored", train_split=0.5),
        judge=JudgeSpec(name="faketest", labels=["neutral", "opinionated"]),
        coeffs=Coeffs(opinion=10, neutral=12),
        sample=SampleSpec(seed=0),
        method="faketest",
    )


# ------------------------------------------------------------------ end-to-end

def test_run_end_to_end_produces_all_artifacts():
    _register_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        results = experiment.run(_fake_config(), backend=_fake_backend(), runs_dir=tmp)
        assert len(results) == 1
        r = results[0]

        # artifacts on disk
        for fname in ("manifest.json", "results.csv", "summary.md",
                      "steering_vector.safetensors", "residuals.safetensors"):
            assert (r.dir / fname).is_file(), f"missing artifact: {fname}"
        for log in ("run.log", "train.txt", "eval.txt"):
            assert (r.dir / "logs" / log).is_file(), f"missing log: {log}"

        # manifest round-trips the config
        manifest = json.loads((r.dir / "manifest.json").read_text())
        assert manifest["model"] == "faketest"
        assert bs.from_dict(manifest["config"]).label == "fake run"

        # tidy results.csv: 5 test examples x 3 conditions
        with (r.dir / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 15
        conds = {row["condition"] for row in rows}
        assert conds == {INITIAL, STEERED_POS, STEERED_NEG}

        # index.csv row with headline metrics
        with (Path(tmp) / "index.csv").open() as f:
            idx = list(csv.DictReader(f))
        assert len(idx) == 1 and idx[0]["status"] == "done"
        assert idx[0]["n_test"] == "5"


def test_run_metrics_match_expected_from_fakes():
    _register_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_fake_config(), backend=_fake_backend(), runs_dir=tmp)[0]
        # INITIAL & STEERED_POS are always opinionated; STEERED_NEG always neutral.
        assert r.counts[INITIAL] == {"opinionated": 5}
        assert r.counts[STEERED_POS] == {"opinionated": 5}
        assert r.counts[STEERED_NEG] == {"neutral": 5}
        # opinion: init already opinionated -> no *new* wins (all same_good)
        assert r.quality["opinion"]["good"] == 0
        assert r.quality["opinion"]["same_good"] == 5
        # neutral: init opinionated -> steered_neg neutral => all "good"
        assert r.quality["neutral"]["good"] == 5


# ------------------------------------------------------------------ metrics units

def _triple(ex_id, init, pos, neg):
    return [
        Result(ex_id, INITIAL, "r", init),
        Result(ex_id, STEERED_POS, "r", pos),
        Result(ex_id, STEERED_NEG, "r", neg),
    ]


def test_steering_quality_logic():
    results = []
    results += _triple("a", "neutral", "opinionated", "neutral")     # opinion good, neutral same_good
    results += _triple("b", "opinionated", "opinionated", "neutral")  # opinion same_good, neutral good
    results += _triple("c", "opinionated", "neutral", "opinionated")  # opinion bad, neutral same_bad
    q = metrics.steering_quality(results, pos_label="opinionated", neg_label="neutral")
    assert q["opinion"] == {"good": 1, "bad": 1, "same_good": 1, "same_bad": 0}
    assert q["neutral"] == {"good": 1, "bad": 0, "same_good": 1, "same_bad": 1}


def test_condition_verdict_counts_and_tidy_rows():
    results = _triple("a", "neutral", "opinionated", "neutral")
    counts = metrics.condition_verdict_counts(results)
    assert counts[INITIAL] == {"neutral": 1}
    rows = metrics.tidy_rows(results, run_id="R", model="M", dataset="D",
                             opin_coeff=10, neut_coeff=12)
    coeff_by_cond = {row["condition"]: row["coeff"] for row in rows}
    assert coeff_by_cond == {INITIAL: 0, STEERED_POS: 10, STEERED_NEG: -12}


# ------------------------------------------------------------------ cli

def test_cli_loads_config_file():
    src_cfg = (
        "from src.bias_steer.config import ExperimentConfig, DatasetSpec, JudgeSpec, Coeffs\n"
        "config = ExperimentConfig(label='c', models=['qwen-7b'],\n"
        "    dataset=DatasetSpec(name='bbq', path='x'), judge=JudgeSpec(name='neutrality'),\n"
        "    coeffs=Coeffs(1, 2))\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cfg.py"
        p.write_text(src_cfg)
        cfg = cli.load_config_file(p)
        assert cfg.label == "c" and cfg.models == ["qwen-7b"]


def test_cli_queue_requires_a_route_file():
    # --queue is implemented (Phase 4); without _coordinator/route.json it must
    # exit cleanly (2), not raise. (The repo has no route file by default.)
    assert cli.main(["run", "--queue"]) == 2


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
