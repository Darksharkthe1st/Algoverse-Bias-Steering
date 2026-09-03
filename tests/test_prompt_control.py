"""run() with a system-prompt control arm (needed-experiments §14).

    python3 tests/test_prompt_control.py

Three modes share the one run path: `steer` (vector only, the historical default),
`prompt` (system-prompt baseline, NO vector), and `both` (all five arms, so
steer-vs-prompt is a per-item comparison). Torch-free: a fake Backend whose
`generate` keys its verdict off the *system prompt* it receives, so the test proves
each arm is fed the right prompt end-to-end (incl. the frozen DEFAULT_POS/NEG_SYS).
"""

import csv
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
from src.bias_steer.config import (  # noqa: E402
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs, ModelSpec,
    DEFAULT_SYS, DEFAULT_POS_SYS, DEFAULT_NEG_SYS,
)
from src.bias_steer.schema import (  # noqa: E402
    Example, INITIAL, STEERED_POS, STEERED_NEG, PROMPT_POS, PROMPT_NEG,
)


class _StubVector:
    shape = (2, 4)
    def to(self, device):
        return self


class _FakeMethod:
    name = "promptfake"
    def capture(self, cache, n_layers): return ("resid", n_layers)
    def build(self, resids_by_label, contrast): return _StubVector()
    def apply(self, model, vector, coeff): return [("sign", 1 if coeff >= 0 else -1)]
    def names(self, n_layers): return [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]


def _register_fakes():
    if "promptfake" not in registry.DATASETS:
        registry.register(
            registry.DATASETS, "promptfake",
            lambda spec: [Example(f"e{i}", f"prompt {i}", {"category": "X"}) for i in range(6)],
        )
    if "promptfake" not in registry.MODELS:
        registry.register(registry.MODELS, "promptfake", ModelSpec("promptfake", "fake/model", True, "S"))
    if "promptfake" not in registry.METHODS:
        registry.register(registry.METHODS, "promptfake", _FakeMethod())
    if "promptfake" not in registry.JUDGES:
        registry.register(registry.JUDGES, "promptfake", lambda responses, examples, spec: list(responses))


def _fake_backend(calls):
    def load(spec):
        model = types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2, d_model=4))
        return types.SimpleNamespace(model=model, tokenizer=None, spec=spec, device="cpu")

    def generate(loaded, prompts, mnt, sysp):
        # Verdict is decided by WHICH system prompt this arm was handed — this is
        # what proves INITIAL/PROMPT_POS/PROMPT_NEG each got the right instruction.
        calls["sys_prompts"].add(sysp)
        if sysp == DEFAULT_POS_SYS:
            verd = "opinionated"
        elif sysp == DEFAULT_NEG_SYS:
            verd = "neutral"
        else:  # default/control prompt
            verd = "opinionated"
        return [verd] * len(prompts)

    def generate_with_hooks(loaded, prompts, hooks, mnt, sysp):
        sign = hooks[0][1] if hooks else 1
        return [("opinionated" if sign > 0 else "neutral")] * len(prompts)

    def generate_with_cache(loaded, prompts, mnt, sysp, capture_names=None):
        return (["opinionated" if i % 2 == 0 else "neutral" for i in range(len(prompts))],
                [None] * len(prompts))

    def save_vector(path, vector, *, n_layers, d_model):
        Path(path).write_text("fake-vector")

    def save_residuals(path, resids, *, n_layers, d_model):
        Path(path).write_text("fake-residuals")

    def load_vector(path):
        calls["load_vector"] += 1
        return _StubVector()

    return experiment.Backend(
        load=load, generate=generate, generate_with_hooks=generate_with_hooks,
        generate_with_cache=generate_with_cache, save_vector=save_vector,
        save_residuals=save_residuals, load_vector=load_vector,
    )


def _cfg(intervention):
    return ExperimentConfig(
        label=f"prompt fake {intervention}",
        models=["promptfake"],
        dataset=DatasetSpec(name="promptfake", path="ignored"),  # 6 examples
        judge=JudgeSpec(name="promptfake", labels=["neutral", "opinionated"]),
        coeffs=Coeffs(opinion=8, neutral=8),
        sample=SampleSpec(seed=0),
        method="promptfake",
        intervention=intervention,
    )


def _rows(run_dir):
    with (run_dir / "results.csv").open() as f:
        return list(csv.DictReader(f))


def test_prompt_mode_runs_baseline_arms_without_a_vector():
    _register_fakes()
    calls = {"load_vector": 0, "sys_prompts": set()}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_cfg("prompt"), backend=_fake_backend(calls), runs_dir=tmp)[0]

        # No vector anywhere: not loaded, not extracted, not required as evidence.
        assert calls["load_vector"] == 0
        assert not (r.dir / "steering_vector.safetensors").exists()
        assert not (r.dir / "residuals.safetensors").exists()

        # Exactly the control + two prompt arms, on all 6 examples (train folded in).
        rows = _rows(r.dir)
        assert {row["condition"] for row in rows} == {INITIAL, PROMPT_POS, PROMPT_NEG}
        assert len(rows) == 18

        # Each arm was handed its own frozen system prompt (proves the wiring).
        assert {DEFAULT_SYS, DEFAULT_POS_SYS, DEFAULT_NEG_SYS} <= calls["sys_prompts"]
        assert r.counts[PROMPT_POS] == {"opinionated": 6}
        assert r.counts[PROMPT_NEG] == {"neutral": 6}

        summary = (r.dir / "summary.md").read_text(encoding="utf-8")
        assert "Prompt-baseline quality" in summary and "intervention: `prompt`" in summary


def test_both_mode_runs_all_five_arms_and_compares():
    _register_fakes()
    calls = {"load_vector": 0, "sys_prompts": set()}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_cfg("both"), vector_path="v.safetensors",
                           backend=_fake_backend(calls), runs_dir=tmp)[0]

        assert calls["load_vector"] == 1  # steer arms need the supplied vector
        rows = _rows(r.dir)
        assert {row["condition"] for row in rows} == {
            INITIAL, STEERED_POS, STEERED_NEG, PROMPT_POS, PROMPT_NEG}
        assert len(rows) == 30  # 6 examples * 5 arms

        # The per-item steer-vs-prompt section is present only in "both".
        summary = (r.dir / "summary.md").read_text(encoding="utf-8")
        assert "Steer vs prompt (per-item" in summary
        assert "Prompt-baseline quality" in summary and "Steering quality (vector)" in summary
        # The complementarity split (which questions each method won) is rendered, so a
        # near-zero Δ can be read as agreement vs cancellation, not just "no difference".
        assert "per-item: both" in summary and "discordant" in summary


def test_steer_mode_is_unchanged():
    _register_fakes()
    calls = {"load_vector": 0, "sys_prompts": set()}
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_cfg("steer"), vector_path="v.safetensors",
                           backend=_fake_backend(calls), runs_dir=tmp)[0]
        rows = _rows(r.dir)
        assert {row["condition"] for row in rows} == {INITIAL, STEERED_POS, STEERED_NEG}
        assert (r.dir / "steering_vector.safetensors").is_file()
        summary = (r.dir / "summary.md").read_text(encoding="utf-8")
        assert "Prompt-baseline quality" not in summary and "Steer vs prompt" not in summary


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
