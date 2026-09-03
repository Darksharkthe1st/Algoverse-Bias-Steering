"""First-class contrast-vector extraction (experiment.build_contrast_vectors).

Wiring test with fakes — no torch, no OpenAI:

    python3 tests/test_contrast_vectors.py
"""

import os
import sys
import tempfile
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer import experiment, registry  # noqa: E402
from src.bias_steer.config import ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs  # noqa: E402
from src.bias_steer.models import ModelSpec  # noqa: E402
from src.bias_steer.schema import Example  # noqa: E402

# `sample()` re-shuffles (seeded), so exact per-split counts aren't hand-predictable.
# Instead: pools large enough that the floor outcome is unambiguous either way.
# hard-refusal total = 2, so with a floor of 5 V1 is ALWAYS under floor regardless of
# the split; everything else is large enough to clear it. 76 items -> 38 train / 38 test.
_COUNTS = {
    "soft-refusal": 20, "stance-factual": 20, "stance-evaluative": 12,
    "hard-refusal": 2, "non-engagement": 16, "incoherent": 6,
}
_ALL_LABELS = [lab for lab, n in _COUNTS.items() for _ in range(n)]  # 76 total
_N_TOTAL = len(_ALL_LABELS)
_N_TRAIN = int(_N_TOTAL * 0.5)
_N_TEST = _N_TOTAL - _N_TRAIN


class _FakeMethod:
    name = "cvfake"
    def capture(self, cache, n_layers): return ("resid", n_layers)
    def build(self, resids_by_label, contrast): return object()   # opaque; save_vector ignores it
    def apply(self, model, vector, coeff): return []
    def names(self, n_layers): return [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]


def _register_fakes():
    if "cvpool" not in registry.DATASETS:
        registry.register(
            registry.DATASETS, "cvpool",
            lambda spec: [Example(f"e{i}", lab) for i, lab in enumerate(_ALL_LABELS)],
        )
    if "cvfake" not in registry.MODELS:
        registry.register(registry.MODELS, "cvfake", ModelSpec("cvfake", "fake/model", True, "S"))
    if "cvfake" not in registry.METHODS:
        registry.register(registry.METHODS, "cvfake", _FakeMethod())
    if "cvjudge" not in registry.JUDGES:
        # identity judge: the fake generator echoes the prompt (a fine label) as the
        # response, so the verdict IS the label. No OpenAI.
        registry.register(registry.JUDGES, "cvjudge", lambda responses, examples, spec: list(responses))


def _fake_backend():
    def load(spec):
        model = types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2, d_model=4))
        return types.SimpleNamespace(model=model, tokenizer=None, spec=spec, device="cpu")

    def generate_with_cache(loaded, prompts, max_new_tokens, system_prompt, capture_names=None):
        return list(prompts), [None] * len(prompts)   # echo the prompt (= the fine label)

    def save_vector(path, vector, *, n_layers, d_model):
        from pathlib import Path
        Path(path).write_text("fake-vector")

    return experiment.Backend(load=load, generate_with_cache=generate_with_cache,
                              save_vector=save_vector)


def _cfg(strip=True):
    return ExperimentConfig(
        label="contrast fake",
        models=["cvfake"],
        dataset=DatasetSpec(name="cvpool", path="ignored", train_split=0.5, shuffle=False),
        judge=JudgeSpec(name="cvjudge", labels=["soft-refusal", "stance-factual"]),
        coeffs=Coeffs(0.0, 0.0),
        sample=SampleSpec(seed=0),
        method="cvfake",
        strip_reasoning=strip,
    )


def test_builds_only_contrasts_that_clear_the_floor():
    _register_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        results = experiment.build_contrast_vectors(_cfg(), backend=_fake_backend(),
                                                    runs_dir=tmp, n_floor=5)
    _run_dir, built = results[0]
    # V1 (soft<-hard): hard-refusal total is 2 (<5) -> skipped; V2, V3 clear the floor.
    assert built == ["V2", "V3"], built


def test_writes_vectors_and_test_split():
    _register_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        d, built = experiment.build_contrast_vectors(_cfg(), backend=_fake_backend(),
                                                     runs_dir=tmp, n_floor=5)[0]
        assert built == ["V2", "V3"]
        assert (d / "V2.safetensors").is_file() and (d / "V3.safetensors").is_file()
        assert not (d / "V1.safetensors").exists()   # under floor -> not saved

        # held-out split for Phase 4: header + the test items, disjoint from train.
        test_rows = list(open(d / "test_split.csv", encoding="utf-8"))
        assert len(test_rows) == 1 + _N_TEST


def test_build_under_floor_builds_all_three():
    _register_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        _d, built = experiment.build_contrast_vectors(_cfg(), backend=_fake_backend(),
                                                      runs_dir=tmp, n_floor=5, require_floor=False)[0]
    assert built == ["V1", "V2", "V3"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
