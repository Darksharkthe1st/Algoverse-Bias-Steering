"""Phase 1 verification: datasets + sampling, steering wiring, judge parsing,
model catalog/prompting.

Runs with plain Python (no ML stack required):

    python3 tests/test_phase1.py

Tests that genuinely need torch (capture/build tensor math) are SKIPPED here and
run on a GPU box where torch is installed. Everything else runs anywhere.
"""

import os
import sys
import types
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402
from src.bias_steer import registry, steering, models  # noqa: E402
from src.bias_steer.datasets import sample, load_bbq  # noqa: E402
from src.bias_steer.judges import parse_verdict, UNMATCHED  # noqa: E402
from src.bias_steer.config import (  # noqa: E402
    DatasetSpec, SampleSpec, ExperimentConfig, JudgeSpec, Coeffs,
)
from src.bias_steer.schema import Example  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------------------------------------------------------- registries

def test_package_imports_without_ml_stack():
    # The mere fact this file imported bias_steer proves the package (and its
    # Phase 1 submodules) load without torch/openai installed.
    for name in ("datasets", "models", "steering", "judges"):
        assert hasattr(bs, name), f"submodule {name} not exposed"


def test_registries_populated_on_import():
    assert {"bbq", "crows", "plain", "hidden_bias"}.issubset(registry.DATASETS)
    assert "mean_diff" in registry.METHODS
    assert "neutrality" in registry.JUDGES
    # model catalog: chat-template flags match the notebook
    assert registry.MODELS["qwen-7b"].chat_template is True
    assert registry.MODELS["gemma-2b"].chat_template is False
    assert registry.MODELS["qwen-7b"].quirks == ["qwen"]


def test_registry_validate_accepts_catalog_config():
    cfg = ExperimentConfig(
        label="t", models=["qwen-7b"],
        dataset=DatasetSpec(name="bbq", path="x"),
        judge=JudgeSpec(name="neutrality"), coeffs=Coeffs(13, 15),
        method="mean_diff",
    )
    registry.validate(cfg)  # no raise


# ---------------------------------------------------------------- datasets

def test_bbq_loader_metadata_and_prompt_parity():
    path = "datasets/BBQ_Prompt_Sets/Age.jsonl"
    exs = load_bbq(DatasetSpec(name="bbq", path=path))
    assert len(exs) > 0
    e = exs[0]
    assert e.metadata["category"] == "Age"
    assert len(e.metadata["answers"]) == 3
    assert "Pick one of three options" in e.prompt

    # FROZEN-LEGACY EQUIVALENCE ANCHOR: the inline loader's prompt must stay
    # byte-identical to legacy `src.data` (proves the inlining introduced no science
    # change). This is the *only* sanctioned reason the test suite imports src.data.
    from src.data import load_bbq_dataset
    from src.bias_steer.datasets import _resolve
    legacy = load_bbq_dataset(str(_resolve(path)))
    assert len(legacy) == len(exs)
    assert exs[0].prompt == legacy[0], "BBQ prompt drifted from legacy format"


def test_resolve_valid_path_and_fails_loud_on_missing():
    from src.bias_steer.datasets import _resolve
    from src.utils import get_repo_root

    # a path that ships with the repo resolves under the root
    good = _resolve("datasets/BBQ_Prompt_Sets/Age.jsonl")
    assert good.exists() and good == get_repo_root() / "datasets/BBQ_Prompt_Sets/Age.jsonl"

    # the natural mistake — a bare parent-relative path doubling the repo name — must
    # fail loud at resolve time, naming both the doubled path and the repo root, not
    # deep inside a loader's open()
    bad = "Algoverse-Bias-Steering/datasets/BBQ_Prompt_Sets/Age.jsonl"
    raised = False
    try:
        _resolve(bad)
    except FileNotFoundError as e:
        raised = True
        msg = str(e)
        assert "Algoverse-Bias-Steering" in msg          # names the doubled resolved path
        assert str(get_repo_root()) in msg               # names the repo root
    assert raised, "_resolve accepted a non-existent doubled path"


def test_plain_loader():
    # any one-per-line file works; reuse a small dataset that ships with the repo
    exs = bs.datasets.load_plain(
        DatasetSpec(name="plain", path="datasets/Homemade_Prompt_Sets/Objects/countries.txt")
    )
    assert len(exs) > 0 and all(isinstance(e, Example) for e in exs)
    assert exs[0].id == "plain-0"


# FROZEN-LEGACY EQUIVALENCE ANCHORS for the inlined loaders (#5): each proves the
# package's now-inline body produces prompts byte-identical to legacy `src.data`,
# before the `from src.data import ...` calls were deleted. Uses synthetic temp files
# (absolute paths, so _resolve passes them through) — no dependence on repo data.

def test_plain_inline_matches_legacy():
    import tempfile
    from src.data import load_plain_dataset
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plain.txt"
        p.write_text("alpha\n  beta  \n\ngamma\n")   # includes padding + a blank line
        legacy = load_plain_dataset(str(p))
        exs = bs.datasets.load_plain(DatasetSpec(name="plain", path=str(p)))
        assert [e.prompt for e in exs] == legacy


def test_crows_inline_matches_legacy():
    import tempfile
    from src.data import load_crows_pairs
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "crows.csv"
        p.write_text('He is a doctor,She is a nurse\n"quoted, cell",\nx,y\n')
        legacy = load_crows_pairs(str(p))                       # flattened cells
        expected = [s for s in legacy if isinstance(s, str) and s.strip()]
        exs = bs.datasets.load_crows(DatasetSpec(name="crows", path=str(p)))
        assert [e.prompt for e in exs] == expected


def test_hidden_bias_inline_matches_legacy():
    import tempfile
    from src.data import load_hidden_bias_dataset
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "hidden.csv"
        p.write_text('ctx one,opt A,opt B\n"ctx, two",opt C,opt D\n')
        legacy = load_hidden_bias_dataset(str(p))               # list[str] prompts
        exs = bs.datasets.load_hidden_bias(DatasetSpec(name="hidden_bias", path=str(p)))
        assert [e.prompt for e in exs] == legacy


def _synthetic(counts: dict) -> list[Example]:
    """Build Examples with a `category` metadata field: {cat: n}."""
    out = []
    for cat, n in counts.items():
        for i in range(n):
            out.append(Example(id=f"{cat}-{i}", prompt="p", metadata={"category": cat}))
    return out


def test_sample_filter():
    exs = _synthetic({"A": 50, "B": 30, "C": 20})
    got = sample(exs, SampleSpec(filter={"category": ["A", "B"]}))
    assert len(got) == 80
    assert {e.metadata["category"] for e in got} == {"A", "B"}


def test_sample_per_group_is_balanced_and_capped_by_availability():
    exs = _synthetic({"A": 50, "B": 30, "C": 5})
    got = sample(exs, SampleSpec(per_group=("category", 10), seed=0))
    by_cat = {}
    for e in got:
        by_cat.setdefault(e.metadata["category"], 0)
        by_cat[e.metadata["category"]] += 1
    assert by_cat == {"A": 10, "B": 10, "C": 5}  # C only had 5


def test_sample_limit_and_determinism():
    exs = _synthetic({"A": 50, "B": 50})
    a = sample(exs, SampleSpec(limit=20, seed=42))
    b = sample(exs, SampleSpec(limit=20, seed=42))
    assert len(a) == 20
    assert [e.id for e in a] == [e.id for e in b], "same seed must give same subset"


def test_sample_returns_de_blocked_order():
    # per_group used to return a list blocked by category (all A, then all B, ...);
    # sample() now shuffles so a positional train/test slice is balanced without the
    # caller having to shuffle first (#6). A blocked list would have exactly one
    # category boundary; an interleaved one has many.
    exs = _synthetic({"A": 30, "B": 30, "C": 30})
    got = sample(exs, SampleSpec(per_group=("category", 30), seed=0))
    assert len(got) == 90
    cats = [e.metadata["category"] for e in got]
    boundaries = sum(1 for i in range(1, len(cats)) if cats[i] != cats[i - 1])
    assert boundaries > 3, f"expected interleaved categories, got blocked (boundaries={boundaries})"
    # first half vs second half are both roughly balanced across categories
    from collections import Counter
    first_half = Counter(cats[:45])
    assert all(first_half[c] > 3 for c in ("A", "B", "C")), first_half


# ---------------------------------------------------------------- judge parsing

def test_parse_verdict_variants():
    labels = ["neutral", "opinionated"]
    assert parse_verdict("Long reasoning...\nANSWER: neutral", labels) == "neutral"
    assert parse_verdict("ANSWER: opinionated because it takes a stance", labels) == "opinionated"
    assert parse_verdict("answer: NEUTRAL", labels) == "neutral"          # case-insensitive
    assert parse_verdict("I think this is neutral overall", labels) == "neutral"  # no ANSWER: -> fallback
    assert parse_verdict("ANSWER: banana", labels) is None               # unknown label
    # multiple ANSWER: lines -> the last one wins
    assert parse_verdict("ANSWER: neutral\n...revised...\nANSWER: opinionated", labels) == "opinionated"


def test_unmatched_bucket_name():
    assert UNMATCHED == "nonsense"


# ---------------------------------------------------------------- judge retry (#7)
# Exercises the retry *mechanism* (retry/backoff/log/count + fail-fast) with injected
# fake exception classes and a stub client — no openai, no real sleeps. The *policy*
# (which openai classes are transient) is a trivial lazy import, checked separately.

def _stub_client(behaviour):
    """Async chat client whose `.chat.completions.create` runs `behaviour(n_call)`."""
    class _Stub:
        def __init__(self):
            self.calls = 0
            self.chat = types.SimpleNamespace(completions=self)

        async def create(self, model, messages, seed=None, temperature=None):
            self.calls += 1
            return behaviour(self.calls)
    return _Stub()


def _reply(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _run_retry(client, transient, stats):
    import asyncio
    from src.bias_steer.judges import base as judge
    orig = judge._backoff_seconds
    judge._backoff_seconds = lambda attempt: 0          # no real sleeps in tests
    try:
        return asyncio.run(
            judge._call_with_retry(client, "m", [], seed=0, temperature=0.0,
                                   transient=transient, stats=stats)
        )
    finally:
        judge._backoff_seconds = orig


def test_judge_retries_transient_then_succeeds():
    class _Transient(Exception):
        pass

    def behaviour(n):
        if n <= 2:
            raise _Transient("rate limited")
        return _reply("  ANSWER: neutral  ")

    client = _stub_client(behaviour)
    stats = {"retries": 0, "items_retried": 0}
    reply = _run_retry(client, (_Transient,), stats)
    assert reply == "ANSWER: neutral"                   # succeeded + stripped
    assert client.calls == 3                            # 2 failures + 1 success
    assert stats == {"retries": 2, "items_retried": 1}  # counted for the summary


def test_judge_fails_fast_on_non_transient():
    class _Transient(Exception):
        pass

    class _Permanent(Exception):
        pass

    def behaviour(n):
        raise _Permanent("400 bad request")

    client = _stub_client(behaviour)
    stats = {"retries": 0, "items_retried": 0}
    raised = False
    try:
        _run_retry(client, (_Transient,), stats)
    except _Permanent:
        raised = True
    assert raised, "non-transient error must propagate"
    assert client.calls == 1, "non-transient error must NOT be retried"
    assert stats == {"retries": 0, "items_retried": 0}


def test_judge_reraises_after_exhausting_retries():
    # not-C: a terminal transient failure still propagates loudly (no UNMATCHED swallow)
    from src.bias_steer.judges import base as judge

    class _Transient(Exception):
        pass

    def behaviour(n):
        raise _Transient("always down")

    client = _stub_client(behaviour)
    stats = {"retries": 0, "items_retried": 0}
    raised = False
    try:
        _run_retry(client, (_Transient,), stats)
    except _Transient:
        raised = True
    assert raised, "exhausted transient retries must re-raise"
    assert client.calls == judge._MAX_RETRIES
    assert stats["retries"] == judge._MAX_RETRIES and stats["items_retried"] == 1


# ---------------------------------------------------------------- steering (structural)

def test_resid_pre_hook_names():
    assert steering.resid_pre_hook_names(3) == [
        "blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre", "blocks.2.hook_resid_pre",
    ]


def test_mean_diff_method_defaults():
    m = registry.METHODS["mean_diff"]
    assert m.capture is steering.capture_mean
    assert m.build is steering.build_mean_difference
    assert m.apply is steering.apply_resid_pre_add


class _StubCfg:
    n_layers = 3
    d_model = 4


class _StubModel:
    cfg = _StubCfg()


class _StubVector:
    """A torch-free stand-in: apply only reads .ndim/.shape (for the shape guard)
    and indexes it per layer when building hooks; the tensor math runs only when a
    hook fires during generation, which this structural test does not do."""

    ndim = 2
    shape = (3, 4)  # (n_layers, d_model)

    def __getitem__(self, layer):
        return f"v{layer}"


def test_apply_builds_hooks_structure_without_torch():
    # building hooks needs only n_layers + an indexable (n_layers, d_model) vector;
    # torch is used only when the hook fires during generation.
    hooks = steering.apply_resid_pre_add(_StubModel(), _StubVector(), coeff=6.0)
    assert [name for name, _ in hooks] == steering.resid_pre_hook_names(3)
    assert len(hooks) == 3


def test_build_chat_messages():
    msgs = models.build_chat_messages("SYS", "USER")
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


# ---------------------------------------------------------------- torch-gated

def test_capture_and_build_math():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch

    d_model, seq = 4, 3
    # capture_mean: mean over the seq dim of one layer's resid_pre
    cache = {"blocks.0.hook_resid_pre": torch.arange(seq * d_model, dtype=torch.float32).reshape(1, seq, d_model)}
    cap = steering.capture_mean(cache, n_layers=1)
    assert cap.shape == (1, d_model)
    assert torch.allclose(cap[0], cache["blocks.0.hook_resid_pre"][0].mean(dim=0))
    # #9: captured residuals must land on CPU (kept off the model's device so they
    # don't accumulate in VRAM across the train phase). Meaningful on a GPU box; on a
    # CPU-only machine it's trivially true but still pins the contract.
    assert cap.device.type == "cpu"
    assert steering.capture_last(cache, n_layers=1).device.type == "cpu"

    # build_mean_difference: mean(pos) - mean(neg)
    n_layers = 2
    pos = [torch.ones(n_layers, d_model), torch.ones(n_layers, d_model) * 3]  # mean 2
    neg = [torch.zeros(n_layers, d_model)]                                    # mean 0
    vec = steering.build_mean_difference({"opinionated": pos, "neutral": neg}, ("opinionated", "neutral"))
    assert vec.shape == (n_layers, d_model)
    assert torch.allclose(vec, torch.ones(n_layers, d_model) * 2)


class _FakeCfg:
    def __init__(self, n_layers, d_model):
        self.n_layers = n_layers
        self.d_model = d_model


class _FakeModel:
    def __init__(self, n_layers, d_model):
        self.cfg = _FakeCfg(n_layers, d_model)


def test_apply_rejects_1d_vector_the_2025_bug():
    """A 1-D archived vector must raise, not silently broadcast a scalar.

    Regression test for the failure in docs/REVIVAL_AUDIT.md: `vector[layer]` on
    a 1-D tensor yields a scalar, and the hook then adds a uniform DC offset
    across the residual width instead of steering along a direction. Nothing
    raised, so the 2025 refusal runs produced plausible numbers that tested
    nothing. Without this test the bug returns the first time anyone loads an
    archived .pt.
    """
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch

    n_layers, d_model = 4, 8
    model = _FakeModel(n_layers, d_model)

    # The exact shape the archived .pt files load as.
    err = _expect(
        steering.SteeringShapeError,
        lambda: steering.apply_resid_pre_add(model, torch.ones(d_model), coeff=1.0),
    )
    assert "1-D" in str(err)

    # Transposed and wrong-width stacks are rejected too.
    _expect(
        steering.SteeringShapeError,
        lambda: steering.apply_resid_pre_add(model, torch.ones(d_model, n_layers), coeff=1.0),
    )
    _expect(
        steering.SteeringShapeError,
        lambda: steering.apply_resid_pre_add(model, torch.ones(n_layers, d_model + 1), coeff=1.0),
    )


def test_apply_accepts_correct_shape_and_steers_every_coordinate():
    """The good path still works, and adds a direction rather than a scalar."""
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch

    n_layers, d_model = 4, 8
    model = _FakeModel(n_layers, d_model)
    hooks = steering.apply_resid_pre_add(model, torch.ones(n_layers, d_model), coeff=2.0)

    assert len(hooks) == n_layers
    assert [h[0] for h in hooks] == steering.resid_pre_hook_names(n_layers)

    # Applying the first hook to a zero residual must move every coordinate by
    # coeff/n_layers — i.e. a (d_model,) direction was added, not a scalar.
    value = torch.zeros(1, 3, d_model)
    _, fn = hooks[0]
    out = fn(value, hook=None)
    assert out.shape == (1, 3, d_model)
    assert torch.allclose(out, torch.full((1, 3, d_model), 2.0 / n_layers))


def _expect(exc_type, fn):
    try:
        fn()
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__}")


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed"
          + ("" if _HAS_TORCH else "  (torch-gated math test ran in skip mode)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
