"""Phase 3 verification: standalone analysis + plug-and-play (new dataset + method).

- the new `stereoset` dataset and `last_token` method plug in via one function +
  one registry line each, with zero edits to experiment/metrics/config/tracking —
  demonstrated by running the pipeline end-to-end through the NEW dataset (this
  part needs no pandas).
- analysis/compare.py (pandas) reads run outputs without importing the engine.

Pandas isn't installed in every environment; analysis tests SKIP when it's absent
(they run wherever pandas exists). The plug-and-play + engine tests always run.

    python3 tests/test_phase3.py
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

import src.bias_steer as bs  # noqa: E402
from src.bias_steer import registry, steering, experiment  # noqa: E402
from src.bias_steer.datasets import load_stereoset  # noqa: E402
from src.bias_steer.config import (  # noqa: E402
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs, ModelSpec,
)
from src.bias_steer.schema import INITIAL, STEERED_POS, STEERED_NEG  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import pandas  # noqa: F401
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

_STEREOSET_PATH = "src/stereoset-data.json"


# ---------------------------------------------------------- new dataset (plug-in)

def test_stereoset_loader():
    exs = load_stereoset(DatasetSpec(name="stereoset", path=_STEREOSET_PATH))
    assert len(exs) > 1000
    cats = {e.metadata["category"] for e in exs}
    assert {"race", "profession", "gender", "religion"} <= cats  # same key BBQ uses
    e = exs[0]
    assert "Pick one" in e.prompt and len(e.metadata["gold_labels"]) == 3


def test_stereoset_registered():
    assert "stereoset" in registry.DATASETS


# ---------------------------------------------------------- new method (plug-in)

def test_last_token_method_registered_and_reuses_defaults():
    assert "last_token" in registry.METHODS and "mean_diff" in registry.METHODS
    m = registry.METHODS["last_token"]
    assert m.capture is steering.capture_last          # the one overridden piece
    assert m.build is steering.build_mean_difference   # reused
    assert m.apply is steering.apply_resid_pre_add      # reused


def test_capture_last_math():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch

    d_model, seq = 4, 3
    resid = torch.arange(seq * d_model, dtype=torch.float32).reshape(1, seq, d_model)
    cap = steering.capture_last({"blocks.0.hook_resid_pre": resid}, n_layers=1)
    assert cap.shape == (1, d_model)
    assert torch.allclose(cap[0], resid[0, -1, :])     # last token, not the mean


# ---------------------------------------------------------- fakes for a real run

def _register_p3_fakes():
    if "p3model" not in registry.MODELS:
        registry.register(registry.MODELS, "p3model", ModelSpec("p3model", "fake/m", True, "S"))
    if "p3method" not in registry.METHODS:
        registry.register(registry.METHODS, "p3method", _FakeMethod())
    if "p3judge" not in registry.JUDGES:
        registry.register(registry.JUDGES, "p3judge", lambda responses, examples, spec: list(responses))


class _FakeVector:
    """Opaque steering vector; needs only `.to()` (run() moves it to device, #9)."""
    def to(self, device):
        return self


class _FakeMethod:
    name = "p3method"

    def capture(self, cache, n_layers):
        return ("resid", n_layers)

    def build(self, resids_by_label, contrast):
        return _FakeVector()

    def apply(self, model, vector, coeff):
        return [("sign", 1 if coeff >= 0 else -1)]


def _fake_backend():
    def load(spec):
        return types.SimpleNamespace(
            model=types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2, d_model=4)),
            tokenizer=None, spec=spec, device="cpu",
        )

    def generate_with_cache(loaded, prompts, mnt, sysp):
        return (["opinionated" if i % 2 == 0 else "neutral" for i in range(len(prompts))],
                [None] * len(prompts))

    def generate(loaded, prompts, mnt, sysp):
        return ["opinionated"] * len(prompts)

    def generate_with_hooks(loaded, prompts, hooks, mnt, sysp):
        sign = hooks[0][1] if hooks else 1
        return [("opinionated" if sign > 0 else "neutral")] * len(prompts)

    return experiment.Backend(
        load=load, generate=generate, generate_with_cache=generate_with_cache,
        generate_with_hooks=generate_with_hooks,
        save_vector=lambda p, v, **kw: Path(p).write_text("v"),
        save_residuals=lambda p, r, **kw: Path(p).write_text("r"),
    )


def _p3_config():
    # the NEW dataset flows through the UNCHANGED engine; sampling keeps it small
    return ExperimentConfig(
        label="stereoset p3",
        models=["p3model"],
        dataset=DatasetSpec(name="stereoset", path=_STEREOSET_PATH, train_split=0.5),
        judge=JudgeSpec(name="p3judge", labels=["neutral", "opinionated"]),
        coeffs=Coeffs(opinion=10, neutral=12),
        sample=SampleSpec(limit=8, seed=0),
        method="p3method",
    )


def _run_stereoset(tmp):
    _register_p3_fakes()
    return experiment.run(_p3_config(), backend=_fake_backend(), runs_dir=tmp)[0]


# ---------------------------------------------------------- plug-and-play (no pandas)

def test_stereoset_plugs_into_run_end_to_end():
    """A brand-new dataset runs through the engine with zero engine edits."""
    with tempfile.TemporaryDirectory() as tmp:
        r = _run_stereoset(tmp)
        assert (r.dir / "results.csv").is_file()
        with (Path(tmp) / "index.csv").open() as f:
            idx = list(csv.DictReader(f))
        assert len(idx) == 1 and idx[0]["dataset"] == "stereoset"
        assert idx[0]["n_test"] == "4"  # limit 8, split 0.5


# ---------------------------------------------------------- analysis (pandas-gated)

def test_analysis_reads_a_real_run():
    if not _HAS_PANDAS:
        print("      (skipped: pandas not installed)")
        return
    from analysis import compare

    with tempfile.TemporaryDirectory() as tmp:
        r = _run_stereoset(tmp)

        table = compare.compare(tmp)
        assert len(table) == 1 and table.iloc[0]["dataset"] == "stereoset"

        res = compare.load_run_results(tmp, r.run_id)
        assert len(res) == 4 * 3  # 4 test examples x 3 conditions

        rates = compare.verdict_rates(res)
        neg_neutral = rates[(rates.condition == STEERED_NEG) & (rates.verdict == "neutral")]
        assert float(neg_neutral["rate"].iloc[0]) == 1.0
        # per-category split is one arg
        by_cat = compare.verdict_rates(res, by="category")
        assert "category" in by_cat.columns


def test_verdict_rates_on_fabricated_frame():
    if not _HAS_PANDAS:
        print("      (skipped: pandas not installed)")
        return
    import pandas as pd

    from analysis import compare

    df = pd.DataFrame([
        {"condition": INITIAL, "verdict": "neutral", "category": "A"},
        {"condition": INITIAL, "verdict": "opinionated", "category": "B"},
        {"condition": STEERED_POS, "verdict": "opinionated", "category": "A"},
    ])
    rates = compare.verdict_rates(df)
    init = rates[rates.condition == INITIAL].set_index("verdict")["rate"].to_dict()
    assert init == {"neutral": 0.5, "opinionated": 0.5}


def test_empty_index_is_safe():
    if not _HAS_PANDAS:
        print("      (skipped: pandas not installed)")
        return
    from analysis import compare

    with tempfile.TemporaryDirectory() as tmp:
        assert compare.load_index(tmp).empty


# ---------------------------------------------------------- analysis purity (always)

def test_analysis_is_engine_free():
    # §7.1: analysis must not *import* the engine or torch (pandas is allowed).
    # Inspect actual import statements via AST — mentions in docstrings are fine.
    import ast

    tree = ast.parse(Path(_REPO_ROOT, "analysis", "compare.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for mod in imported:
        assert "bias_steer" not in mod, f"analysis must not import the engine (found {mod!r})"
        assert mod != "torch" and not mod.startswith("torch."), f"analysis must not import torch (found {mod!r})"


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
    skipped = "" if (_HAS_TORCH and _HAS_PANDAS) else "  (some tests ran in skip mode)"
    print(f"\n{len(tests) - failed}/{len(tests)} passed{skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
