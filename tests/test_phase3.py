"""Phase 3 verification: standalone analysis + plug-and-play (new dataset + method).

- analysis/compare.py reads run outputs (index.csv + results.csv) with no engine
  import and no pandas/torch.
- the new `stereoset` dataset and `last_token` method plug in via one function +
  one registry line each, with zero edits to experiment/metrics/config/tracking —
  demonstrated by running the pipeline end-to-end through the NEW dataset.

    python3 tests/test_phase3.py
"""

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
from src.bias_steer.schema import Example, INITIAL, STEERED_POS, STEERED_NEG  # noqa: E402
from analysis import compare  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

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


class _FakeMethod:
    name = "p3method"

    def capture(self, cache, n_layers):
        return ("resid", n_layers)

    def build(self, resids_by_label, contrast):
        return "VECTOR"

    def apply(self, model, vector, coeff):
        return [("sign", 1 if coeff >= 0 else -1)]


def _fake_backend():
    def load(spec):
        return types.SimpleNamespace(
            model=types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2)),
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
        save_vector=lambda p, v: Path(p).write_text("v"),
        save_residuals=lambda p, r: Path(p).write_text("r"),
    )


def _p3_config():
    # the NEW dataset flows through the UNCHANGED engine; only sampling keeps it small
    return ExperimentConfig(
        label="stereoset p3",
        models=["p3model"],
        dataset=DatasetSpec(name="stereoset", path=_STEREOSET_PATH, train_split=0.5),
        judge=JudgeSpec(name="p3judge", labels=["neutral", "opinionated"]),
        coeffs=Coeffs(opinion=10, neutral=12),
        sample=SampleSpec(limit=8, seed=0),
        method="p3method",
    )


# ---------------------------------------------------------- analysis over a real run

def test_analysis_reads_a_real_run():
    _register_p3_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        r = experiment.run(_p3_config(), backend=_fake_backend(), runs_dir=tmp)[0]

        # cross-run table from index.csv
        table = compare.compare(tmp)
        assert len(table) == 1
        assert table[0]["dataset"] == "stereoset"
        assert compare.format_table(table).startswith("run_id")

        # tidy per-response rows, aggregated after the fact
        rows = compare.load_run_results(tmp, r.run_id)
        assert len(rows) == 4 * 3  # 4 test examples x 3 conditions (limit 8, split .5)
        # fake generators: STEERED_NEG is always neutral, INITIAL always opinionated
        assert compare.rate(rows, STEERED_NEG, "neutral") == 1.0
        assert compare.rate(rows, INITIAL, "opinionated") == 1.0
        assert compare.verdict_counts(rows, condition=STEERED_POS) == {"opinionated": 4}

        # group_by column (per-category) works straight off the tidy rows
        by_cat = compare.verdict_counts(rows, condition=INITIAL, group_by="category")
        assert isinstance(by_cat, dict) and by_cat


# ---------------------------------------------------------- analysis units + purity

def test_verdict_counts_and_rate_on_fabricated_csv():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run1"
        run_dir.mkdir()
        (run_dir / "results.csv").write_text(
            "run_id,model,dataset,condition,coeff,example_id,verdict,category\n"
            "run1,m,d,initial,0,e0,neutral,A\n"
            "run1,m,d,initial,0,e1,opinionated,B\n"
            "run1,m,d,steered_neg,-5,e0,neutral,A\n"
        )
        rows = compare.load_results(run_dir)
        assert compare.verdict_counts(rows, condition=INITIAL) == {"neutral": 1, "opinionated": 1}
        assert compare.rate(rows, INITIAL, "neutral") == 0.5
        assert compare.verdict_counts(rows, condition=INITIAL, group_by="category") == {
            "A": {"neutral": 1}, "B": {"opinionated": 1},
        }


def test_analysis_is_engine_free():
    # §7.1: analysis must not import the engine, torch, or pandas.
    text = Path(_REPO_ROOT, "analysis", "compare.py").read_text()
    for forbidden in ("bias_steer", "import torch", "import pandas"):
        assert forbidden not in text, f"analysis/compare.py must not reference {forbidden!r}"


def test_empty_index_is_safe():
    with tempfile.TemporaryDirectory() as tmp:
        assert compare.load_index(tmp) == []
        assert compare.format_table(compare.compare(tmp)) == "(no rows)"


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
          + ("" if _HAS_TORCH else "  (torch-gated test ran in skip mode)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
