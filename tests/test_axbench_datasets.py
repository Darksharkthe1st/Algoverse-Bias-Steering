"""AxBench Concept500 loader (arXiv:2501.17148).

    python3 tests/test_axbench_datasets.py

Structural tests (registration, path parsing, error messages) run anywhere.
Tests that read the real parquet self-skip if the variant has not been fetched
(scripts/fetch_axbench.py) or pandas is unavailable.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402,F401  (populates registries)
from src.bias_steer import datasets  # noqa: E402
from src.bias_steer.config import DatasetSpec, SampleSpec  # noqa: E402
from src.bias_steer.registry import DATASETS  # noqa: E402

_DEFAULT = datasets._AXBENCH_DIR / "2b" / "l20" / "train" / "data.parquet"


def _have_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- structural

def test_registered():
    assert "axbench" in DATASETS


def test_rejects_bad_path():
    for bad in ("2b/l20", "2b/l20/dev", "3b/l20/train", "2b/l99/train", "garbage"):
        try:
            datasets.load_axbench(DatasetSpec(name="axbench", path=bad))
        except ValueError as e:
            assert "variant" in str(e) or "split" in str(e)
            continue
        raise AssertionError(f"expected ValueError for path {bad!r}")


def test_missing_file_error_names_fetch():
    if _DEFAULT.exists():
        print("      (skipped: default variant fetched, cannot test missing path)")
        return
    try:
        datasets.load_axbench(DatasetSpec(name="axbench", path="9b/l31/train"))
    except FileNotFoundError as e:
        assert "fetch_axbench.py" in str(e)
        return
    raise AssertionError("expected FileNotFoundError naming the fetch script")


def test_dir_is_worktree_relative():
    assert datasets._AXBENCH_DIR.name == "axbench"
    assert datasets._AXBENCH_DIR.parent.name == "third_party"


# ---------------------------------------------------------------- data-gated

def test_loads_default_variant():
    if not _DEFAULT.exists():
        print("      (skipped: 2b/l20/train not fetched)")
        return
    if not _have_pandas():
        print("      (skipped: pandas unavailable)")
        return
    ex = datasets.load_axbench(DatasetSpec(name="axbench", path="2b/l20/train"))
    assert ex, "default variant loaded empty"
    e0 = ex[0]
    assert e0.id.startswith("axbench-2b-l20-train-")
    assert isinstance(e0.prompt, str) and e0.prompt
    for k in ("label", "category", "concept_id", "output_concept", "output", "variant", "split"):
        assert k in e0.metadata, f"missing metadata key {k!r}"
    # the DiffMean contrast lives in `label` and must contain both poles
    labels = {e.metadata["label"] for e in ex}
    assert {"positive", "negative"} <= labels, f"labels={labels} lack a pole"
    assert e0.metadata["variant"] == "2b/l20" and e0.metadata["split"] == "train"
    print(f"      2b/l20/train n={len(ex)}  labels={sorted(labels)}  "
          f"genres={sorted({e.metadata['category'] for e in ex})}")


def test_label_bucketing_contrast_available():
    # The whole point of the AxBench-native path: both poles present so
    # build_mean_difference(contrast=("positive","negative")) has data on each side.
    if not (_DEFAULT.exists() and _have_pandas()):
        print("      (skipped: not fetched / no pandas)")
        return
    ex = datasets.load_axbench(DatasetSpec(name="axbench", path="2b/l20/train"))
    pos = [e for e in ex if e.metadata["label"] == "positive"]
    neg = [e for e in ex if e.metadata["label"] == "negative"]
    assert pos and neg, f"pos={len(pos)} neg={len(neg)} — need both for a contrast"
    # negatives carry the EEEEE sentinel concept
    assert any(e.metadata["output_concept"] == "EEEEE" for e in neg)


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
