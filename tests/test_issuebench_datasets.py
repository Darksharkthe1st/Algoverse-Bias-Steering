"""IssueBench prompt-split loader (arXiv:2502.08395).

    python3 tests/test_issuebench_datasets.py

Structural tests (registration, split validation, path resolution, error
messages) run anywhere. Tests that read the real parquet self-skip if the split
has not been fetched (scripts/fetch_issuebench.py) or pandas is unavailable.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402,F401  (populates registries)
from src.bias_steer import datasets  # noqa: E402
from src.bias_steer.config import DatasetSpec  # noqa: E402
from src.bias_steer.registry import DATASETS  # noqa: E402

_DEBUG = datasets._ISSUEBENCH_DIR / "prompts" / "prompts_debug-00000-of-00001.parquet"


def _have_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- structural

def test_registered():
    assert "issuebench" in DATASETS


def test_rejects_unknown_split():
    for bad in ("train", "tiny", "DEBUG", ""):
        try:
            datasets.load_issuebench(DatasetSpec(name="issuebench", path=bad))
        except ValueError as e:
            assert "split" in str(e)
            continue
        except FileNotFoundError:
            # "" defaults to "debug", which is a valid split -> file-missing is fine
            if bad == "":
                continue
            raise AssertionError(f"expected ValueError for split {bad!r}")
        raise AssertionError(f"expected an error for split {bad!r}")


def test_missing_file_error_names_fetch():
    # A valid split that isn't fetched should point at the fetch script.
    import shutil
    if _DEBUG.exists():
        print("      (skipped: debug split is fetched, cannot test the missing path)")
        return
    try:
        datasets.load_issuebench(DatasetSpec(name="issuebench", path="sample"))
    except FileNotFoundError as e:
        assert "fetch_issuebench.py" in str(e)
        return
    raise AssertionError("expected FileNotFoundError naming the fetch script")


def test_dir_is_worktree_relative():
    # third_party/issuebench under the worktree root (parents[2] of datasets.py),
    # NOT get_repo_root — matches the refusal splits dir rationale.
    assert datasets._ISSUEBENCH_DIR.name == "issuebench"
    assert datasets._ISSUEBENCH_DIR.parent.name == "third_party"


# ---------------------------------------------------------------- data-gated

def test_loads_debug_split():
    if not _DEBUG.exists():
        print("      (skipped: debug split not fetched)")
        return
    if not _have_pandas():
        print("      (skipped: pandas unavailable)")
        return
    ex = datasets.load_issuebench(DatasetSpec(name="issuebench", path="debug"))
    assert ex, "debug split loaded empty"
    e0 = ex[0]
    assert e0.id == "issuebench-debug-0"
    assert isinstance(e0.prompt, str) and e0.prompt
    # schema keys the loader promises downstream code
    for k in ("category", "topic_id", "topic_text", "topic_polarity", "template_id", "split"):
        assert k in e0.metadata, f"missing metadata key {k!r}"
    # category mirrors topic_polarity so sample(per_group=("category", n)) works
    assert e0.metadata["category"] == e0.metadata["topic_polarity"]
    assert e0.metadata["split"] == "debug"
    print(f"      debug n={len(ex)}  polarities={sorted({e.metadata['category'] for e in ex})}")


def test_max_rows_caps_load():
    if not (_DEBUG.exists() and _have_pandas()):
        print("      (skipped: debug split not fetched / no pandas)")
        return
    spec = DatasetSpec(name="issuebench", path="debug")
    spec.max_rows = 5
    ex = datasets.load_issuebench(spec)
    assert len(ex) == 5, f"max_rows=5 -> {len(ex)}"


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
