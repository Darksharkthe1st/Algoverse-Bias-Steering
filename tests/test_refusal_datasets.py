"""Chunk B — refusal harmful/harmless split loader (torch-free).

    python3 tests/test_refusal_datasets.py

Structural tests (filename parsing, path resolution, error messages) run anywhere.
Tests that read a real split file self-skip if the splits have not been fetched
(scripts/fetch_refusal_artifacts.py).
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

_SPLITS_DIR = datasets._REFUSAL_SPLITS_DIR


def _have(name):
    return (_SPLITS_DIR / name).exists()


# ---------------------------------------------------------------- structural

def test_registered():
    assert "refusal" in DATASETS


def test_parse_label_split():
    assert datasets._parse_label_split("harmful_train") == ("harmful", "train")
    assert datasets._parse_label_split("harmless_test") == ("harmless", "test")


def test_parse_rejects_bad_names():
    for bad in ("evil_train", "harmful_dev", "harmful", "harmless_"):
        try:
            datasets._parse_label_split(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_path_resolution_is_worktree_relative():
    # bare filename -> under the fetched splits dir
    assert datasets._refusal_split_path("harmful_train.json") == _SPLITS_DIR / "harmful_train.json"
    # a path with a separator -> worktree root, NOT get_repo_root (worktree-safe)
    root = _SPLITS_DIR.parents[3]  # splits -> dataset -> refusal_direction -> third_party -> root
    got = datasets._refusal_split_path("third_party/refusal_direction/dataset/splits/harmful_val.json")
    assert got == root / "third_party/refusal_direction/dataset/splits/harmful_val.json"


def test_empty_path_errors():
    try:
        datasets.load_refusal(DatasetSpec(name="refusal", path=""))
    except ValueError as e:
        assert "split file" in str(e)
        return
    raise AssertionError("expected ValueError for empty path")


def test_missing_file_error_names_fetch():
    try:
        datasets.load_refusal(DatasetSpec(name="refusal", path="harmful_train.json.NOPE"))
    except ValueError:
        return  # bad stem is rejected before the file check — also acceptable
    except FileNotFoundError as e:
        assert "fetch_refusal_artifacts.py" in str(e)
        return
    raise AssertionError("expected an error for a missing split")


# ---------------------------------------------------------------- data-gated

def test_loads_harmful_train():
    if not _have("harmful_train.json"):
        print("      (skipped: splits not fetched)")
        return
    ex = datasets.load_refusal(DatasetSpec(name="refusal", path="harmful_train.json"))
    assert len(ex) == 260, f"harmful_train n={len(ex)} != 260 (pinned commit 9d852fa)"
    e0 = ex[0]
    assert e0.metadata == {"label": "harmful", "split": "train", "category": None}
    assert e0.id == "refusal-harmful-train-0"
    assert isinstance(e0.prompt, str) and e0.prompt
    # deterministic order: reloading yields identical ids+prompts
    ex2 = datasets.load_refusal(DatasetSpec(name="refusal", path="harmful_train.json"))
    assert [e.id for e in ex] == [e.id for e in ex2]
    assert [e.prompt for e in ex] == [e.prompt for e in ex2]


def test_all_six_splits_labelled():
    names = [f"{lbl}_{sp}.json" for lbl in ("harmful", "harmless") for sp in ("train", "val", "test")]
    if not all(_have(n) for n in names):
        print("      (skipped: not all splits fetched)")
        return
    for n in names:
        lbl, sp = datasets._parse_label_split(n[:-5])
        ex = datasets.load_refusal(DatasetSpec(name="refusal", path=n))
        assert ex, f"{n} loaded empty"
        assert all(e.metadata["label"] == lbl and e.metadata["split"] == sp for e in ex)
        print(f"      {n:22s} n={len(ex)}")


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
