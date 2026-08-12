"""Refusal-direction repro (arXiv:2406.11717): artifact loading + catalog wiring.

    python3 tests/test_refusal.py

Structural tests (mapping, catalog membership) run anywhere. Tests that actually
`torch.load` a `direction.pt` are gated twice: they skip if torch is absent AND
if the artifacts have not been fetched (run scripts/fetch_refusal_artifacts.py).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402
from src.bias_steer import refusal, registry, artifacts  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Ground-truth (layer, pos) from each model's committed direction_metadata.json,
# verified against the mean_diffs storage-offset provenance in Chunk 0.
_EXPECTED_META = {
    "qwen-1_8b-chat": (15, -2),
    "gemma-2b-it": (10, -2),
    "yi-6b-chat": (20, -5),
    "meta-llama-3-8b-instruct": (12, -5),
    "llama-2-7b-chat-hf": (14, -1),
}
_EXPECTED_DMODEL = {
    "qwen-1_8b-chat": 2048, "gemma-2b-it": 2048, "yi-6b-chat": 4096,
    "meta-llama-3-8b-instruct": 4096, "llama-2-7b-chat-hf": 4096,
}


# ---------------------------------------------------------------- structural

def test_run_dir_mapping_targets_registered_models():
    # Every upstream run dir maps to a model that exists in the catalog, so a
    # loaded direction can always be paired with a loadable model.
    for run_dir, model_key in refusal.RUN_DIR_TO_MODEL.items():
        assert model_key in registry.MODELS, f"{run_dir} -> {model_key} not in MODELS"


def test_llama2_catalog_entry():
    spec = registry.MODELS["llama-2-7b"]
    assert spec.hf_id == "meta-llama/Llama-2-7b-chat-hf"
    assert spec.chat_template is True  # refusal direction needs the chat template


def test_resolve_accepts_both_key_forms():
    assert refusal._resolve("qwen-1.8b") == ("qwen-1_8b-chat", "qwen-1.8b")
    assert refusal._resolve("qwen-1_8b-chat") == ("qwen-1_8b-chat", "qwen-1.8b")


def test_resolve_rejects_unknown():
    try:
        refusal._resolve("gpt-4")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown model")


def test_missing_artifact_error_names_fetch_command():
    # Resolvable model, but point the loader at a run dir with no file: the error
    # must tell the user how to fetch. Use a model unlikely to be fetched here.
    if "gemma-7b" in refusal.available_run_dirs():
        print("      (skipped: unexpected artifact present)")
        return
    # gemma-7b isn't in the mapping; craft the check via a resolvable-but-absent one.
    # If nothing is fetched, any known key triggers FileNotFoundError.
    if refusal.available_run_dirs():
        print("      (skipped: artifacts fetched; missing-file path not exercised)")
        return
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    try:
        refusal.load_refusal_direction("qwen-1.8b")
    except FileNotFoundError as e:
        assert "fetch_refusal_artifacts.py" in str(e)
        return
    raise AssertionError("expected FileNotFoundError naming the fetch script")


# ---------------------------------------------------------------- torch + data gated

def test_generic_pt_loader_reads_direction_tensor():
    # artifacts.load_pt_tensor is the reusable, refusal-agnostic loader for
    # external .pt vectors; exercise it directly on a fetched file.
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    fetched = refusal.available_run_dirs()
    if not fetched:
        print("      (skipped: no artifacts fetched)")
        return
    import torch
    path = refusal.artifact_dir(fetched[0]) / "direction.pt"
    t = artifacts.load_pt_tensor(path)
    assert isinstance(t, torch.Tensor) and t.ndim == 1 and t.numel() > 0


def test_load_directions_shape_and_provenance():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    fetched = refusal.available_run_dirs()
    if not fetched:
        print("      (skipped: no artifacts fetched — run scripts/fetch_refusal_artifacts.py)")
        return
    for run_dir in fetched:
        rd = refusal.load_refusal_direction(run_dir)
        # shape: a single (d_model,) vector
        assert rd.direction.ndim == 1, f"{run_dir}: expected 1-D, got {rd.direction.shape}"
        assert rd.direction.dtype.is_floating_point
        # metadata matches the committed ground truth
        if run_dir in _EXPECTED_META:
            assert (rd.layer, rd.pos) == _EXPECTED_META[run_dir], \
                f"{run_dir}: (layer,pos)={ (rd.layer, rd.pos) } != {_EXPECTED_META[run_dir]}"
            assert rd.d_model == _EXPECTED_DMODEL[run_dir], \
                f"{run_dir}: d_model={rd.d_model} != {_EXPECTED_DMODEL[run_dir]}"
        # the raw direction is non-degenerate
        assert float(rd.direction.norm()) > 0
        # loaded model_key is a catalog model
        assert rd.model_key in registry.MODELS
        print(f"      {run_dir:28s} d_model={rd.d_model} layer={rd.layer} pos={rd.pos} "
              f"|r|={float(rd.direction.norm()):.3f}")


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
          + ("" if _HAS_TORCH else "  (torch-gated load test ran in skip mode)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
