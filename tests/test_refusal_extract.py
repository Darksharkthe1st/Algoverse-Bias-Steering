"""Chunk C/D — refusal extraction: capture_prompt_positions / build_refusal_grid /
run_extraction label-bucketing.

    python3 tests/test_refusal_extract.py

Structural tests (templates, deterministic sampling, method registration) run
anywhere. Numeric tests use stub caches + a fake ExtractionBackend — they need
torch but NO model download, and self-skip if torch is absent.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402,F401
from src.bias_steer import refusal_extract as rx  # noqa: E402
from src.bias_steer import steering  # noqa: E402
from src.bias_steer.registry import METHODS  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# The refusal splits are third-party artifacts fetched by
# `scripts/fetch_refusal_artifacts.py`, not committed (they are gitignored). A
# test needing them must self-skip when they are absent, the same way the
# numeric tests self-skip without torch. Hard-failing reports a missing download
# as a broken repo — which is how this suite came to carry a permanent red that
# everyone learned to read past.
_SPLITS_DIR = os.path.join(
    _REPO_ROOT, "third_party", "refusal_direction", "dataset", "splits"
)
_HAS_SPLITS = os.path.isdir(_SPLITS_DIR) and bool(os.listdir(_SPLITS_DIR))
_NO_SPLITS = "refusal splits not fetched: python scripts/fetch_refusal_artifacts.py"


# ---------------------------------------------------------------- structural

def test_method_registered():
    m = METHODS["refusal_extract"]
    assert m.capture is steering.capture_prompt_positions
    assert m.build is steering.build_refusal_grid


def test_templates_verbatim_and_no_system():
    # exact upstream literals (system=None variant)
    assert rx.format_refusal_prompt("qwen-1.8b", "X") == \
        "<|im_start|>user\nX<|im_end|>\n<|im_start|>assistant\n"
    assert rx.format_refusal_prompt("yi-6b", "X") == \
        "<|im_start|>user\nX<|im_end|>\n<|im_start|>assistant\n"
    assert rx.format_refusal_prompt("gemma-2b", "X") == \
        "<start_of_turn>user\nX<end_of_turn>\n<start_of_turn>model\n"
    assert rx.format_refusal_prompt("llama-2-7b", "X") == "[INST] X [/INST] "
    assert rx.format_refusal_prompt("llama3-8b", "X") == (
        "<|start_header_id|>user<|end_header_id|>\n\nX"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    # no system prompt leaks in
    for k in rx.REFUSAL_TEMPLATES:
        assert "system" not in rx.format_refusal_prompt(k, "X").lower()


def test_run_dir_alias_resolves():
    assert rx.format_refusal_prompt("qwen-1_8b-chat", "X") == rx.format_refusal_prompt("qwen-1.8b", "X")


def test_eoi_suffix():
    assert rx.eoi_suffix("qwen-1.8b") == "<|im_end|>\n<|im_start|>assistant\n"
    assert rx.eoi_suffix("llama-2-7b") == " [/INST] "


def test_sampling_deterministic_and_sized():
    if not _HAS_SPLITS:
        # Gate on actually running under pytest, not on pytest being importable:
        # `pytest.skip()` outside a pytest run raises Skipped, which the plain
        # `python3 tests/...` runner would report as a failure.
        if "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip(_NO_SPLITS)
        print(f"      (skipped: {_NO_SPLITS})")
        return
    s1 = rx.load_and_sample_repro(n_train=128, n_val=32)
    s2 = rx.load_and_sample_repro(n_train=128, n_val=32)
    assert s1 == s2  # deterministic (seed 42, fixed input order)
    assert len(s1["harmful_train"]) == 128 and len(s1["harmless_train"]) == 128
    assert len(s1["harmful_val"]) == 32 and len(s1["harmless_val"]) == 32
    # different seed -> different selection
    assert rx.load_and_sample_repro(seed=0)["harmful_train"] != s1["harmful_train"]


# ---------------------------------------------------------------- numeric (torch-gated)

def _stub_cache(n_layers, seq, d_model, fill):
    """One prompt's resid_pre cache: each layer L, position p -> fill(L, p)."""
    cache = {}
    for L in range(n_layers):
        t = torch.zeros(1, seq, d_model)
        for p in range(seq):
            t[0, p, :] = fill(L, p)
        cache[f"blocks.{L}.hook_resid_pre"] = t
    return cache


def test_capture_prompt_positions_shape_order_and_values():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    n_layers, seq, d_model, n_pos = 3, 7, 4, 5
    # value encodes (layer, absolute position) so we can verify slicing + axis order
    cache = _stub_cache(n_layers, seq, d_model, fill=lambda L, p: 100 * L + p)
    grid = steering.capture_prompt_positions(cache, n_layers, n_pos)
    assert grid.shape == (n_pos, n_layers, d_model), grid.shape
    # last n_pos positions are absolute seq indices [2..6]; axis0 index i -> pos seq-n_pos+i
    for i in range(n_pos):
        abs_pos = seq - n_pos + i
        for L in range(n_layers):
            assert torch.allclose(grid[i, L], torch.full((d_model,), float(100 * L + abs_pos)))


def test_build_refusal_grid_is_harmful_minus_harmless():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    n_pos, n_layers, d_model = 5, 3, 4
    harmful = [torch.full((n_pos, n_layers, d_model), 3.0),
               torch.full((n_pos, n_layers, d_model), 5.0)]  # mean 4
    harmless = [torch.full((n_pos, n_layers, d_model), 1.0),
                torch.full((n_pos, n_layers, d_model), 1.0)]  # mean 1
    grid = steering.build_refusal_grid({"harmful": harmful, "harmless": harmless})
    assert grid.shape == (n_pos, n_layers, d_model)
    assert torch.allclose(grid, torch.full((n_pos, n_layers, d_model), 3.0))  # 4 - 1


def test_run_extraction_buckets_by_label_with_fake_backend():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)"); return
    n_layers, seq, d_model, n_pos = 4, 6, 8, 5

    class _Cfg:  # noqa
        def __init__(s): s.n_layers = n_layers; s.device = "cpu"

    class _Model:  # noqa
        def __init__(s): s.cfg = _Cfg()

    class _Spec:  # noqa
        name = "qwen-1.8b"

    class _Loaded:  # noqa
        def __init__(s): s.model = _Model(); s.tokenizer = None; s.spec = _Spec()

    def fake_load(spec):
        return _Loaded()

    def fake_caches(loaded, instructions, capture_names, batch_size):
        # harmful prompts -> constant 2.0, harmless -> constant 0.5 (per instruction)
        val = 2.0 if instructions and instructions[0].startswith("H!") else 0.5
        for _ in instructions:
            yield _stub_cache(n_layers, seq, d_model, fill=lambda L, p: val)

    backend = rx.ExtractionBackend(load=fake_load, prompt_caches=fake_caches)
    harmful = ["H!a", "H!b", "H!c"]
    harmless = ["x", "y"]
    grid, by_label = rx.run_extraction("qwen-1.8b", harmful, harmless,
                                       n_pos=n_pos, backend=backend)
    assert set(by_label) == {"harmful", "harmless"}
    assert len(by_label["harmful"]) == 3 and len(by_label["harmless"]) == 2
    assert grid.shape == (n_pos, n_layers, d_model)
    # mean_harmful(2.0) - mean_harmless(0.5) = 1.5 everywhere
    assert torch.allclose(grid, torch.full((n_pos, n_layers, d_model), 1.5))


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed"
          + ("" if _HAS_TORCH else "  (torch-gated numeric tests skipped)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
