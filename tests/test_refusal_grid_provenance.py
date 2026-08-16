"""Chunk A0 — torch-free provenance of the refusal candidate grid.

    python3 tests/test_refusal_grid_provenance.py

Proves we understand the exact layout of the paper's `mean_diffs.pt` and that
`direction.pt` is the storage-view `mean_diffs[pos, layer]` named in
`direction_metadata.json` — decoded from the pickle with the stdlib, NO torch and
NO model. Gated only on the artifacts being fetched (scripts/fetch_refusal_artifacts.py).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from tools.refusal_grid_provenance import parse_rebuild_tensor  # noqa: E402

_ARTIFACT_ROOT = Path(_REPO_ROOT) / "third_party" / "refusal_direction" / "runs"

# Ground truth from each committed direction_metadata.json (layer, pos), and the
# grid shape (n_pos, n_layers, d_model) proven in Chunk A0.
_EXPECTED = {
    "qwen-1_8b-chat":            {"meta": (15, -2), "shape": (5, 24, 2048)},
    "gemma-2b-it":              {"meta": (10, -2), "shape": (5, 18, 2048)},
    "yi-6b-chat":               {"meta": (20, -5), "shape": (6, 32, 4096)},
    "meta-llama-3-8b-instruct": {"meta": (12, -5), "shape": (5, 32, 4096)},
    "llama-2-7b-chat-hf":       {"meta": (14, -1), "shape": (6, 32, 4096)},
}


def _fetched():
    if not _ARTIFACT_ROOT.exists():
        return []
    return sorted(
        d.name for d in _ARTIFACT_ROOT.iterdir()
        if (d / "generate_directions" / "mean_diffs.pt").exists()
        and (d / "direction.pt").exists()
    )


def test_grid_is_float64_and_expected_shape():
    fetched = _fetched()
    if not fetched:
        print("      (skipped: no artifacts fetched — run scripts/fetch_refusal_artifacts.py)")
        return
    for name in fetched:
        grid = parse_rebuild_tensor(_ARTIFACT_ROOT / name / "generate_directions" / "mean_diffs.pt")
        assert grid["dtype"] == "DoubleStorage", f"{name}: grid dtype {grid['dtype']} != DoubleStorage"
        assert grid["storage_offset"] == 0, f"{name}: grid storage_offset {grid['storage_offset']} != 0"
        if name in _EXPECTED:
            assert grid["shape"] == _EXPECTED[name]["shape"], \
                f"{name}: shape {grid['shape']} != {_EXPECTED[name]['shape']}"
            n_pos, n_layers, d_model = grid["shape"]
            # C-contiguous stride: (n_layers*d_model, d_model, 1)
            assert grid["stride"] == (n_layers * d_model, d_model, 1), \
                f"{name}: stride {grid['stride']} not C-contiguous for {grid['shape']}"


def test_direction_is_storage_view_at_metadata_pos_layer():
    fetched = _fetched()
    if not fetched:
        print("      (skipped: no artifacts fetched)")
        return
    for name in fetched:
        grid = parse_rebuild_tensor(_ARTIFACT_ROOT / name / "generate_directions" / "mean_diffs.pt")
        direc = parse_rebuild_tensor(_ARTIFACT_ROOT / name / "direction.pt")
        meta = json.loads((_ARTIFACT_ROOT / name / "direction_metadata.json").read_text())
        n_pos, n_layers, d_model = grid["shape"]

        # same underlying storage (the whole grid) -> direction.pt is a view
        assert direc["storage_numel"] == grid["storage_numel"], \
            f"{name}: direction storage numel != grid storage numel"
        assert direc["shape"] == (d_model,), f"{name}: direction shape {direc['shape']}"

        # decode offset -> (pos_index, layer); position index i maps to token (i - n_pos)
        assert direc["storage_offset"] % d_model == 0
        row = direc["storage_offset"] // d_model
        pos_index, layer = divmod(row, n_layers)
        pos_token = pos_index - n_pos
        assert (layer, pos_token) == (int(meta["layer"]), int(meta["pos"])), \
            f"{name}: decoded (layer={layer}, pos={pos_token}) != meta ({meta['layer']}, {meta['pos']})"
        if name in _EXPECTED:
            assert (layer, pos_token) == _EXPECTED[name]["meta"], \
                f"{name}: (layer,pos)={(layer, pos_token)} != {_EXPECTED[name]['meta']}"
        print(f"      {name:28s} grid={grid['shape']} -> direction @ (pos={pos_token}, layer={layer})")


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
