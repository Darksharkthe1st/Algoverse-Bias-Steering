#!/usr/bin/env python3
"""Phase 3 — compare OUR native refusal vector to the paper's published direction.

Loads a Phase-1 steering vector `(n_layers, d_model)` and, for each layer, reports
its cosine similarity to:
  - the paper's single published `direction.pt` (the canonical refusal direction), and
  - the paper's per-layer `mean_diffs[pos, layer]` grid at the selected position.

Read against the null floor (~1/sqrt(d_model)): a cosine within a few multiples of
that is indistinguishable from chance. This tells us how our own extraction
convention maps onto the paper's refusal direction, and which layer of our vector
to validate in Phase 2.

    python scripts/refusal_native_compare.py \
        --vector runs/<phase1_run>/steering_vector.safetensors --model qwen-1.8b

CPU-only (just loads two tensors); no model needed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer import artifacts, refusal, refusal_compare as rc  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vector", required=True, help="Phase-1 steering_vector.safetensors (n_layers, d_model)")
    ap.add_argument("--model", default="qwen-1.8b", help="catalog key or run-dir the paper direction is for")
    args = ap.parse_args(argv)

    ours = artifacts.load_vector(args.vector).float()          # (n_layers, d_model)
    paper = refusal.load_refusal_direction(args.model)          # RefusalDirection
    run_dir = refusal._resolve(args.model)[0]
    grid = artifacts.load_pt_tensor(
        refusal.artifact_dir(run_dir) / "generate_directions" / "mean_diffs.pt").float()  # (n_pos, n_layers, d_model)

    n_layers, d_model = ours.shape
    floor = rc.null_cosine_std(d_model)
    pos_idx = paper.pos % grid.shape[0]

    cos_dir = rc.per_layer_cosine(ours, paper.direction)                 # vs single published direction
    cos_grid = [rc.cosine(ours[l], grid[pos_idx, l]) for l in range(n_layers)]  # vs paper's per-layer diff

    print(f"model={args.model}  ours={tuple(ours.shape)}  paper direction @ layer {paper.layer} pos {paper.pos}")
    print(f"null cosine std ~ {floor:.4f} (|cos| < {3*floor:.3f} ~ chance)\n")
    print(f"{'layer':>5} {'cos vs published dir':>22} {'cos vs paper grid[l]':>22}")
    for l in range(n_layers):
        mark = "  <- paper layer" if l == paper.layer else ""
        print(f"{l:>5} {cos_dir[l]:>22.4f} {cos_grid[l]:>22.4f}{mark}")

    best_dir = max(range(n_layers), key=lambda l: abs(cos_dir[l]))
    best_grid = max(range(n_layers), key=lambda l: abs(cos_grid[l]))
    print(f"\nbest |cos| vs published direction: layer {best_dir} ({cos_dir[best_dir]:+.4f})")
    print(f"best |cos| vs paper grid:          layer {best_grid} ({cos_grid[best_grid]:+.4f})")
    print(f"at the paper's own layer {paper.layer}: vs dir {cos_dir[paper.layer]:+.4f}, "
          f"vs grid {cos_grid[paper.layer]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
