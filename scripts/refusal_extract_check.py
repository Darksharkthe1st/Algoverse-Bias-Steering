#!/usr/bin/env python3
"""Chunk A — extract the refusal candidate grid and validate it by cosine vs the
paper's committed `mean_diffs.pt`, cell-by-cell.

THIS SCRIPT LOADS A MODEL. Do not run it in the planning/CPU environment — run it
on the Lambda GPU box (see docs/06-refusal-generation.md). It:

  1. reproduces upstream's exact instruction sets:
       load_and_sample_repro()  (random.seed(42); random.sample(...))
       then filter_data via refusal_score  (filter_train=True by default upstream)
  2. extracts our grid (n_pos, n_layers, d_model) = mean_harmful - mean_harmless,
     reading resid_pre at the last n_pos post-instruction prompt tokens (no gen),
  3. loads their mean_diffs.pt (fetched artifact), casts both to float64,
  4. prints cosine similarity per (pos, layer) and checks:
       >= 0.999 per cell  -> recipe exactly right
       >= 0.95            -> acceptable, investigate outliers
       <  0.95            -> real problem (formatting/positions/filtering)
     and reports the cell at the model's published (pos, layer).

Usage:
    python scripts/refusal_extract_check.py                 # qwen-1.8b (anchor)
    python scripts/refusal_extract_check.py --model gemma-2b --no-filter
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen-1.8b", help="catalog key or run-dir name")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-filter", action="store_true",
                    help="skip the model-based filter (upstream default is filter_train=True; "
                         "skipping will lower cosine because the mean is over a different set)")
    args = ap.parse_args(argv)

    import torch
    from src.bias_steer.registry import MODELS
    from src.bias_steer import models, refusal
    from src.bias_steer import refusal_extract as rx

    model_key = rx._resolve_model_key(args.model)
    print(f"[1/5] loading {model_key} ...")
    loaded = models.load_model(MODELS[model_key])
    n_pos = rx.n_positions(loaded)
    n_layers = loaded.model.cfg.n_layers
    print(f"      n_pos={n_pos}  n_layers={n_layers}  device={loaded.model.cfg.device}")

    print("[2/5] sampling instructions (seed 42) ...")
    sampled = rx.load_and_sample_repro(n_train=args.n_train, n_val=args.n_val)
    harmful, harmless = sampled["harmful_train"], sampled["harmless_train"]

    if not args.no_filter:
        print("[3/5] filtering (refusal_score; keep harmful>0, harmless<0) ...")
        rt = rx.refusal_toks(model_key)
        h_scores = rx.get_refusal_scores(loaded, harmful, rt, args.batch_size)
        l_scores = rx.get_refusal_scores(loaded, harmless, rt, args.batch_size)
        harmful = [x for x, s in zip(harmful, h_scores) if s > 0]
        harmless = [x for x, s in zip(harmless, l_scores) if s < 0]
        print(f"      kept harmful={len(harmful)}/{args.n_train}  harmless={len(harmless)}/{args.n_train}")
    else:
        print("[3/5] filtering SKIPPED (--no-filter)")

    print("[4/5] extracting grid (prompt resid_pre, last n_pos, no generation) ...")
    grid, _ = rx.run_extraction(model_key, harmful, harmless, n_pos=n_pos,
                                batch_size=args.batch_size)
    grid = grid.to(torch.float64).cpu()

    print("[5/5] cosine vs committed mean_diffs.pt ...")
    theirs = refusal.artifacts.load_pt_tensor(
        refusal.artifact_dir(refusal.MODEL_TO_RUN_DIR[model_key]) / "generate_directions" / "mean_diffs.pt"
    ).to(torch.float64).cpu()
    assert theirs.shape == grid.shape, f"shape ours {tuple(grid.shape)} != theirs {tuple(theirs.shape)}"

    cos = torch.nn.functional.cosine_similarity(
        grid.reshape(n_pos * n_layers, -1), theirs.reshape(n_pos * n_layers, -1), dim=-1
    ).reshape(n_pos, n_layers)

    print("\ncosine per (pos_index, layer):")
    for i in range(n_pos):
        row = "  ".join(f"{cos[i, L].item():.3f}" for L in range(n_layers))
        print(f"  pos[{i}] (tok {i - n_pos:+d}): {row}")
    worst = cos.min().item()
    print(f"\nmin={worst:.4f}  mean={cos.mean().item():.4f}  max={cos.max().item():.4f}")

    meta = refusal.load_refusal_direction(model_key)
    sel_pos_idx = meta.pos + n_pos  # token pos -> axis index
    sel_cos = cos[sel_pos_idx, meta.layer].item()
    print(f"selected cell (pos={meta.pos}, layer={meta.layer}) cosine = {sel_cos:.5f}")

    verdict = ("EXACT (>=0.999)" if worst >= 0.999
               else "ACCEPTABLE (>=0.95), investigate outliers" if worst >= 0.95
               else "PROBLEM (<0.95): debug formatting/positions/filtering")
    print(f"\nVERDICT (worst cell): {verdict}")
    return 0 if worst >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
