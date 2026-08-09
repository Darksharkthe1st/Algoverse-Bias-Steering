"""Parity ladder, rung 3: cosine-compare a fresh steering vector to the archive.

Loads an archived `*_steer_vec.pkl` (a torch tensor, `[n_layers, d_model]`) and a
run's `steering_vector.safetensors`, and reports per-layer cosine similarity.

Read the per-layer numbers, not just the mean. The layers carry wildly different
norms (the archive notes a ~1400x spread), so a norm-weighted mean can look fine
while early layers are uncorrelated. A faithful `capture` + `build` should show
high cosine across essentially all layers.

Interpretation:
- uniformly high              -> capture/build math matches
- high late, low early        -> early layers are near-zero and noise-dominated;
                                 check the norms column before worrying
- uniformly ~0                -> capture reads the wrong hook point or the wrong
                                 token reduction
- uniformly ~-1               -> the contrast poles are swapped (mean(neg) - mean(pos))

Usage:
    python tools/parity_vector.py <run_dir_or_safetensors> [--archive PATH]
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_ARCHIVE = "experiments/best_vecs/log_103_Qwen1.5-1.8B-Chat_steer_vec.pkl"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", help="run directory, or a steering_vector.safetensors path")
    p.add_argument("--archive", default=DEFAULT_ARCHIVE, help="archived *_steer_vec.pkl")
    args = p.parse_args(argv)

    import torch
    from src.bias_steer.artifacts import load_vector

    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    fresh_path = run if run.suffix == ".safetensors" else run / "steering_vector.safetensors"
    fresh = load_vector(fresh_path).float()

    archive_path = Path(args.archive)
    if not archive_path.is_absolute():
        archive_path = root / archive_path
    with open(archive_path, "rb") as f:
        old = pickle.load(f)
    old = old.float().cpu()
    fresh = fresh.cpu()

    print(f"fresh   : {fresh_path}  shape {tuple(fresh.shape)}")
    print(f"archive : {archive_path.name}  shape {tuple(old.shape)}")
    if fresh.shape != old.shape:
        print(f"\nerror: shape mismatch — cannot compare")
        return 1

    cos = torch.nn.functional.cosine_similarity(fresh, old, dim=-1)
    print(f"\n  layer   cosine    |fresh|      |archive|")
    for i in range(len(cos)):
        print(f"  {i:5d}   {cos[i]:+.4f}   {fresh[i].norm():10.4f}   {old[i].norm():10.4f}")

    # Norm-weighted mean too: the layers that actually dominate the injected
    # signal are the high-norm ones, so an unweighted mean over-counts the noisy
    # near-zero early layers.
    w = old.norm(dim=-1)
    weighted = float((cos * w).sum() / w.sum())
    print(f"\n  mean cosine (unweighted): {float(cos.mean()):+.4f}")
    print(f"  mean cosine (norm-weighted): {weighted:+.4f}")
    print(f"  layers with cosine > 0.9 : {int((cos > 0.9).sum())}/{len(cos)}")
    print(f"  layers with cosine < 0   : {int((cos < 0).sum())}/{len(cos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
