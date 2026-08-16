"""Is our grid misaligned against theirs on the position or layer axis?

Extract once, then cross-correlate: for each of our (pos,layer) cells find which
of THEIR cells it best matches. If our pos[i] best-matches their pos[i+/-1], the
position axis is off by one; likewise for layers.
"""
import sys; sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")
import torch
from src.bias_steer.registry import MODELS
from src.bias_steer import models, refusal
from src.bias_steer import refusal_extract as rx

key = "qwen-1.8b"
loaded = models.load_model(MODELS[key])
n_pos, n_layers = rx.n_positions(loaded), loaded.model.cfg.n_layers
s = rx.load_and_sample_repro(n_train=128, n_val=32)
grid, _ = rx.run_extraction(key, s["harmful_train"], s["harmless_train"], n_pos=n_pos, batch_size=16)
grid = grid.to(torch.float64).cpu()
theirs = refusal.artifacts.load_pt_tensor(
    refusal.artifact_dir(refusal.MODEL_TO_RUN_DIR[key]) / "generate_directions" / "mean_diffs.pt"
).to(torch.float64).cpu()

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()

print("\n=== position-axis alignment (at layer 15) ===")
print("      " + "".join(f" their_p{j}" for j in range(n_pos)))
for i in range(n_pos):
    row = "".join(f"  {cos(grid[i,15], theirs[j,15]):+.3f}" for j in range(n_pos))
    best = max(range(n_pos), key=lambda j: cos(grid[i,15], theirs[j,15]))
    print(f"our_p{i}{row}   -> best their_p{best}{'  <-- MATCH' if best==i else '  <-- MISALIGNED'}")

print("\n=== layer-axis alignment (at pos index 3 = tok -2) ===")
mis = 0
for L in range(1, n_layers):
    best = max(range(1, n_layers), key=lambda M: cos(grid[3,L], theirs[3,M]))
    if best != L:
        mis += 1
        if mis <= 6:
            print(f"  our layer {L:2d} -> best their layer {best:2d}  (cos {cos(grid[3,L],theirs[3,best]):.3f}"
                  f" vs same-layer {cos(grid[3,L],theirs[3,L]):.3f})")
print(f"  layers whose best match is NOT the same index: {mis}/{n_layers-1}")

print("\n=== norm ratio ours/theirs (pos 3) ===")
for L in [1, 5, 10, 15, 20, 23]:
    print(f"  layer {L:2d}: ours {grid[3,L].norm():.4f}  theirs {theirs[3,L].norm():.4f}"
          f"  ratio {(grid[3,L].norm()/theirs[3,L].norm()):.3f}")
