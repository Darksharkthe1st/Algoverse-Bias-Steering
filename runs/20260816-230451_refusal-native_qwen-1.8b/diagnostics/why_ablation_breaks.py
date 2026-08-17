"""Why does ablating our L12/L15 row destroy the model while L19/L22 does not?

Hypothesis: our mean-diff is taken over MEAN-POOLED response residuals, so each
per-layer row keeps a large component along the residual stream's dominant mean
direction. Directional ablation removes the direction at EVERY layer and site, so
ablating a row that overlaps that mean direction removes something computation
needs everywhere -> token-loop collapse.

Test: cosine of each of our rows against the mean residual at the same layer
(computed from the run's own saved residuals), next to the paper's direction as
the reference for "clean" (their extraction is a difference at ONE prompt
position, so it should overlap the mean far less).

CPU only. Also re-scores the Phase-1 eval half for degeneration.
"""

import sys

sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from src.bias_steer import artifacts, refusal  # noqa: E402

RUN = ("/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering/"
       "runs/20260816-230451_refusal-native_qwen-1.8b")

ours = artifacts.load_vector(f"{RUN}/steering_vector.safetensors").float()   # (24, 2048)
paper = refusal.load_refusal_direction("qwen-1.8b").direction.float()        # (2048,)

resids = load_file(f"{RUN}/residuals.safetensors")
print("residual tensors:", {k: tuple(v.shape) for k, v in resids.items()})

# Grand mean residual per layer over BOTH buckets = the "dominant mean direction".
stacked = torch.cat([v.float().reshape(-1, *v.shape[-2:]) for v in resids.values()], dim=0)
mean_resid = stacked.mean(dim=0)          # (n_layers, d_model)
print("grand-mean residual:", tuple(mean_resid.shape))


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a[None].float(), b[None].float())[0])


print(f"\n{'layer':>5} {'cos(ours_l, mean_resid_l)':>26} {'cos(paper, mean_resid_l)':>26} {'||ours_l||':>11}")
for l in range(ours.shape[0]):
    print(f"{l:>5} {cos(ours[l], mean_resid[l]):>26.4f} "
          f"{cos(paper, mean_resid[l]):>26.4f} {float(ours[l].norm()):>11.3f}")
