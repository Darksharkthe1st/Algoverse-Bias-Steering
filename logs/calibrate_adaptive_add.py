"""Measure baseline per-position projection (x . r_hat_L) on real Qwen3-8b resid_pre
activations, for a handful of prompts, to pick a sane `target` for adaptive_add
before trusting its judged numbers (see configs/exp/adaptive_add_qwen3_8b.py).

Loads the model once (unsteered), renders the SAME prompts/system-prompt path the
real experiment uses, and reports per-layer projection stats (mean/std/percentiles
of x . r_hat_L across positions and prompts) so `target` is chosen at a scale
comparable to what the residual stream already exhibits at each layer.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.bias_steer import artifacts, models, steering
from src.bias_steer.config import DatasetSpec, SampleSpec
from src.bias_steer.datasets import DATASETS, sample as sample_fn
from src.bias_steer.registry import MODELS

VECTOR_PATH = "runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors"
DATASET_PATH = "datasets/Snapshots/log_103_comparison_200.json"
N_PROMPTS = 8
SYSTEM_PROMPT = ""  # filled from the real config below


def main():
    from configs.exp.adaptive_add_qwen3_8b import config as real_config

    vector = artifacts.load_vector(VECTOR_PATH)
    n_layers, d_model = vector.shape
    print(f"vector shape: {tuple(vector.shape)}", file=sys.stderr)

    spec = MODELS["qwen3-8b"]
    loaded = models.load_model(spec)
    r_hat = steering.unit_perlayer(vector).to(loaded.device)

    ds_spec = DatasetSpec(name="snapshot", path=DATASET_PATH, train_split=0.5, shuffle=False)
    examples = DATASETS["snapshot"](ds_spec)
    examples = sample_fn(examples, real_config.sample)
    prompts = [e.prompt for e in examples[:N_PROMPTS]]

    sys_prompt = real_config.system_prompt
    token_lists, strs = models.render_prompts(loaded, prompts, sys_prompt)
    tokens = loaded.model.to_tokens(strs)

    resid_names = [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = loaded.model.run_with_cache(
            tokens, names_filter=lambda n: n in resid_names
        )

    print("\nlayer, n, mean, std, min, p25, p50, p75, max")
    overall_vals = []
    for l in range(n_layers):
        x = cache[f"blocks.{l}.hook_resid_pre"]  # (batch, seq, d_model)
        proj = (x.float() @ r_hat[l].float()).flatten()
        vals = proj.detach().cpu()
        q = torch.quantile(vals, torch.tensor([0.25, 0.5, 0.75]))
        print(f"{l}, {len(vals)}, {vals.mean():.4f}, {vals.std():.4f}, "
              f"{vals.min():.4f}, {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {vals.max():.4f}")
        overall_vals.append(vals)

    overall = torch.cat(overall_vals)
    print(f"\noverall: n={len(overall)} mean={overall.mean():.4f} std={overall.std():.4f} "
          f"min={overall.min():.4f} max={overall.max():.4f}")


if __name__ == "__main__":
    main()
