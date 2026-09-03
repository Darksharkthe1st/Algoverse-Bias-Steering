"""Per-layer mechanism analysis for the linear_add vs adaptive_add_linear result.

Bundles the three measurements that clarify WHY the unconditional linear ramp
(`linear_add`) collapses while the one-sided clamp (`adaptive_add_linear`) stays
coherent at the same coeff. Reuses the SAME committed opinion vector and the
extraction run's saved residuals -- no refit, no judged generation.

Runs in two independently-usable halves so a partial result is still useful:

  OFFLINE (no model load; needs only residuals.safetensors + the vector):
    #3  per-layer signal check -- how strongly does each layer's mean-diff
        direction r_hat_L actually separate opinionated vs neutral activations
        (d-prime). Tests the "early layers carry little opinion signal" theory
        directly: is the early-layer direction real signal or ~noise?
    #4  per-layer relative-dose + clamp-inactivity. For each coeff:
          target_L = coeff * L / n_layers            (1-indexed L; linear_add's
                                                       increment == adaptive's target)
          ratio_L  = target_L / natural_scale_L      (how big the add is vs the
                                                       layer's own projection scale)
          clamp_inactive_frac_L = fraction of examples already at/above target_L
                                  (where adaptive's floor is a NO-OP -> self-limiting)
        This is the quantitative form of "the clamp stops touching the deep,
        high-projection layers, but the unconditional add keeps piling on there."

  FORWARD (needs the GPU + model; single forward pass per condition, NO generation):
    #1  per-layer residual-stream norm ||x_L|| (captured at hook_resid_post) under
        three conditions -- unsteered, linear_add(coeff), adaptive_add_linear(coeff)
        -- over a handful of real prompts. The smoking gun for the blow-up story:
        ||x_L|| should balloon with depth under linear_add and track baseline
        under the clamp. `x_L` = the 4096-dim hidden state flowing between blocks;
        ||x_L|| is its L2 norm, per token position, medianed over real positions.

Every number is a mechanism measurement, not a judged steering result -- there is
no judge and no coeff sweep of outcomes here. "a direction," never "the"
(CLAUDE.md §5). Shapes are asserted at load (CLAUDE.md §6).

Usage (on the GPU box, model already available):
    python experiments/adaptive_vs_fixed/analyze_layer_mechanism.py \
        --run runs/20260901-092009_anchor-qwen3-8b_qwen3-8b \
        --coeffs 8,30 --n-prompts 24

    # offline-only (#3 + #4), no model load, runs anywhere torch is installed:
    python experiments/adaptive_vs_fixed/analyze_layer_mechanism.py \
        --run runs/20260901-092009_anchor-qwen3-8b_qwen3-8b --skip-forward
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # same fix the scratch calibration script needed


def _load_config(path):
    spec = importlib.util.spec_from_file_location("_mech_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def _percentiles(t, qs=(0.25, 0.5, 0.75)):
    import torch
    return {f"p{int(q*100)}": float(torch.quantile(t.float(), q)) for q in qs}


# --------------------------------------------------------------------------- #
# OFFLINE: #3 (signal / d-prime) and #4 (relative dose + clamp inactivity)
# --------------------------------------------------------------------------- #
def offline_resid_analysis(resid_path, vector, contrast, coeffs):
    """Read residuals.safetensors (per-label (n_ex, n_layers, d_model) stacks) and
    the vector; return the #3 and #4 per-layer tables. No model needed."""
    import torch
    from safetensors.torch import load_file

    from src.bias_steer import steering

    pos_label, neg_label = contrast
    resids = load_file(str(resid_path))
    for lbl in (pos_label, neg_label):
        assert lbl in resids, (
            f"residuals.safetensors has no '{lbl}' bucket (found {list(resids)}). "
            f"Contrast is {contrast}; cannot run the signal check without both poles."
        )
    n_layers, d_model = vector.shape
    for lbl, t in resids.items():
        assert t.ndim == 3 and t.shape[1:] == (n_layers, d_model), (
            f"residual bucket {lbl!r}: expected (n_ex, {n_layers}, {d_model}), "
            f"got {tuple(t.shape)} (CLAUDE.md §6)"
        )

    r_hat = steering.unit_perlayer(vector).float()                 # (n_layers, d_model)
    pos = resids[pos_label].float()                                # (n_pos, L, d)
    neg = resids[neg_label].float()                                # (n_neg, L, d)

    # Project every example's per-layer residual onto that layer's unit direction.
    proj_pos = torch.einsum("eld,ld->el", pos, r_hat)              # (n_pos, L)
    proj_neg = torch.einsum("eld,ld->el", neg, r_hat)              # (n_neg, L)
    all_proj = torch.cat([proj_pos, proj_neg], dim=0)              # (n_all, L)

    # #3: per-layer separability of the two classes along r_hat_L (a signal/noise
    # readout of the direction the steering actually uses). d-prime = standardized
    # mean gap; ~0 => the layer's direction barely distinguishes the classes.
    mp, mn = proj_pos.mean(0), proj_neg.mean(0)
    vp, vn = proj_pos.var(0, unbiased=False), proj_neg.var(0, unbiased=False)
    dprime = (mp - mn) / (0.5 * (vp + vn)).clamp_min(1e-12).sqrt()  # (L,)

    # #4: natural per-layer projection scale (magnitude), pooled over both classes.
    natural_scale = all_proj.abs().median(0).values               # (L,) median |proj|

    per_coeff = {}
    for c in coeffs:
        L_idx = torch.arange(1, n_layers + 1, dtype=torch.float32)
        target = c * L_idx / n_layers                              # linear_add increment == adaptive target
        ratio = target / natural_scale.clamp_min(1e-12)
        # POS arm (floor): clamp is a no-op wherever the projection already >= target.
        clamp_inactive = (all_proj >= target).float().mean(0)      # (L,)
        per_coeff[str(c)] = {
            "target_per_layer": target.tolist(),
            "ratio_target_over_natural": ratio.tolist(),
            "clamp_inactive_frac": clamp_inactive.tolist(),
        }

    return {
        "n_layers": n_layers,
        "n_pos": int(pos.shape[0]),
        "n_neg": int(neg.shape[0]),
        "vector_norm_per_layer": vector.norm(dim=-1).float().tolist(),   # ||vector[L]||
        "dprime_per_layer": dprime.tolist(),                             # #3
        "natural_proj_scale_per_layer": natural_scale.tolist(),         # median |proj|
        "natural_proj_pos_median": proj_pos.median(0).values.tolist(),
        "natural_proj_neg_median": proj_neg.median(0).values.tolist(),
        "per_coeff": per_coeff,                                          # #4
    }


# --------------------------------------------------------------------------- #
# FORWARD: #1 (residual-stream norm under each method) -- needs the model
# --------------------------------------------------------------------------- #
def forward_norm_analysis(config, vector, coeffs, n_prompts, batch_size):
    """Single forward pass (no generation) per condition; capture ||x_L|| at every
    layer's hook_resid_post over real (non-pad, non-first) token positions."""
    import torch

    from src.bias_steer import datasets, models, steering
    from src.bias_steer.registry import DATASETS, MODELS

    model_key = config.models[0]
    spec = MODELS[model_key]
    examples = DATASETS[config.dataset.name](config.dataset)
    examples = datasets.sample(examples, config.sample)[:n_prompts]
    prompts = [e.prompt for e in examples]
    print(f"[#1] loading {spec.hf_id} @ {spec.revision} ...", flush=True)
    loaded = models.load_model(spec)
    model = loaded.model
    n_layers = model.cfg.n_layers
    vector = vector.to(loaded.device)
    post_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    pad_id = model.tokenizer.pad_token_id

    def run_condition(steer_hooks):
        # Accumulate per-layer norms at real positions across batches.
        per_layer = [[] for _ in range(n_layers)]
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            _, strs = models.render_prompts(loaded, chunk, config.system_prompt)
            tokens = model.to_tokens(strs)                          # (b, seq), left-padded
            real = tokens != pad_id if pad_id is not None else torch.ones_like(tokens, dtype=torch.bool)
            # Drop the first real token per row (BOS / attention-sink outlier — see
            # the calibration note in GPU_RUN_LOG.md).
            first_real = real.float().argmax(dim=1)
            real[torch.arange(real.shape[0]), first_real] = False

            grabbed = {}

            def cap(value, hook):                                   # (b, seq, d_model)
                grabbed[hook.name] = value.norm(dim=-1).detach().float().cpu()
                return value

            cap_hooks = [(nm, cap) for nm in post_names]
            with torch.no_grad(), model.hooks(fwd_hooks=steer_hooks + cap_hooks):
                model(tokens, return_type=None)
            m = real.cpu()
            for L, nm in enumerate(post_names):
                per_layer[L].append(grabbed[nm][m])
        out = []
        for L in range(n_layers):
            vals = torch.cat(per_layer[L])
            out.append({"median": float(vals.median()), **_percentiles(vals)})
        return out

    conditions = {"unsteered": run_condition([])}
    for c in coeffs:
        lin = steering.apply_linear_add_perlayer(model, vector, coeff=c)
        adp = steering.apply_adaptive_additive_linear_floor(model, vector, coeff=c)
        print(f"[#1] forward pass: linear_add c={c} and adaptive_add_linear c={c}", flush=True)
        conditions[f"linear_add_c{c}"] = run_condition(lin)
        conditions[f"adaptive_add_linear_c{c}"] = run_condition(adp)

    return {"model": model_key, "n_prompts": len(prompts), "conditions": conditions}


# --------------------------------------------------------------------------- #
def _print_summary(offline, forward, coeffs):
    n = offline["n_layers"]
    dp = offline["dprime_per_layer"]
    vn = offline["vector_norm_per_layer"]
    print("\n================ #3 signal (d-prime of r_hat_L: opinion vs neutral) ================")
    print(f"  layer 1:   d'={dp[0]:+.3f}   ||vector||={vn[0]:.3f}")
    print(f"  layer {n//2:>2}:  d'={dp[n//2]:+.3f}   ||vector||={vn[n//2]:.3f}")
    print(f"  layer {n:>2}:  d'={dp[-1]:+.3f}   ||vector||={vn[-1]:.3f}")
    print(f"  (near-0 d' at shallow layers => that layer's direction barely encodes the contrast)")

    print("\n================ #4 clamp-inactivity (fraction already past target) ================")
    for c in coeffs:
        frac = offline["per_coeff"][str(c)]["clamp_inactive_frac"]
        cross = next((L + 1 for L, f in enumerate(frac) if f >= 0.5), None)
        print(f"  coeff={c}: clamp inactive for >=50% of examples from layer "
              f"{cross if cross else '(never)'} onward "
              f"(L1={frac[0]:.2f}, L{n}={frac[-1]:.2f}) "
              f"-> deep layers self-limit under adaptive, but linear_add still adds there")

    if forward is not None:
        print("\n================ #1 residual-stream norm ||x_L|| (deepest layer) ================")
        deep = {k: v[-1]["median"] for k, v in forward["conditions"].items()}
        base = deep["unsteered"]
        for k, v in deep.items():
            print(f"  {k:>26}: ||x_last|| median={v:9.1f}   ({v / base:5.1f}x unsteered)")
        print("  (linear_add >> adaptive ~ unsteered would confirm the unbounded-blow-up mechanism)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="extraction run dir holding steering_vector.safetensors + residuals.safetensors")
    ap.add_argument("--residuals", default=None, help="override path to residuals.safetensors")
    ap.add_argument("--vector", default=None, help="override path to steering_vector.safetensors")
    ap.add_argument("--config", default="configs/exp/linear_add_c30_qwen3_8b.py",
                    help="config supplying the model + dataset for #1 (any linear_add config works)")
    ap.add_argument("--coeffs", default="8,30", help="comma-separated coeffs to analyze")
    ap.add_argument("--n-prompts", type=int, default=24, help="#1 prompt count (forward passes only)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-forward", action="store_true", help="run only the offline #3/#4 (no model load)")
    ap.add_argument("--out", default="experiments/adaptive_vs_fixed/mechanism_layer_scaling.json")
    args = ap.parse_args()

    from src.bias_steer import artifacts
    run = Path(args.run)
    vec_path = Path(args.vector) if args.vector else run / "steering_vector.safetensors"
    resid_path = Path(args.residuals) if args.residuals else run / "residuals.safetensors"
    coeffs = [float(c) if "." in c else int(c) for c in args.coeffs.split(",")]

    config = _load_config(args.config)
    # Contrast = (positive_pole, negative_pole) = judge.labels[1], labels[0] (experiment._contrast).
    labels = config.judge.labels
    contrast = (labels[1], labels[0])

    vector = artifacts.load_vector(str(vec_path))
    assert vector.ndim == 2, f"vector must be (n_layers, d_model); got {tuple(vector.shape)} (CLAUDE.md §6)"

    if not resid_path.is_file():
        raise SystemExit(
            f"residuals not found at {resid_path}. This file is git-ignored (bulky), so it "
            f"lives only on the box that ran extraction. If the anchor run predates save_residuals, "
            f"re-extract (or point --residuals at wherever the (n_ex, n_layers, d_model) stacks are). "
            f"#1 (--skip-forward-able) does NOT need it; #3/#4 do."
        )
    offline = offline_resid_analysis(resid_path, vector, contrast, coeffs)

    forward = None
    if not args.skip_forward:
        forward = forward_norm_analysis(config, vector, coeffs, args.n_prompts, args.batch_size)

    result = {"run": str(run), "coeffs": coeffs, "contrast": contrast,
              "offline_resid_analysis": offline, "forward_norm_analysis": forward}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    _print_summary(offline, forward, coeffs)


if __name__ == "__main__":
    main()
