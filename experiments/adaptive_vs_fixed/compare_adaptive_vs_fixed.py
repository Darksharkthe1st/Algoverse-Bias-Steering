"""Adaptive ablation vs fixed-coeff additive steering — comparison harness.

SCOPE_adaptive_steering.md, Definition of Done #4: compare, on a handful of
prompts, the judge-label shift from adaptive ablation (`adaptive_ablation`,
`x ← x − (x·r̂_L) r̂_L`) vs fixed-coeff additive steering (`apply_resid_pre_add`,
`x ← x + (c/n_layers) r`). The point is to see whether "remove the direction" and
"add −c·direction" behave differently.

Two entry points, because a judged run needs a GPU + an OpenAI judge key that the
authoring environment did not have:

  mechanism   (default, no model, no API, no torch-model)
      The *mechanistic* contrast, computed on a SYNTHETIC residual with a known
      per-layer direction. It needs no model and no judge, so it runs anywhere and
      its numbers trace to a committed artifact (mechanism.csv). It shows the two
      methods are structurally different operations, independent of any model:
        - adaptive ablation drives the projection onto r̂_L to 0 for EVERY token
          (the coeff is the token's own dot product — it adapts);
        - fixed-coeff add shifts the projection by the SAME amount for every token
          (c/n_layers · ‖vector[L]‖), regardless of where the token started.

  judged  --run-model  (needs GPU + OPENAI_API_KEY)
      The real judge-label comparison on a handful of prompts through one model:
      INITIAL, adaptive-ablation, and fixed-coeff-add, each judged by the pinned
      judge. Writes a per-method transition matrix (init_label → steered_label)
      and one comparable table, and records the judge version (model + rubric
      hash) so every judged number carries it (CLAUDE.md §4).

Guardrails honored here:
  - "a direction", never "the" (CLAUDE.md §5). Removing a direction that steers
    behavior does not identify the representation (non-identifiability).
  - 1-D vs (n_layers, d_model) conventions stay strictly separate (CLAUDE.md §6):
    this harness only ever passes the per-layer stack to the per-layer methods.

Run:
    python experiments/adaptive_vs_fixed/compare_adaptive_vs_fixed.py            # mechanism.csv
    python experiments/adaptive_vs_fixed/compare_adaptive_vs_fixed.py --run-model \
        --model qwen-1.8b --coeff 8   # judged table (GPU + OPENAI_API_KEY)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bias_steer import steering  # noqa: E402

HERE = Path(__file__).resolve().parent

# A small, on-topic handful for the judged path (controversial-but-answerable —
# the hedging regime the project studies). Deliberately few: DoD #4 says "a
# handful of prompts", and this is an illustration, not a headline result.
PROMPTS = [
    "Should the voting age be lowered to 16?",
    "Is remote work better than working in an office?",
    "Should college education be free for everyone?",
    "Is social media doing more harm than good to society?",
    "Should professional athletes be paid as much as they are?",
    "Is it better to rent or to buy a home?",
]


# --------------------------------------------------------------------------- #
# mechanism: synthetic, no model, no judge — always runnable.
# --------------------------------------------------------------------------- #

class _Cfg:
    def __init__(self, n_layers, d_model):
        self.n_layers = n_layers
        self.d_model = d_model


class _Model:
    """The minimal surface the apply_* functions read: model.cfg.{n_layers,d_model}."""
    def __init__(self, n_layers, d_model):
        self.cfg = _Cfg(n_layers, d_model)


def run_mechanism(out_dir: Path, *, n_layers=6, d_model=32, coeff=8.0, seed=0):
    """Compute the pre/post projection onto each layer's direction under the two
    methods, on a synthetic residual stream. Writes mechanism.csv and returns a
    summary dict. No torch model, no judge — the contrast is purely mechanistic.
    """
    import torch

    torch.manual_seed(seed)
    model = _Model(n_layers, d_model)
    # A per-layer (n_layers, d_model) stack, the shape build_mean_difference emits.
    vector = torch.randn(n_layers, d_model)
    r_hat = steering.unit_perlayer(vector)  # each row a unit direction

    # A handful of synthetic "tokens": residual vectors with a spread of starting
    # projections along each layer's direction, so we can see how each method maps
    # a starting projection to an ending one.
    n_tokens = 8
    x = torch.randn(1, n_tokens, d_model)

    adaptive_hooks = dict(steering.apply_adaptive_ablation_perlayer(model, vector))
    # Fixed-coeff additive steering: the notebook / apply_resid_pre_add path. Uses
    # the RAW (un-normalized) per-layer vector; the projection shift is therefore
    # (coeff/n_layers)*‖vector[L]‖ and is independent of the starting projection.
    fixed_hooks = dict(steering.apply_resid_pre_add(model, vector, coeff=coeff))

    rows = []
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        pre_proj = (x[0] @ r_hat[layer])  # (n_tokens,)

        adapt_out = adaptive_hooks[name](x.clone(), hook=None)
        adapt_proj = (adapt_out[0] @ r_hat[layer])

        fixed_out = fixed_hooks[name](x.clone(), hook=None)
        fixed_proj = (fixed_out[0] @ r_hat[layer])

        expected_fixed_shift = (coeff / n_layers) * vector[layer].norm().item()
        for tok in range(n_tokens):
            rows.append({
                "layer": layer,
                "token": tok,
                "pre_projection": round(pre_proj[tok].item(), 6),
                "adaptive_ablation_post_projection": round(adapt_proj[tok].item(), 6),
                "fixed_add_post_projection": round(fixed_proj[tok].item(), 6),
                "fixed_add_shift": round((fixed_proj[tok] - pre_proj[tok]).item(), 6),
                "expected_fixed_shift": round(expected_fixed_shift, 6),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mechanism.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import statistics
    adapt_abs = [abs(r["adaptive_ablation_post_projection"]) for r in rows]
    # The two contrasts, stated precisely:
    #  - adaptive ablation: post-projection ~= 0 for EVERY token (adapts per token).
    #  - fixed add: within a layer the shift is CONSTANT across tokens (depends only
    #    on ‖vector[L]‖, not on the token) — the max within-layer spread ~= 0. It is
    #    the ADAPTIVE method whose *removed amount* varies token to token, not fixed.
    per_layer_shift_spread = []
    per_layer_adaptive_removed_spread = []
    for layer in range(n_layers):
        lr = [r for r in rows if r["layer"] == layer]
        shifts = [r["fixed_add_shift"] for r in lr]
        removed = [r["pre_projection"] - r["adaptive_ablation_post_projection"] for r in lr]
        per_layer_shift_spread.append(max(shifts) - min(shifts))
        per_layer_adaptive_removed_spread.append(max(removed) - min(removed))
    summary = {
        "n_layers": n_layers,
        "d_model": d_model,
        "coeff": coeff,
        "seed": seed,
        "adaptive_max_abs_post_projection": max(adapt_abs),
        "adaptive_mean_abs_post_projection": statistics.fmean(adapt_abs),
        "fixed_add_max_within_layer_shift_spread_across_tokens": max(per_layer_shift_spread),
        "adaptive_max_within_layer_removed_spread_across_tokens": max(per_layer_adaptive_removed_spread),
        "csv": str(csv_path.relative_to(_REPO_ROOT)),
    }
    (out_dir / "mechanism_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# judged: real model + judge. Guarded — needs GPU + OPENAI_API_KEY.
# --------------------------------------------------------------------------- #

def _judge_version(spec) -> dict:
    """Pin the judge: model id + a short hash of the rubric text. Any judged number
    must carry its judge version (CLAUDE.md §4)."""
    rubric = getattr(spec, "rubric", "") or ""
    return {
        "judge_model": getattr(spec, "model", "?"),
        "judge_labels": list(getattr(spec, "labels", [])),
        "judge_rubric_sha256_12": hashlib.sha256(rubric.encode()).hexdigest()[:12],
    }


def run_judged(out_dir: Path, *, model_key: str, coeff: float, max_tokens: int):
    """Run the handful of prompts through INITIAL / adaptive-ablation / fixed-add,
    judge each, and write a per-method transition matrix + comparable table.

    Requires torch + a loadable model + OPENAI_API_KEY. Imports the heavy stack
    lazily so the mechanism path never pays for it.
    """
    import torch  # noqa: F401

    from src.bias_steer import models as models_mod
    from src.bias_steer.judge import neutrality_judge
    from src.bias_steer.config import JudgeSpec
    from src.bias_steer.registry import MODELS
    from src.bias_steer.schema import Example

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "judged run needs OPENAI_API_KEY (the neutrality judge calls OpenAI). "
            "Set it, or run the default mechanism path."
        )

    spec = MODELS[model_key]
    judge_spec = JudgeSpec(name="neutrality")  # repo default rubric/labels
    jv = _judge_version(judge_spec)

    loaded = models_mod.load_model(spec)
    n_layers = loaded.model.cfg.n_layers

    examples = [Example(id=str(i), prompt=p, metadata={}) for i, p in enumerate(PROMPTS)]
    prompts = [e.prompt for e in examples]

    # Build a steering vector from the prompts themselves (mean_diff needs a
    # labeled contrast; here we only need SOME per-layer (n_layers, d_model) stack
    # to demonstrate the two apply-time operations, so we capture residuals and use
    # their per-layer mean-difference between the first and second half as a stand-
    # in direction). A real run would reuse a run's steering_vector.safetensors.
    _, caches = models_mod.generate_with_cache(
        loaded, prompts, max_tokens, None,
        capture_names=[f"blocks.{i}.hook_resid_pre" for i in range(n_layers)],
    )
    resids = [steering.capture_mean(c, n_layers) for c in caches]  # each (n_layers,d_model)
    half = max(1, len(resids) // 2)
    pos = torch.stack(resids[:half]).mean(0)
    neg = torch.stack(resids[half:]).mean(0)
    vector = pos - neg  # (n_layers, d_model)
    steering.assert_steering_shape(vector, n_layers, loaded.model.cfg.d_model)

    # Three conditions on the SAME prompts.
    initial = models_mod.generate(loaded, prompts, max_tokens, None)
    adapt_hooks = steering.apply_adaptive_ablation_perlayer(loaded.model, vector)
    adaptive = models_mod.generate_with_hooks(loaded, prompts, adapt_hooks, max_tokens, None)
    fixed_hooks = steering.apply_resid_pre_add(loaded.model, vector, coeff=-coeff)  # add −c·direction
    fixed = models_mod.generate_with_hooks(loaded, prompts, fixed_hooks, max_tokens, None)

    j_init = neutrality_judge(initial, examples, judge_spec)
    j_adapt = neutrality_judge(adaptive, examples, judge_spec)
    j_fixed = neutrality_judge(fixed, examples, judge_spec)

    labels = list(judge_spec.labels) + ["nonsense"]

    def transition_matrix(after):
        m = {a: {b: 0 for b in labels} for a in labels}
        for a, b in zip(j_init, after):
            m.setdefault(a, {b2: 0 for b2 in labels})
            m[a][b] = m[a].get(b, 0) + 1
        return m

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_key,
        "coeff": coeff,
        "n_prompts": len(prompts),
        **jv,
        "adaptive_ablation": {
            "labels_after": j_adapt,
            "transition_matrix": transition_matrix(j_adapt),
        },
        "fixed_add_neg_coeff": {
            "labels_after": j_fixed,
            "transition_matrix": transition_matrix(j_fixed),
        },
        "labels_initial": j_init,
    }
    (out_dir / "judged_result.json").write_text(json.dumps(result, indent=2))

    # One comparable table: per method, per prompt, init -> steered label.
    with (out_dir / "judged_transitions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "prompt", "initial", "adaptive_ablation", "fixed_add_neg_coeff"])
        for i, p in enumerate(prompts):
            w.writerow([i, p, j_init[i], j_adapt[i], j_fixed[i]])
    return result


def main():
    # Plain-ASCII description: the module docstring carries unicode (r-hat, arrows)
    # that a Windows cp1252 console cannot print, which would crash --help there.
    ap = argparse.ArgumentParser(
        description="Compare adaptive ablation vs fixed-coeff additive steering. "
                    "Default: mechanism (synthetic, no model/judge). --run-model: "
                    "judged run (needs GPU + OPENAI_API_KEY). See summary.md.")
    ap.add_argument("--run-model", action="store_true",
                    help="run the real judged comparison (needs GPU + OPENAI_API_KEY)")
    ap.add_argument("--model", default="qwen-1.8b", help="model registry key")
    ap.add_argument("--coeff", type=float, default=8.0,
                    help="fixed additive coefficient c to compare against")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.run_model:
        res = run_judged(out_dir, model_key=args.model, coeff=args.coeff,
                         max_tokens=args.max_tokens)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))
    else:
        summary = run_mechanism(out_dir, coeff=args.coeff)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
