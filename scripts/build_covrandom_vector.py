"""Exp-2 — build a covariance-matched random direction, the "it's the dose" control.

`docs/HANDOFF_lane_a.md` §Exp-2. Kills the reviewer objection "you didn't move a
representation, you just perturbed the residual stream hard enough". At the reported
dose, a random vector matched to the opinion direction on **norm AND residual
covariance** must move the opinion rate ~0, and the opinion direction must exceed it
by >=4 SE.

Norm-matching alone is too weak a control. The residual stream is wildly
anisotropic, so a norm-matched *isotropic* direction points mostly into
low-variance subspaces the model barely uses: it is easy to beat, and beating it
shows only that the opinion direction lies somewhere the model is sensitive to.
Matching the covariance puts the control in the same anisotropic envelope as the
real direction, so what remains between them is direction, not magnitude.

    python scripts/build_covrandom_vector.py \
        --extract-run runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b \
        --out runs/laneA_exp2_covrandom_vector --seed 0

## How the covariance shaping is done

The handoff sketches `cholesky(C + eps*I)` on the (d, d) covariance. This uses the
algebraically equivalent but better-behaved form: with `X` the centred residuals
`(N, d)` and `w ~ N(0, I_N)`,

    r = X^T w      =>      Cov(r) = X^T X = (N - 1) * C_hat

so `r` is an exact draw from a Gaussian with the empirical covariance (up to the
scale that norm-matching removes anyway). It needs no (d, d) matrix and no ridge
fudge — which matters here because N=400 and d=4096, so `C_hat` has rank <= 399 and
is singular. A Cholesky of `C + 1e-4*I` on a singular matrix does not regularise the
draw so much as add an isotropic component in the 3697 directions the data never
visited, which is the very thing covariance-matching is supposed to avoid.

## Per-layer, not one direction tiled

The opinion vector is `(n_layers, d_model)` — a *different* direction at each layer,
applied as `(coeff / n_layers) * vector[layer]`. So the control is built per layer:
layer L's random direction is drawn from layer L's residual covariance and scaled to
layer L's opinion-vector norm. Tiling a single direction across all layers (the
handoff's snippet) would differ from the thing it controls in two ways at once —
direction AND per-layer norm profile — and could not isolate either.

## Which covariance, and the finding that a covariance match is NOT enough

`--covariance within-pole` (the default) centres each pole on its own mean, so the
between-pole difference — which IS the opinion vector — is not part of what is
sampled. `pooled` centres both on a global mean. Measured on this extraction, the two
give the same answer, and it is not the expected one:

    pooled       mean |cos| with the opinion direction 0.4502   max 0.9669
    within-pole  mean |cos| 0.4522   max 0.9686
    (a random draw in a rank-398 subspace would give ~0.05)

So a covariance-matched draw is roughly 45% aligned with the direction it is supposed
to control for, and at some layers 97%. The reason is visible in the spectrum: at the
middle layers the within-pole residual covariance is very nearly rank ONE, and the
opinion vector lies along that one direction.

    layer  |cos(v_opinion, PC1)|   PC1 share of variance
       12          0.885                   0.952
       18          0.748                   0.916
       24          0.522                   0.684
       35          0.648                   0.129

Where a representation is low-rank, "matched on the residual covariance" and
"a different direction" are in direct tension: nearly every vector the covariance can
produce is the opinion direction again. A covariance-matched control is therefore NOT
automatically a random-direction control, and the alignment has to be measured rather
than assumed — measuring it is why this script reports `cos_with_opinion` per layer
instead of only norms.

`--orthogonalise` (recommended, and what Exp-2 reports) projects the opinion
direction out of the draw per layer and then rematches the norm, which is the control
that actually answers "it's the dose, not the direction": same per-layer norm, same
anisotropic envelope, zero component along the direction under test. Both vectors are
cheap to build; which one a number came from is recorded in
`covrandom_provenance.json`, and the non-orthogonalised draw's alignment is the
evidence for why the orthogonalised one is the headline control.
"""

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def build(extract_run: Path, seed: int, covariance: str = "within-pole",
          orthogonalise: bool = False) -> tuple:
    """Return (control_vector, opinion_vector, diagnostics)."""
    import torch
    from safetensors.torch import load_file

    resid_path = extract_run / "residuals.safetensors"
    vec_path = extract_run / "steering_vector.safetensors"
    if not resid_path.is_file():
        raise SystemExit(
            f"{resid_path} is missing. residuals.safetensors is git-ignored (bulky), so "
            f"it only exists on the box that ran the extraction — the covariance cannot "
            f"be estimated without it."
        )

    v_opin = load_file(str(vec_path))["vector"].float()          # (n_layers, d_model)
    resids = load_file(str(resid_path))                          # label -> (N, L, d)
    labels = sorted(resids)
    stacks = [resids[k].float() for k in labels]
    if covariance == "within-pole":
        # Centre each pole on ITS OWN mean, so the between-pole difference (which is
        # the opinion vector) is not part of the covariance being sampled.
        stacks = [s_ - s_.mean(dim=0, keepdim=True) for s_ in stacks]
    elif covariance != "pooled":
        raise SystemExit(f"--covariance must be within-pole|pooled, got {covariance!r}")
    X = torch.cat(stacks, dim=0)                                 # (N_total, L, d)
    n_total, n_layers, d_model = X.shape
    if tuple(v_opin.shape) != (n_layers, d_model):
        raise SystemExit(
            f"vector {tuple(v_opin.shape)} and residuals {(n_layers, d_model)} disagree"
        )

    gen = torch.Generator().manual_seed(seed)
    control = torch.zeros_like(v_opin)
    per_layer = []
    for layer in range(n_layers):
        Xl = X[:, layer, :]                                      # (N, d)
        # Already per-pole centred for within-pole; this removes the residual global
        # mean, which for `pooled` is the only centring and for `within-pole` is ~0.
        Xl = Xl - Xl.mean(dim=0, keepdim=True)
        w = torch.randn(n_total, generator=gen, dtype=Xl.dtype)
        r = Xl.transpose(0, 1) @ w                               # (d,) ~ N(0, X^T X)
        target = v_opin[layer].norm()
        if orthogonalise:
            # Remove the component along the direction under test, so what is left is
            # "same envelope, different direction" and nothing else. Done BEFORE the
            # norm rematch, or the projection would shrink the control's norm.
            u = v_opin[layer] / v_opin[layer].norm()
            r = r - (r @ u) * u
        rn = r.norm()
        if rn == 0:
            raise SystemExit(f"layer {layer}: degenerate draw (zero norm)")
        r = r / rn * target
        control[layer] = r
        cos = torch.nn.functional.cosine_similarity(
            r.unsqueeze(0), v_opin[layer].unsqueeze(0)).item()
        per_layer.append({
            "layer": layer,
            "opinion_norm": round(target.item(), 4),
            "control_norm": round(r.norm().item(), 4),
            "cos_with_opinion": round(cos, 4),
        })

    # Degrees of freedom: within-pole centring costs one per pole, pooled one overall.
    rank_bound = min(n_total - len(labels) if covariance == "within-pole" else n_total - 1,
                     d_model)
    diagnostics = {
        "extract_run": extract_run.name,
        "covariance": covariance,
        "orthogonalised_to_opinion": bool(orthogonalise),
        "residual_labels_pooled": labels,
        "n_residual_samples": int(n_total),
        "n_layers": int(n_layers), "d_model": int(d_model),
        "seed": seed,
        "method": "r = X^T w, w ~ N(0, I_N), per layer, rescaled to the opinion "
                  "vector's per-layer norm",
        "covariance_rank_bound": int(rank_bound),
        "max_abs_cos_with_opinion": round(
            max(abs(p["cos_with_opinion"]) for p in per_layer), 4),
        "mean_abs_cos_with_opinion": round(
            sum(abs(p["cos_with_opinion"]) for p in per_layer) / n_layers, 4),
        "norm_match_max_abs_error": round(
            max(abs(p["control_norm"] - p["opinion_norm"]) for p in per_layer), 6),
        "per_layer": per_layer,
    }
    return control, v_opin, diagnostics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract-run", required=True, type=Path,
                    help="run dir holding residuals.safetensors + steering_vector.safetensors")
    ap.add_argument("--out", required=True, type=Path,
                    help="directory to write steering_vector.safetensors + provenance into")
    ap.add_argument("--covariance", default="within-pole",
                    choices=("within-pole", "pooled"),
                    help="which residual covariance to sample (see the module "
                         "docstring: `pooled` is dominated by the contrast itself and "
                         "does NOT give a random-direction control)")
    ap.add_argument("--orthogonalise", action="store_true",
                    help="project the opinion direction out of the draw (per layer) "
                         "before rematching its norm — the control Exp-2 reports; see "
                         "the module docstring for why a covariance match alone is not "
                         "a random-direction control here")
    ap.add_argument("--seed", type=int, default=0,
                    help="one draw per seed; a single draw is a single sample of the "
                         "control distribution, so report which seed a number came from")
    a = ap.parse_args(argv)

    import torch  # noqa: F401  (imported here so --help works without the ML stack)
    from safetensors.torch import save_file

    from src.bias_steer import steering

    extract = a.extract_run if a.extract_run.is_absolute() else _REPO / a.extract_run
    out = a.out if a.out.is_absolute() else _REPO / a.out
    control, v_opin, diag = build(extract, a.seed, covariance=a.covariance,
                                  orthogonalise=a.orthogonalise)

    # The guard that the 2025 bug bypassed: assert the control is (n_layers, d_model)
    # before anything can apply it. A control that silently became a scalar would
    # "prove" the specificity claim by doing nothing at all.
    steering.assert_steering_shape(control, diag["n_layers"], diag["d_model"])

    out.mkdir(parents=True, exist_ok=True)
    # Saved in the same dtype as the vector it controls, so the two arms differ in
    # direction only and not in numerical precision.
    save_file({"vector": control.to(v_opin.dtype).contiguous()},
              str(out / "steering_vector.safetensors"))
    (out / "covrandom_provenance.json").write_text(json.dumps(diag, indent=2) + "\n")

    print(f"wrote {out / 'steering_vector.safetensors'}  "
          f"shape ({diag['n_layers']}, {diag['d_model']})  seed={a.seed}  "
          f"covariance={a.covariance}"
          + ("  orthogonalised" if a.orthogonalise else ""))
    print(f"  norm match: max abs error {diag['norm_match_max_abs_error']:.2e}")
    print(f"  |cos| with the opinion direction: mean "
          f"{diag['mean_abs_cos_with_opinion']:.4f}, max "
          f"{diag['max_abs_cos_with_opinion']:.4f}  "
          f"(expected ~{(1 / diag['covariance_rank_bound']) ** 0.5:.4f} for a random "
          f"draw in a rank-{diag['covariance_rank_bound']} subspace)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
