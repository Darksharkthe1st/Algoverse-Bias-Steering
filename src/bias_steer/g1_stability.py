"""G1 direction-stability statistic (contract §12 A6, `docs/PREREG.md` §7a).

Arditi's published `direction.pt` exists for five models. `Qwen/Qwen3-8B` is not
one of them, so the frozen "extraction cosine ≥ 0.95 against the reference" test
is not merely hard on the submission model — it is undefined. Switching models to
keep an evaluable gate would let an artifact's availability choose the science.

The replacement asks the question the reference cosine was a proxy for — *is
this direction reproducibly estimable from the data, or is it noise?* — using
only the model's own activations:

    S_split = cos( r̂(half A), r̂(half B) )

for two disjoint halves of the harmful/harmless contrast pool, each extracted by
the identical frozen recipe. Two independent estimates of the same population
direction agree; an artifact of estimation noise does not.

**The null is calibrated, not assumed.** Residual streams are strongly
anisotropic, so "chance" cosine is nowhere near 0 and a Gaussian random-direction
null would badly understate it. The null here is a *label permutation*: shuffle
harmful/harmless across the pooled prompts and recompute S_split by the same
code path. That holds the prompt set, the layer, the token position, the sample
sizes and the geometry fixed, and removes only the thing under test — the label
signal.

**The null is nearly free.** A direction is a difference of means over already-
cached residuals, so B permutations cost B re-averagings of a cached tensor, not
B forward passes. Budget the run for one extraction pass; the null rides along.

Nothing here imports torch at module scope, and nothing here needs a model — the
inputs are cached activations, so this is unit-testable on synthetic tensors.
"""

from __future__ import annotations

import math

#: Pre-registered minimum split-half cosine. NOT copied from Arditi's 0.999/0.95
#: — derived, see `disattenuated_alignment`: S_split = 0.68 is the point at which
#: the full-pool direction's estimated alignment with the population direction
#: reaches 0.90.
S_SPLIT_FLOOR = 0.68

#: One-sided permutation test level for "better than chance geometry".
NULL_ALPHA = 0.01

#: Permutations. Cheap (re-averaging cached tensors), so this is set by the
#: resolution wanted at the 99th percentile, not by compute.
N_PERMUTATIONS = 500


def _unit(v):
    return v / v.norm()


def _direction(harmful, harmless):
    """Frozen recipe: difference in means, normalised. `(n, d)` -> `(d,)`.

    fp32 regardless of the model's dtype (contract §4) — a mean over hundreds of
    fp16 vectors loses precision exactly where the difference is small, which is
    where the stability question lives.
    """
    return _unit(harmful.float().mean(0) - harmless.float().mean(0))


def split_half_cosine(harmful, harmless, *, seed: int = 0) -> float:
    """cos between directions extracted from two disjoint halves of the pool.

    Halves are split within each label, so both halves keep the 1:1 harmful:
    harmless balance the recipe assumes.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    a_h, b_h = _halves(harmful, g)
    a_l, b_l = _halves(harmless, g)
    r_a = _direction(harmful[a_h], harmless[a_l])
    r_b = _direction(harmful[b_h], harmless[b_l])
    return float(r_a @ r_b)


def _halves(x, g):
    import torch

    perm = torch.randperm(x.shape[0], generator=g)
    cut = x.shape[0] // 2
    return perm[:cut], perm[cut:2 * cut]


def permutation_null(harmful, harmless, *, n: int = N_PERMUTATIONS,
                     seed: int = 0) -> list[float]:
    """Empirical null for `split_half_cosine` under shuffled labels.

    Pools the two label groups and re-draws the harmful/harmless assignment,
    preserving group sizes, then runs the identical split-half computation. What
    survives is every source of agreement *except* the label signal: prompt
    distribution, layer, token position, and the residual stream's anisotropy.
    """
    import torch

    pooled = torch.cat([harmful, harmless], dim=0).float()
    n_h = harmful.shape[0]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        idx = torch.randperm(pooled.shape[0], generator=g)
        out.append(split_half_cosine(pooled[idx[:n_h]], pooled[idx[n_h:]],
                                     seed=int(torch.randint(1 << 30, (1,), generator=g))))
    return out


def disattenuated_alignment(s_split: float) -> float:
    """Estimated cos between the FULL-pool direction and the population direction.

    Model each half-direction as signal plus independent isotropic noise. Then
    E[cos(A, B)] is the signal fraction rho, and cos(half, truth) ~ sqrt(rho).
    The full pool has twice the data, so its noise variance halves and its signal
    fraction becomes 2*rho/(1 + rho); hence

        alignment_full ~ sqrt( 2 * S_split / (1 + S_split) )

    At S_split = 0.68 this is 0.90, which is where `S_SPLIT_FLOOR` comes from.

    The isotropy assumption is an approximation and it is optimistic: real
    residual noise is correlated with the signal direction, so treat this as an
    upper bound on direction quality, not an estimate to quote in the paper.
    It sets a floor to *pass*, which is the conservative use of an upper bound.
    """
    s = max(0.0, min(1.0, s_split))
    return math.sqrt(2.0 * s / (1.0 + s)) if s > 0 else 0.0


def quantile(xs: list[float], q: float) -> float:
    """Linear-interpolated quantile. Local, so the null needs no numpy/scipy."""
    if not xs:
        raise ValueError("empty sample")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def assess(harmful, harmless, *, n_permutations: int = N_PERMUTATIONS,
           seed: int = 0) -> dict:
    """Run the G1a stability leg and return every number it decided on.

    Passing requires BOTH:

    * `s_split > null_q99` — the agreement is driven by the labels, not by the
      geometry the prompts share anyway. This is the identifiability claim.
    * `s_split >= S_SPLIT_FLOOR` — the direction is estimated well enough to be
      worth intervening with. This is the precision claim.

    Both, because either alone is gameable: a tight null can be cleared by a
    direction too noisy to act on, and a high cosine can be produced by shared
    prompt geometry with no label signal in it at all.
    """
    s = split_half_cosine(harmful, harmless, seed=seed)
    null = permutation_null(harmful, harmless, n=n_permutations, seed=seed + 1)
    q = quantile(null, 1.0 - NULL_ALPHA)
    # One-sided permutation p, with the observed value counted (Phipson & Smyth):
    # never reports p = 0, which would be a claim the null sample cannot support.
    p = (sum(1 for v in null if v >= s) + 1) / (len(null) + 1)
    return {
        "s_split": s,
        "null_q99": q,
        "null_median": quantile(null, 0.5),
        "p_permutation": p,
        "n_permutations": len(null),
        "alignment_full_est": disattenuated_alignment(s),
        "floor": S_SPLIT_FLOOR,
        "beats_null": s > q,
        "clears_floor": s >= S_SPLIT_FLOOR,
        "pass": bool(s > q and s >= S_SPLIT_FLOOR),
    }
