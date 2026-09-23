"""Run-2 analysis: floors with intervals, the negative control, the specificity
control, and the cross-category matrix.

Written fresh rather than by editing `bias_taxonomy.py`, deliberately.  The
shipped functions are what produced every run-1 artifact, and changing them in
place would make run 1 unreproducible.  Three of them also carry defects the
pre-registration closes, so run 2 needs different behaviour, not patched
behaviour:

  `extraction_floor`  returns a q05 over 10 draws and no interval          (S1)
  `split_half`        shuffles items, not scenario pairs                   (N5)
  `summarize_cosine`  takes an unweighted median across layers             (N1)

Each replacement below names the defect it closes and the rule that fixes it.
"""

from __future__ import annotations

import numpy as np

from src.bias_steer.bias_taxonomy import assert_direction, per_layer_cosine
from . import pairing


# --------------------------------------------------------------------------- #
# Estimator — the primary contrast, notes/13 §2.1
# --------------------------------------------------------------------------- #

def mean_diff_direction(resid_a: np.ndarray, resid_b: np.ndarray) -> np.ndarray:
    """mean(arm A) - mean(arm B) -> (n_layers, d_model).

    No free hyperparameter, which was the original reason for choosing a
    difference of means and which carries over intact to the annotation-labelled
    contrast (notes/13 §2.1).
    """
    if resid_a.ndim != 3 or resid_b.ndim != 3:
        raise ValueError(f"expected (n_items, n_layers, d_model); "
                         f"got {resid_a.shape} and {resid_b.shape}")
    if resid_a.shape[1:] != resid_b.shape[1:]:
        raise ValueError(f"layer/d_model mismatch: {resid_a.shape} vs {resid_b.shape}")
    return assert_direction(resid_a.mean(axis=0) - resid_b.mean(axis=0))


# --------------------------------------------------------------------------- #
# Layer summary — closes N1
# --------------------------------------------------------------------------- #

def norm_weighted_mean_cosine(cos_per_layer, reference) -> float:
    """Collapse a per-layer cosine profile, weighting by the reference's L2 norm.

    `notes/13` §8 fixes this rule in advance.  The justification is measured, not
    aesthetic: on qwen-14b, 21 of 40 layers carry under 10% of the peak norm
    (notes/12 N1), so an unweighted median treats a near-silent layer as equal to
    the highest-signal one — and it averages a DIFFERENT population for each
    estimator, which compounds S4.

    The rule also cancels to first order between the observed and control arms,
    because it is applied identically to both.  The unweighted median is reported
    alongside as a pre-declared sensitivity; run 1's measured difference was
    <= 0.033.
    """
    c = np.asarray(cos_per_layer, dtype=np.float64)
    w = np.linalg.norm(assert_direction(reference), axis=1)
    ok = np.isfinite(c) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(c[ok], weights=w[ok]))


def unweighted_median_cosine(cos_per_layer) -> float:
    """Run 1's rule, retained as the pre-declared sensitivity (notes/13 §8)."""
    c = np.asarray(cos_per_layer, dtype=np.float64)
    f = c[np.isfinite(c)]
    return float(np.median(f)) if f.size else float("nan")


def summarize(direction_a, direction_b) -> dict:
    """Both layer-summary rules at once, so the sensitivity is never optional."""
    cos = per_layer_cosine(direction_a, direction_b)
    return {
        "norm_weighted_mean": norm_weighted_mean_cosine(cos, direction_a),
        "unweighted_median": unweighted_median_cosine(cos),
    }


# --------------------------------------------------------------------------- #
# The floor, with an interval — closes S1, and splits by pair, closing N5
# --------------------------------------------------------------------------- #

def bootstrap_ci(values, *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on the MEAN of `values`.

    The statistic is the mean rather than run 1's q05 because a quantile over B
    draws has materially larger error than the mean, and run 1 combined the worst
    of both (notes/13 §4).
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def extract_from_pairs(pairs, capture, *, transform=None) -> np.ndarray:
    """Direction from a list of Pair, via the injected `capture` callable.

    `capture(rows, arm_sign) -> (n, n_layers, d_model)` is injected so this stays
    torch-free and unit-testable, the same discipline `extraction_floor` already
    uses in the shipped module.

    `transform` post-processes the direction — used to project out the length
    component so a floor can be computed on the residualised direction.
    """
    a_rows, b_rows = pairing.arms(pairs)
    d = mean_diff_direction(capture(a_rows, +1.0), capture(b_rows, -1.0))
    return transform(d) if transform is not None else d


def floor(pairs, capture, *, n_splits: int = 400, seed: int = 0,
          n_boot: int = 2000, transform=None) -> dict:
    """Split-half floor: mean, bootstrap 95% CI, and the raw cosines.

    `n_splits = 400` is a calculation, not a default (notes/13 §4): at run 1's
    90th-percentile split SD of 0.2023, 393 splits give a 95% CI half-width of
    +/-0.020 on the mean.  Run 1 used 10.

    Splits are BY SCENARIO PAIR (notes/19 §6.3), so each half is an independent
    sample of scenarios and both halves are automatically arm-balanced.
    """
    if len(pairs) < 4:
        raise ValueError(f"need >= 4 pairs to split-half; got {len(pairs)}")

    nwm, umed = [], []
    for k in range(n_splits):
        A, B = pairing.split_pairs(pairs, seed=seed + k)
        if not A or not B:
            continue
        dA = extract_from_pairs(A, capture, transform=transform)
        dB = extract_from_pairs(B, capture, transform=transform)
        s = summarize(dA, dB)
        nwm.append(s["norm_weighted_mean"])
        umed.append(s["unweighted_median"])

    lo, hi = bootstrap_ci(nwm, n_boot=n_boot, seed=seed)
    finite = [x for x in nwm if np.isfinite(x)]
    return {
        "n_pairs": len(pairs), "n_splits": len(nwm),
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "sensitivity_unweighted_median_mean":
            float(np.mean([x for x in umed if np.isfinite(x)])) if umed else float("nan"),
        "cosines": [float(x) for x in nwm],
    }


def negative_control_floor(pairs, capture, *, n_splits: int = 400, seed: int = 0,
                           n_boot: int = 2000, transform=None) -> dict:
    """The same floor with arm labels shuffled WITHIN each scenario.

    `notes/13` §3.1 step 2.  Swapping the two members of a pair is the tightest
    possible label shuffle: the two items are the same scenario, so topic,
    vocabulary, prompt format and n are held not merely matched but identical.
    This is what closes N4 — the null is now a matched alternative rather than
    noise.
    """
    shuffled = pairing.shuffle_arm_labels(pairs, seed=seed + 10_000)
    return floor(shuffled, capture, n_splits=n_splits, seed=seed + 20_000,
                 n_boot=n_boot, transform=transform)


def reproduces(observed: dict, control: dict) -> str:
    """The usability criterion.  No threshold constant exists (notes/13 §3.1).

    Returns "YES", "NO", or "INDETERMINATE" — the last being the pre-declared
    straddle case of §3.2, which is excluded from the primary clustering and
    reported by name with both intervals.
    """
    if not np.isfinite(observed["ci_lo"]) or not np.isfinite(control["ci_hi"]):
        return "INDETERMINATE"
    if observed["ci_lo"] > control["ci_hi"]:
        return "YES"
    if observed["ci_hi"] < control["ci_lo"]:
        return "NO"
    return "INDETERMINATE"


# --------------------------------------------------------------------------- #
# The specificity control — notes/19 §3.3, hole (a)
# --------------------------------------------------------------------------- #

def length_direction(rows, capture) -> np.ndarray:
    """mean(longest third) - mean(shortest third), both inside ONE arm.

    `context_condition` is held exactly fixed, so the only systematic difference
    between the two groups is context length.  This measures the confound rather
    than inferring it — which is the whole reason the candidate rule (median
    cross-category cosine vs lowest within-category floor) was rejected: no
    statistic computed only from cross-category cosines can tell "we measured
    length" apart from "bias is one mechanism".
    """
    long_rows, short_rows = pairing.length_terciles(rows)
    return mean_diff_direction(capture(long_rows, +1.0), capture(short_rows, -1.0))


def pooled_length_direction(per_category_rows: dict, capture) -> np.ndarray:
    """`d_len_bar` — the best estimate of "what reading a longer context does"."""
    dirs = [length_direction(rows, capture) for rows in per_category_rows.values()]
    return assert_direction(np.mean(np.stack(dirs, axis=0), axis=0))


def specificity_control(pairs_by_category: dict, capture_by_category: dict,
                        directions: dict, floors: dict, negatives: dict,
                        d_len_bar, len_selfcheck: dict, *,
                        n_splits: int = 400, seed: int = 0) -> dict:
    """Is a category's direction bias, or is it sentence length?

    RULE (notes/19 §3.3 A-2, corrected twice against the pilot), threshold-free:

        category C FAILS iff

            |cos(d_C, d_len_bar)|  >=  sqrt( CI_lo(floor_C) * CI_lo(floor_len) )

    The right-hand side is the LARGEST cosine two noisy estimates of the SAME
    underlying direction could be expected to show, given how well each
    reproduces against itself.  If the observed alignment reaches that ceiling,
    the two are indistinguishable at the available precision — the direction is
    length.  No constant appears: both terms are floors this pipeline already
    computes, and the geometric mean is the standard correction for attenuation
    by measurement error.

    TWO EARLIER VERSIONS FAILED, AND THE PILOT IS WHAT SHOWED IT:

    1.  `|cos| >= CI_lo(floor_C)` can never fire.  A large length confound is a
        big, stable, shared component, so it inflates the floor by as much as it
        inflates the alignment.  On planted pure-length data it read 0.92 against
        a floor of 0.966 and returned PASS — an entirely artifactual direction,
        certified clean.  Comparing two quantities driven by the same cause is
        not a control.

    2.  "does C still beat its negative control once `d_len_bar` is projected
        out" is better, but ATTENUATED: `d_len_bar` is itself only measured to a
        self-floor of ~0.75, so projecting it out removes only the part of the
        confound that was estimated.  On the same pure-length data the projected
        floor was still 0.805 against a control of 0.378 — reported as
        reproducing.  You cannot fully project out a confound you can only
        estimate noisily.

    The disattenuated comparison is what remains.  The projected floor is still
    computed and reported, because it is a useful descriptive number and it is
    how the attenuation was found; it is no longer the decision.
    """
    def _proj(d):
        return project_out(d, d_len_bar)

    len_floor_lo = max(0.0, float(len_selfcheck.get("ci_lo") or 0.0))
    per_cat, fails = {}, []
    for name, d in directions.items():
        cos_len = summarize(d, d_len_bar)["norm_weighted_mean"]
        cat_floor_lo = max(0.0, floors[name]["ci_lo"])
        ceiling = float(np.sqrt(cat_floor_lo * len_floor_lo))

        pairs, cap = pairs_by_category[name], capture_by_category[name]
        proj_floor = floor(pairs, cap, n_splits=n_splits, seed=seed, transform=_proj)
        proj_neg = negative_control_floor(pairs, cap, n_splits=n_splits, seed=seed,
                                          transform=_proj)

        failed = bool(ceiling > 0 and abs(cos_len) >= ceiling)
        per_cat[name] = {
            "abs_cos_with_length_direction": abs(cos_len),
            "length_variance_share": float(cos_len ** 2),
            "own_floor_ci_lo": floors[name]["ci_lo"],
            "length_floor_ci_lo": len_floor_lo,
            "indistinguishability_ceiling": ceiling,
            "ratio_to_ceiling": float(abs(cos_len) / ceiling) if ceiling > 0 else float("nan"),
            "reproduces_before_projection": reproduces(floors[name], negatives[name]),
            "reproduces_after_projection": reproduces(proj_floor, proj_neg),
            "floor_after_projection": proj_floor["mean"],
            "negative_control_after_projection": proj_neg["mean"],
            "verdict": "LENGTH" if failed else "BIAS-SPECIFIC",
        }
        fails.append(failed)

    return {
        "rule": "fails iff |cos(d_C, d_len_bar)| >= sqrt(CI_lo(floor_C) * CI_lo(floor_len))",
        "per_category": per_cat,
        "n_failing": int(sum(fails)),
        "n_categories": len(fails),
        "overall": "FAIL" if sum(fails) * 2 > len(fails) else "PASS",
    }


def length_direction_selfcheck(per_category_rows: dict, capture, *,
                               n_splits: int = 50, seed: int = 0) -> dict:
    """A-4: `d_len_bar` must reproduce against its own split-half floor.

    If the length direction is itself noise, the specificity control compares
    against noise and is vacuous.  Declared in advance so it is not discovered
    afterwards.

    The split is WITHIN each category and then pooled, exactly mirroring
    `pooled_length_direction`.  Pooling rows across categories BEFORE ranking
    them by length does not work, and the pilot demonstrated why: ambiguous
    Disability_status contexts average 106 characters and Physical_appearance
    134, so a pooled long tercile came out 9:4 Physical and the short tercile
    10:3 Disability.  The resulting "length" direction was substantially
    `d_Physical - d_Disability` — a CATEGORY direction wearing a length label.

    That is defect N4's disease in a new place: a contrast built by ranking a
    pooled, topic-heterogeneous set encodes topic, not the ranked variable.
    Ranking inside a category holds topic fixed, so length is what varies.
    """
    cos = []
    for k in range(n_splits):
        halves_a, halves_b = {}, {}
        for cat, rows in per_category_rows.items():
            idx = np.random.default_rng(seed + k).permutation(len(rows))
            halves_a[cat] = [rows[i] for i in idx[:len(idx) // 2]]
            halves_b[cat] = [rows[i] for i in idx[len(idx) // 2:]]
        if min(len(v) for v in halves_a.values()) < 6:
            continue
        if min(len(v) for v in halves_b.values()) < 6:
            continue
        dA = pooled_length_direction(halves_a, capture)
        dB = pooled_length_direction(halves_b, capture)
        # Both halves use the same (long minus short) convention, so this cosine
        # is positive when the length direction is real. No abs() here: a
        # negative self-cosine means the direction is noise, and that must show.
        cos.append(summarize(dA, dB)["norm_weighted_mean"])
    lo, hi = bootstrap_ci(cos, seed=seed)
    finite = [x for x in cos if np.isfinite(x)]
    return {"n_splits": len(cos),
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
            "usable": bool(np.isfinite(lo) and lo > 0.5)}


def project_out(direction, reference):
    """A-5, the pre-declared remedy: remove the length component, per layer.

    Reported ALONGSIDE the unprojected result, never instead of it.  If the bias
    signal is itself largely shared across categories, this removes signal along
    with confound, so a projected result is a LOWER BOUND and can never be the
    headline.
    """
    D, R = assert_direction(direction), assert_direction(reference)
    rn = np.linalg.norm(R, axis=1, keepdims=True)
    Rh = R / np.where(rn > 0, rn, 1.0)
    return assert_direction(D - (D * Rh).sum(axis=1, keepdims=True) * Rh)


# --------------------------------------------------------------------------- #
# Cross-category matrix
# --------------------------------------------------------------------------- #

def cross_category(directions: dict) -> dict:
    """Pairwise cosines between category directions, with their summary.

    The reference paper's logic (notes/17 §2.5): within-category >= 0.95
    establishes the noise floor, cross-category 0.4-0.6 sits far below it,
    therefore the directions are genuinely distinct.  This produces the second
    half of that comparison.
    """
    names = sorted(directions)
    n = len(names)
    M = np.full((n, n), np.nan)
    off = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = summarize(directions[a], directions[b])["norm_weighted_mean"]
            M[i, j] = v
            if i < j:
                off.append(v)
    finite = [x for x in off if np.isfinite(x)]
    return {
        "names": names,
        "matrix": M.tolist(),
        "offdiagonal": [float(x) for x in off],
        "median_offdiagonal": float(np.median(finite)) if finite else float("nan"),
        "max_offdiagonal": float(np.max(finite)) if finite else float("nan"),
    }
