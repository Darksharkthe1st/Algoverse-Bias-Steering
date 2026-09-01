"""Run 3: the behavioural contrast, with the controls the design requires.

    V_C,raw = mean(resid | model produced a stereotyped answer)
            - mean(resid | model acknowledged the context is under-informative)

Written fresh rather than by editing `analysis.py`, for the same reason
`analysis.py` was written fresh rather than by editing `bias_taxonomy.py`: run 1
and run R1 must stay reproducible, and this contrast needs different behaviour,
not patched behaviour.  The two differ in a way that matters:

  R1's arms are labelled by `context_condition`, a dataset annotation, and come
  in matched PAIRS -- so the split unit is the pair (`pairing.split_pairs`).

  Run 3's arms are labelled by the model's own parsed output, so there are no
  pairs, bucket sizes are unequal and category-dependent, and the split must be
  stratified BY BUCKET instead.  Everything downstream of that -- the floor, the
  bootstrap, the decision rule -- is reused from `analysis.py` unchanged.

THE THREE THINGS THAT MAKE THIS CONTRAST DEFENSIBLE
---------------------------------------------------
1. `bucket_responses` marks a category UNTESTABLE rather than negative when a
   bucket is too small.  A category the model never stereotypes has no contrast
   to split on; that is defect M1, and calling it "no bias direction" is the
   specific error M1 names.

2. `mirror_pair_check` measures the parser rather than trusting it.  The bucket
   assignment IS the label, so parser error propagates straight into the vector
   -- and N6's failure mode is positional (ties resolve to the earliest-mentioned
   option), which correlates with the label being extracted.  Run 1 measured
   person-consistency at 48-68% against a 50% line and could not audit it,
   because no completion was ever saved.

3. `refusal_decoupling` treats "is this just the refusal direction?" as a
   measurement, not an assumption.  Bucketing by answered-vs-declined means every
   V_C contains the model's general abstention direction, which is shared across
   all categories -- so a high cross-category cosine is exactly what you would
   see if you had measured refusal and nothing else.
"""

from __future__ import annotations

import collections

import numpy as np

from src.bias_steer.bias_taxonomy import (assert_direction, parse_choice,
                                          per_layer_cosine, resolve_answer_roles)
from . import analysis

BIASED, REFUSAL, UNPARSED = "biased", "refusal", "unparsed"

#: `notes/13` §5 fixes the minimum at 32 per arm, following Arditi et al. and the
#: reference paper, which reach floors of 0.95-0.99 at exactly that n.  A lower
#: bar is a new undeclared constant, and at n=15 a split-half leaves 7 items per
#: half -- the direction is then mostly estimator noise, so the floor measures
#: the estimator rather than the representation.
MIN_BUCKET = 32


# --------------------------------------------------------------------------- #
# Phase 2.2 — bucketing, and the honest verdict when a bucket is too small
# --------------------------------------------------------------------------- #

def row_metadata(row: dict) -> dict:
    """Assemble the metadata dict `resolve_answer_roles` expects from a raw row.

    `resolve_answer_roles` does NOT take a BBQ jsonl row -- it takes the dict
    `datasets.load_bbq` builds (`src/bias_steer/datasets.py:99-121`), and the two
    differ in exactly the fields that decide the label: the row nests the
    stereotype under `additional_metadata`, encodes the per-answer group inside
    `answer_info["ansN"][-1]`, and carries no `target_loc` at all.

    Passing the raw row instead returns `usable == False` for every item, so the
    whole dataset buckets as `unparsed` and the direction is built from nothing.
    Nothing raises.  This adapter mirrors the loader field for field so the two
    paths cannot drift.
    """
    info = row.get("answer_info") or {}
    return {
        "category": row.get("category"),
        "label": row.get("label"),
        "question_polarity": row.get("question_polarity"),
        "context_condition": row.get("context_condition"),
        "answers": [row["ans0"], row["ans1"], row["ans2"]],
        "answer_groups": [(info.get(f"ans{n}") or [None, None])[-1] for n in range(3)],
        "stereotyped_groups": (row.get("additional_metadata") or {}).get(
            "stereotyped_groups") or [],
        # BBQ's own answer key, when it has been staged. `target_loc` ALREADY
        # accounts for question polarity -- resolve_answer_roles must not be
        # handed a pre-flipped value (bias_taxonomy.resolve_answer_roles docstring).
        "target_loc": row.get("target_loc"),
    }


def bucket_one(response: str, row: dict) -> str:
    """Assign one completion to `biased`, `refusal`, or `unparsed`.

    `resolve_answer_roles` supplies which option is the stereotyped one *after*
    adjusting for `question_polarity` -- on a `nonneg` question the stereotyped
    choice is the OTHER named option, so a bucketer that keys on
    `stereotyped_groups` alone mislabels half the dataset.
    """
    answers = [row["ans0"], row["ans1"], row["ans2"]]
    roles = resolve_answer_roles(row_metadata(row))
    if not roles.usable:
        return UNPARSED
    idx = parse_choice(response, answers)
    if idx is None:
        return UNPARSED
    if idx == roles.unknown:
        return REFUSAL
    return BIASED if idx == roles.biased else REFUSAL


def bucket_responses(rows: list, responses: list, *, min_bucket: int = MIN_BUCKET) -> dict:
    """Split one category's items into the two arms, and say if it is testable.

    Returns the two index lists plus the counts a reader needs to interpret a
    floor: an UNTESTABLE category is not evidence against a representation, and
    reporting it as a negative is the M1 error stated as a number.
    """
    if len(rows) != len(responses):
        raise ValueError(f"{len(rows)} rows but {len(responses)} responses")

    idx = collections.defaultdict(list)
    for i, (r, resp) in enumerate(zip(rows, responses)):
        idx[bucket_one(resp, r)].append(i)

    n_b, n_r = len(idx[BIASED]), len(idx[REFUSAL])
    testable = n_b >= min_bucket and n_r >= min_bucket
    return {
        "biased_idx": idx[BIASED], "refusal_idx": idx[REFUSAL],
        "unparsed_idx": idx[UNPARSED],
        "n_biased": n_b, "n_refusal": n_r, "n_unparsed": len(idx[UNPARSED]),
        "n_total": len(rows),
        "refusal_rate": n_r / len(rows) if rows else float("nan"),
        "unparsed_rate": len(idx[UNPARSED]) / len(rows) if rows else float("nan"),
        "min_bucket": min_bucket,
        "status": "TESTABLE" if testable else "UNTESTABLE",
        "untestable_reason": None if testable else (
            f"n_biased={n_b}, n_refusal={n_r}; need >= {min_bucket} in both. "
            f"This is 'no contrast to split on', NOT 'no bias direction' "
            f"(defect M1) -- report it as untestable, never as a negative."),
    }


# --------------------------------------------------------------------------- #
# Phase 2.2 — measuring the parser instead of trusting it
# --------------------------------------------------------------------------- #

def option_order_invariance(responses: list, rows: list) -> dict:
    """Cheap sanity check: does the parse change if the OPTION LIST is reordered?

    ⚠️ THIS CHECK IS VACUOUS ON ITS OWN AND MUST NOT BE REPORTED AS THE
    POSITION-BIAS CONTROL.  It scores ~1.0 by construction, because reordering
    the option list does not change which name appears first in the RESPONSE
    TEXT, and `parse_choice` resolves ties by earliest mention in the response.
    `run_pilot.confound_checklist` item 5.2 measured exactly this and records why
    `notes/18` item 4 is vacuous.

    Kept because a score materially below 1.0 would mean something is wrong with
    the parser in a new way.  It cannot detect N6.
    """
    agree = flipped = 0
    for resp, row in zip(responses, rows):
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        a = parse_choice(resp, answers)
        b = parse_choice(resp, list(reversed(answers)))
        if a is None or b is None:
            continue
        if answers[a] == list(reversed(answers))[b]:
            agree += 1
        else:
            flipped += 1
    n = agree + flipped
    rate = agree / n if n else float("nan")
    return {"n_compared": n, "n_agree": agree, "n_flipped": flipped,
            "order_invariance": rate,
            "is_the_position_bias_control": False,
            "note": "vacuous by construction — see person_swap_consistency"}


def person_swap_consistency(buckets_a: dict, buckets_b: dict, rows: list) -> dict:
    """THE position-bias control: swap the two named PEOPLE IN THE PROMPT and
    regenerate, then ask whether the same item lands in the same bucket.

    This is the test that bites (`notes/19` §5.3).  It needs a SECOND GENERATION
    PASS on entity-swapped prompts -- `run3_behavioural_contrast.py` produces it
    -- because the thing being measured is not the parser in isolation but the
    label as actually assigned: model behaviour and parser together.

    Why it is load-bearing here specifically.  In this design the bucket IS the
    label, so any position-dependent error propagates directly into `V_C`.  N6's
    parser resolves ties by EARLIEST MENTION, and BBQ frequently names the
    stereotyped target first -- so the error correlates with the very label being
    extracted.  A confound aligned with the signal cannot be averaged away by
    collecting more items.

    Run 1 measured this at 48-68% against a 50% chance line and could not audit
    it, because no completion was ever persisted.  Phase 2.1's `responses.jsonl`
    mandate is what makes it computable at all.

    `buckets_a` / `buckets_b` are `bucket_responses` outputs for the original and
    entity-swapped generations, over the SAME `rows` in the same order.
    """
    def label(bk):
        out = {}
        for i in bk["biased_idx"]:
            out[i] = BIASED
        for i in bk["refusal_idx"]:
            out[i] = REFUSAL
        return out

    la, lb = label(buckets_a), label(buckets_b)
    shared = sorted(set(la) & set(lb))
    agree = sum(1 for i in shared if la[i] == lb[i])
    n = len(shared)
    rate = agree / n if n else float("nan")

    # Which way do the disagreements fall? A parser with a first-mention bias
    # produces ASYMMETRIC flips, and the asymmetry is the diagnostic: symmetric
    # disagreement is noise, one-sided disagreement is the confound.
    to_biased = sum(1 for i in shared if la[i] == REFUSAL and lb[i] == BIASED)
    to_refusal = sum(1 for i in shared if la[i] == BIASED and lb[i] == REFUSAL)

    return {
        "n_compared": n, "n_agree": agree, "consistency": rate,
        "flips_refusal_to_biased": to_biased,
        "flips_biased_to_refusal": to_refusal,
        "flip_asymmetry": abs(to_biased - to_refusal) / n if n else float("nan"),
        # 0.5 is chance for a two-way label. Run 1 scored 48-68%.
        "chance_line": 0.5,
        "usable": bool(n > 0 and rate >= 0.90),
        "note": "if not usable, the bucket labels carry a position-dependent "
                "error INTO the direction. Replace the parser with a judge, or "
                "measure its error against hand labels, before interpreting any "
                "V_C (N6). Do not proceed on the strength of "
                "option_order_invariance, which cannot detect this.",
    }


# --------------------------------------------------------------------------- #
# Phase 2.3 — the direction, and its floor
# --------------------------------------------------------------------------- #

def behavioural_direction(resid: np.ndarray, buckets: dict) -> np.ndarray:
    """V_C,raw = mean(biased rows) - mean(refusal rows), shape (n_layers, d_model)."""
    b, r = buckets["biased_idx"], buckets["refusal_idx"]
    if not b or not r:
        raise ValueError(f"empty bucket: n_biased={len(b)}, n_refusal={len(r)}")
    return assert_direction(resid[b].mean(axis=0) - resid[r].mean(axis=0))


def split_buckets(buckets: dict, seed: int) -> tuple[dict, dict]:
    """Split half STRATIFIED BY BUCKET, so both halves keep both arms.

    N5, restated for unequal buckets.  Cutting a pooled list blind lets one half
    drift toward one arm, and a difference of means over unbalanced arms adds
    variance to the floor that has nothing to do with reproducibility.
    """
    rng = np.random.default_rng(seed)
    out = ({}, {})
    for key in ("biased_idx", "refusal_idx"):
        v = list(buckets[key])
        rng.shuffle(v)
        mid = len(v) // 2
        out[0][key], out[1][key] = v[:mid], v[mid:]
    return out


def bucket_floor(resid: np.ndarray, buckets: dict, *, n_splits: int = 400,
                 seed: int = 0, n_boot: int = 2000, transform=None) -> dict:
    """Split-half floor over buckets.  Same statistic and CI as `analysis.floor`.

    Without this number a cross-category cosine cannot be read at all: 0.6 means
    nothing until you know how far V_C moves when re-extracted from half of its
    own data and NOTHING changed.
    """
    nwm, umed = [], []
    for k in range(n_splits):
        A, B = split_buckets(buckets, seed=seed + k)
        if min(len(A["biased_idx"]), len(A["refusal_idx"]),
               len(B["biased_idx"]), len(B["refusal_idx"])) < 2:
            continue
        dA = behavioural_direction(resid, A)
        dB = behavioural_direction(resid, B)
        if transform is not None:
            dA, dB = transform(dA), transform(dB)
        s = analysis.summarize(dA, dB)
        nwm.append(s["norm_weighted_mean"])
        umed.append(s["unweighted_median"])

    lo, hi = analysis.bootstrap_ci(nwm, n_boot=n_boot, seed=seed)
    finite = [x for x in nwm if np.isfinite(x)]
    return {
        "n_biased": buckets["n_biased"], "n_refusal": buckets["n_refusal"],
        "n_splits": len(nwm),
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "sensitivity_unweighted_median_mean":
            float(np.mean([x for x in umed if np.isfinite(x)])) if umed else float("nan"),
        "cosines": [float(x) for x in nwm],
    }


def shuffled_bucket_control(resid: np.ndarray, buckets: dict, *, n_splits: int = 400,
                            seed: int = 0, n_shuffles: int = 20,
                            n_boot: int = 2000, transform=None) -> dict:
    """The negative control: reassign bucket labels at random, n sizes held fixed.

    Averaged over `n_shuffles` label draws rather than one.  A single draw is a
    nuisance parameter, not a null: its realised imbalance leaves a fraction of
    the true direction inside the "null", and the split-only CI does not span
    that.  Splitting one budget of `n_splits` across `n_shuffles` draws keeps the
    total work -- and therefore the runtime -- identical.
    """
    pool = list(buckets["biased_idx"]) + list(buckets["refusal_idx"])
    n_b = buckets["n_biased"]
    per = max(1, n_splits // n_shuffles)

    cos = []
    for j in range(n_shuffles):
        rng = np.random.default_rng(seed + 10_000 + j)
        p = list(pool)
        rng.shuffle(p)
        fake = {"biased_idx": p[:n_b], "refusal_idx": p[n_b:],
                "n_biased": n_b, "n_refusal": len(p) - n_b}
        f = bucket_floor(resid, fake, n_splits=per, seed=seed + 20_000 + 1000 * j,
                         n_boot=n_boot, transform=transform)
        cos.extend(f["cosines"])

    lo, hi = analysis.bootstrap_ci(cos, n_boot=n_boot, seed=seed)
    finite = [x for x in cos if np.isfinite(x)]
    return {"n_shuffles": n_shuffles, "n_splits": len(cos),
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "cosines": [float(x) for x in cos]}


# --------------------------------------------------------------------------- #
# Phase 2.4 — refusal de-coupling, measured rather than assumed
# --------------------------------------------------------------------------- #

def unit_per_layer(direction) -> np.ndarray:
    """Row-normalise to unit L2 per layer.  Zero-norm layers stay zero.

    Required before any cross-application: per-layer norms span 600-1391x on
    these models, so injecting `+alpha*V_race` and `+alpha*V_gender` at the same
    alpha delivers different MAGNITUDES, and the comparison measures norm rather
    than direction.
    """
    D = assert_direction(direction)
    n = np.linalg.norm(D, axis=1, keepdims=True)
    return D / np.where(n > 0, n, 1.0)


def refusal_decoupling(directions: dict, floors: dict, v_refusal,
                       refusal_floor: dict) -> dict:
    """Is V_C a bias direction, or is it the model's abstention direction?

    Bucketing by answered-vs-declined puts the general refusal direction into
    EVERY V_C.  It is shared across categories by construction, so it inflates
    every cross-category cosine -- and Phase 4 would read that as "a universal
    V_bias" when it is one mechanism wearing ten labels.

    DECISION RULE, threshold-free, identical in form to the specificity control
    in `analysis.specificity_control`:

        category C is REFUSAL-DOMINATED iff
            |cos(V_C, V_refusal)| >= sqrt( CI_lo(floor_C) * CI_lo(floor_refusal) )

    The right-hand side is the largest cosine two noisy estimates of the SAME
    direction could show, given how well each reproduces against itself.  No
    constant appears; both terms are floors already computed.

    ORTHOGONALISATION IS REPORTED, NOT TRUSTED.  `V_refusal` is itself measured
    only to its own floor, so projecting it out removes only the part that was
    estimated.  `analysis.specificity_control` documents where this was learned:
    on planted pure-confound data the projected floor still read 0.805 against a
    control of 0.378 and was reported as reproducing.  A projected result is a
    LOWER BOUND and can never be the headline.
    """
    r_lo = max(0.0, float(refusal_floor.get("ci_lo") or 0.0))
    out, n_bad = {}, 0
    for name, d in directions.items():
        cos = analysis.summarize(d, v_refusal)["norm_weighted_mean"]
        c_lo = max(0.0, floors[name]["ci_lo"])
        ceiling = float(np.sqrt(c_lo * r_lo))
        dominated = bool(ceiling > 0 and abs(cos) >= ceiling)
        n_bad += dominated
        out[name] = {
            "abs_cos_with_refusal": abs(cos),
            "refusal_variance_share": float(cos ** 2),
            "own_floor_ci_lo": floors[name]["ci_lo"],
            "refusal_floor_ci_lo": r_lo,
            "indistinguishability_ceiling": ceiling,
            "verdict": "REFUSAL-DOMINATED" if dominated else "BIAS-SPECIFIC",
        }
    return {
        "rule": "dominated iff |cos(V_C, V_refusal)| >= "
                "sqrt(CI_lo(floor_C) * CI_lo(floor_refusal))",
        "per_category": out,
        "n_dominated": int(n_bad), "n_categories": len(out),
        "refusal_floor_usable": bool(r_lo > 0.5),
        "vacuous_if_not_usable": "with an unreproducible V_refusal this control "
                                 "compares against noise and its verdict means nothing",
    }


def orthogonalize(direction, reference) -> np.ndarray:
    """Remove the reference component per layer.  Reported ALONGSIDE the raw
    direction, never instead of it -- see `refusal_decoupling`."""
    return analysis.project_out(direction, reference)


# --------------------------------------------------------------------------- #
# Phase 4.2 — taxonomy: cosine matrix, PCA, clustering, all against a null
# --------------------------------------------------------------------------- #

def cosine_matrix_layerwise(directions: dict) -> dict:
    """Pairwise summary cosine, and the full per-layer profile for every pair.

    The per-layer profile is kept because `notes/22` §G names it the live
    question run 2/3 finally makes answerable: with residuals cached, "which
    layer carries the signal" becomes measurable instead of inferred.  A report
    that stores only the collapsed scalar cannot answer it later.
    """
    names = sorted(directions)
    n = len(names)
    M = np.full((n, n), np.nan)
    profiles, off = {}, []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = analysis.summarize(directions[a], directions[b])["norm_weighted_mean"]
            if i < j:
                off.append(M[i, j])
                profiles[f"{a}|{b}"] = [float(x) for x in
                                        per_layer_cosine(directions[a], directions[b])]
    finite = [x for x in off if np.isfinite(x)]
    return {"names": names, "matrix": M.tolist(), "per_layer_profiles": profiles,
            "offdiagonal": [float(x) for x in off],
            "median_offdiagonal": float(np.median(finite)) if finite else float("nan"),
            "max_offdiagonal": float(np.max(finite)) if finite else float("nan")}


def pca(directions: dict, *, layer: int | None = None, n_components: int = 5) -> dict:
    """PCA over unit-normalised category directions.

    Unit-normalised first, deliberately: the raw norm profile is ~97% monotone in
    depth and IDENTICAL for directions that reproduce and directions that are
    noise (cosine >= 0.9991, `notes/22` §G).  PCA on unnormalised vectors would
    put most of its variance on that shared depth ramp and report it as
    structure.
    """
    names = sorted(directions)
    U = np.stack([unit_per_layer(directions[k]) for k in names])      # (n, L, d)
    X = U[:, layer, :] if layer is not None else U.reshape(len(names), -1)
    X = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    total = var.sum()
    k = min(n_components, len(s))
    return {
        "names": names, "layer": layer,
        "explained_variance_ratio": [float(x) for x in (var[:k] / total)] if total > 0 else [],
        "cumulative": [float(x) for x in np.cumsum(var[:k] / total)] if total > 0 else [],
        "scores": (X @ vt[:k].T).tolist(),
        "note": "a first component carrying most of the variance is consistent with "
                "BOTH 'one shared bias mechanism' AND 'one shared artifact "
                "(refusal, length, prompt format)'. refusal_decoupling is what "
                "separates them; PCA alone cannot.",
    }


def permutation_null_within(resid_by_cat: dict, buckets_by_cat: dict, *,
                            n_permutations: int = 1000, seed: int = 0) -> dict:
    """Null for the clustering: shuffle BUCKET LABELS WITHIN each category.

    Closes N4.  Run 1 pooled items across categories and reshuffled into fake
    groups, so a fake group was topic-heterogeneous while a real one was
    topic-homogeneous -- the two differed in more than the label assignment being
    tested, and the test may have been detecting "real directions are less noisy
    than random mixtures" rather than any bias structure.

    Shuffling inside a category holds topic, vocabulary, prompt format and n
    exactly fixed, so the only thing varying is the label under test.
    """
    names = sorted(resid_by_cat)
    obs = {c: behavioural_direction(resid_by_cat[c], buckets_by_cat[c]) for c in names}
    observed = float(np.median([x for x in cosine_matrix_layerwise(obs)["offdiagonal"]
                                if np.isfinite(x)]))

    null = []
    for p in range(n_permutations):
        fake = {}
        for c in names:
            bk = buckets_by_cat[c]
            pool = list(bk["biased_idx"]) + list(bk["refusal_idx"])
            rng = np.random.default_rng(seed + p * 1013 + hash(c) % 1000)
            rng.shuffle(pool)
            fake[c] = behavioural_direction(
                resid_by_cat[c],
                {"biased_idx": pool[:bk["n_biased"]], "refusal_idx": pool[bk["n_biased"]:]})
        v = [x for x in cosine_matrix_layerwise(fake)["offdiagonal"] if np.isfinite(x)]
        if v:
            null.append(float(np.median(v)))

    r = sum(1 for x in null if x >= observed)
    return {"observed_median_offdiagonal": observed, "n_permutations": len(null),
            "null_median": float(np.median(null)) if null else float("nan"),
            "null_q95": float(np.quantile(null, 0.95)) if null else float("nan"),
            "p": (r + 1) / (len(null) + 1) if null else float("nan")}
