"""Tests for the bias-taxonomy analysis layer (JZ-1/JZ-2).

The module is torch-free by design, so everything here runs on CPU with no model
download. Directions are synthesized, which lets the decision rule be tested
against KNOWN ground truth — including the case that matters most: data with no
real structure must be reported as having none.
"""

import numpy as np
import pytest

from src.bias_steer import bias_taxonomy as bt


N_LAYERS, D_MODEL = 8, 64


def _direction(rng, base=None, noise=0.0):
    """A synthetic (n_layers, d_model) direction, optionally a noisy copy of `base`."""
    v = rng.normal(size=(N_LAYERS, D_MODEL))
    if base is not None:
        v = base + noise * v
    return v


# --------------------------------------------------------------------------- #
# Shape discipline — the guard against the 2025 bug
# --------------------------------------------------------------------------- #

def test_assert_direction_rejects_1d_with_an_explanatory_message():
    with pytest.raises(bt.DirectionShapeError) as e:
        bt.assert_direction(np.zeros(D_MODEL))
    assert "1-D" in str(e.value)
    assert "scalar" in str(e.value)  # names the actual failure mode


def test_assert_direction_rejects_3d_grid_and_nonfinite():
    with pytest.raises(bt.DirectionShapeError):
        bt.assert_direction(np.zeros((5, N_LAYERS, D_MODEL)))
    bad = np.zeros((N_LAYERS, D_MODEL))
    bad[0, 0] = np.nan
    with pytest.raises(bt.DirectionShapeError):
        bt.assert_direction(bad)


def test_per_layer_cosine_refuses_a_1d_operand():
    rng = np.random.default_rng(0)
    with pytest.raises(bt.DirectionShapeError):
        bt.per_layer_cosine(_direction(rng), np.zeros(D_MODEL))


# --------------------------------------------------------------------------- #
# BBQ answer roles — deterministic labelling, polarity is the subtle part
# --------------------------------------------------------------------------- #

def _meta(polarity="neg"):
    # Mirrors a real Religion row: ans0 Jewish, ans1 unknown, ans2 Muslim.
    return {
        "answer_groups": ["Jewish", "unknown", "Muslim"],
        "stereotyped_groups": ["Muslim"],
        "question_polarity": polarity,
    }


def test_negative_polarity_biased_answer_is_the_stereotyped_group():
    roles = bt.resolve_answer_roles(_meta("neg"))
    assert (roles.unknown, roles.stereo, roles.nonstereo) == (1, 2, 0)
    assert roles.biased == 2
    assert roles.usable


def test_nonnegative_polarity_inverts_which_answer_is_biased():
    """The half-the-labels-silently-inverted trap: on "who was the victim?" the
    biased response names the NON-stereotyped group."""
    roles = bt.resolve_answer_roles(_meta("nonneg"))
    assert roles.stereo == 2
    assert roles.biased == 0
    assert roles.biased != roles.stereo


def test_unknown_polarity_refuses_to_guess():
    roles = bt.resolve_answer_roles(_meta("???"))
    assert roles.biased is None
    assert not roles.usable


def test_missing_stereotype_metadata_yields_no_biased_index():
    roles = bt.resolve_answer_roles(
        {"answer_groups": ["A", "unknown", "B"], "stereotyped_groups": [],
         "question_polarity": "neg"}
    )
    assert roles.stereo is None and roles.biased is None
    assert roles.unknown == 1          # still found the unknown option


@pytest.mark.parametrize("marker", ["unknown", "Cannot be determined", "Not enough info"])
def test_unknown_option_recognized_across_bbq_phrasings(marker):
    roles = bt.resolve_answer_roles(
        {"answer_groups": ["Jewish", marker, "Muslim"],
         "stereotyped_groups": ["Muslim"], "question_polarity": "neg"}
    )
    assert roles.unknown == 1


# --------------------------------------------------------------------------- #
# Answer parsing — an unparsed response must never become a behaviour class
# --------------------------------------------------------------------------- #

ANSWERS = ["The Jewish one", "Can't answer", "The Muslim one"]


def test_parse_choice_exact_match():
    assert bt.parse_choice("The Muslim one", ANSWERS) == 2
    assert bt.parse_choice("I think: Can't answer.", ANSWERS) == 1


def test_parse_choice_returns_none_when_nothing_matches():
    assert bt.parse_choice("I'd rather not say", ANSWERS) is None
    assert bt.parse_choice("", ANSWERS) is None


def test_parse_choice_returns_none_when_two_answers_match_equally():
    ambiguous = ["red", "blue", "green"]
    assert bt.parse_choice("maybe red or blue", ambiguous) is None


def test_parse_choice_prefers_the_more_specific_answer():
    """"The Muslim one" contains "Muslim"; the longer, more specific option wins."""
    assert bt.parse_choice("The Muslim one", ["Muslim", "Can't answer", "The Muslim one"]) == 2


def test_unparsed_responses_are_counted_separately_not_as_unbiased():
    c = bt.ChoiceCounts(biased=3, unknown=5, other=2, unparsed=90)
    assert c.scored == 10                 # unparsed excluded from the denominator
    assert c.bias_rate == pytest.approx(0.3)


def test_bias_rate_is_none_when_nothing_scored():
    assert bt.ChoiceCounts(unparsed=7).bias_rate is None


# --------------------------------------------------------------------------- #
# The two floors
# --------------------------------------------------------------------------- #

def test_random_floor_matches_one_over_sqrt_d():
    assert bt.random_floor(2048) == pytest.approx(0.0221, abs=1e-4)
    assert bt.random_floor(4096) == pytest.approx(0.0156, abs=1e-4)


def test_identical_directions_have_cosine_one_and_orthogonal_have_zero():
    rng = np.random.default_rng(1)
    v = _direction(rng)
    assert bt.summarize_cosine(bt.per_layer_cosine(v, v)) == pytest.approx(1.0)

    a = np.zeros((N_LAYERS, D_MODEL)); a[:, 0] = 1.0
    b = np.zeros((N_LAYERS, D_MODEL)); b[:, 1] = 1.0
    assert bt.summarize_cosine(bt.per_layer_cosine(a, b)) == pytest.approx(0.0)


def test_zero_norm_layer_becomes_nan_not_a_spurious_zero():
    rng = np.random.default_rng(2)
    a, b = _direction(rng), _direction(rng)
    a[3, :] = 0.0
    cos = bt.per_layer_cosine(a, b)
    assert np.isnan(cos[3])
    assert np.isfinite(np.delete(cos, 3)).all()
    assert np.isfinite(bt.summarize_cosine(cos))   # median skips the NaN


def test_split_half_is_seeded_disjoint_and_complete():
    items = list(range(11))
    a, b = bt.split_half(items, seed=7)
    assert sorted(a + b) == items
    assert not set(a) & set(b)
    assert (a, b) == bt.split_half(items, seed=7)          # reproducible
    assert (a, b) != bt.split_half(items, seed=8)          # seed actually matters


def test_extraction_floor_is_high_for_a_stable_topic_and_low_for_noise():
    """The floor must be able to come out LOW — that is the whole point of it."""
    rng = np.random.default_rng(3)
    signal = _direction(rng)

    # Stable topic: every item is the same direction plus a little noise.
    stable = [signal + 0.05 * _direction(rng) for _ in range(40)]
    # Unstable topic: every item is unrelated noise, no shared direction at all.
    unstable = [_direction(rng) for _ in range(40)]

    def mean_extract(subset):
        return np.mean(np.stack(subset), axis=0)

    hi = bt.extraction_floor(stable, mean_extract, n_splits=6, seed=0)
    lo = bt.extraction_floor(unstable, mean_extract, n_splits=6, seed=0)

    assert hi["median"] > 0.95
    assert lo["median"] < 0.5
    assert hi["q05"] <= hi["median"] <= hi["q95"]
    assert hi["n_items"] == 40 and hi["n_splits"] == 6


def test_extraction_floor_rejects_too_few_items():
    with pytest.raises(ValueError, match="at least 4"):
        bt.extraction_floor([1, 2], lambda s: np.zeros((N_LAYERS, D_MODEL)))


def test_distinguishable_uses_the_floor_not_zero():
    # 0.55 looks "different" against 0, but not against a floor of 0.60.
    assert not bt.distinguishable(0.55, floor=0.60)
    assert bt.distinguishable(0.55, floor=0.95)


# --------------------------------------------------------------------------- #
# Structure and its null
# --------------------------------------------------------------------------- #

def _mean_extract(subset):
    return np.mean(np.stack(subset), axis=0)


def _topics(rng, *, n_groups, per_group, items, spread):
    """Items in `n_groups` latent groups. `spread=0` => all topics identical."""
    bases = [_direction(rng) for _ in range(n_groups)]
    out = {}
    for t in range(n_groups * per_group):
        base = bases[t % n_groups]
        centre = base + spread * _direction(rng)
        out[f"topic{t}"] = [centre + 0.1 * _direction(rng) for _ in range(items)]
    return out


def test_cosine_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(4)
    dirs = {"a": _direction(rng), "b": _direction(rng), "c": _direction(rng)}
    names, M = bt.cosine_matrix(dirs)
    assert names == ["a", "b", "c"]
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)


def test_cosine_matrix_needs_two_topics():
    with pytest.raises(ValueError, match="at least 2"):
        bt.cosine_matrix({"only": np.zeros((N_LAYERS, D_MODEL))})


def test_clustering_recovers_planted_groups():
    """Sanity: when groups really exist, the dendrogram must find them."""
    from scipy.cluster.hierarchy import fcluster

    rng = np.random.default_rng(5)
    topics = _topics(rng, n_groups=2, per_group=3, items=12, spread=0.05)
    dirs = {k: _mean_extract(v) for k, v in topics.items()}
    names, M = bt.cosine_matrix(dirs)
    Z = bt.cluster_topics(names, M)

    labels = fcluster(Z, t=2, criterion="maxclust")
    # topics alternate between the two planted groups by construction
    planted = [int(n.replace("topic", "")) % 2 for n in names]
    same = [labels[i] == labels[j]
            for i in range(len(names)) for j in range(i + 1, len(names))
            if planted[i] == planted[j]]
    assert all(same), "topics from the same planted group were split apart"


def test_permutation_null_flags_structureless_data():
    """The load-bearing negative case. All topics share ONE direction, so any
    apparent clustering is noise and the null must not be beaten."""
    rng = np.random.default_rng(6)
    topics = _topics(rng, n_groups=1, per_group=6, items=12, spread=0.0)
    dirs = {k: _mean_extract(v) for k, v in topics.items()}
    names, M = bt.cosine_matrix(dirs)
    observed = bt.cluster_strength(bt.cluster_topics(names, M))

    null = bt.permutation_null(topics, _mean_extract, n_permutations=30, seed=0)
    p = bt.null_p_value(observed, null)
    assert p > 0.05, f"reported structure in structureless data (p={p})"


def test_permutation_null_detects_real_structure():
    rng = np.random.default_rng(7)
    topics = _topics(rng, n_groups=2, per_group=3, items=12, spread=0.02)
    dirs = {k: _mean_extract(v) for k, v in topics.items()}
    names, M = bt.cosine_matrix(dirs)
    observed = bt.cluster_strength(bt.cluster_topics(names, M))

    null = bt.permutation_null(topics, _mean_extract, n_permutations=30, seed=0)
    assert bt.null_p_value(observed, null) <= 0.05


def test_null_p_value_never_returns_exactly_zero():
    null = {"strengths": [0.0] * 50}
    assert bt.null_p_value(999.0, null) == pytest.approx(1 / 51)


def test_cluster_strength_is_zero_for_a_single_merge():
    Z = np.array([[0.0, 1.0, 0.5, 2.0]])
    assert bt.cluster_strength(Z) == 0.0


# --------------------------------------------------------------------------- #
# The report reads out honestly in both directions
# --------------------------------------------------------------------------- #

def test_report_says_no_structure_when_the_null_is_not_beaten():
    r = bt.TaxonomyReport(p_value=0.42, floors={"a": {"q05": 0.9}})
    assert "NO STRUCTURE" in r.verdict()


def test_report_refuses_to_claim_structure_without_an_extraction_floor():
    r = bt.TaxonomyReport(p_value=0.001, floors={})
    assert "not reportable" in r.verdict()


def test_report_quotes_the_floor_when_it_claims_structure():
    r = bt.TaxonomyReport(p_value=0.001, floors={"a": {"q05": 0.88}, "b": {"q05": 0.91}})
    v = r.verdict()
    assert "STRUCTURE" in v and "0.880" in v      # reports the WORST floor


def test_report_is_incomplete_without_a_null():
    assert "incomplete" in bt.TaxonomyReport().verdict()
