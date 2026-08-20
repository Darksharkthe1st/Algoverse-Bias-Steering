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


# --- BBQ's own answer key (target_loc) takes precedence ------------------- #

def test_target_loc_is_used_directly_when_present():
    meta = _meta("neg") | {"target_loc": 0}
    roles = bt.resolve_answer_roles(meta)
    assert roles.biased == 0
    assert roles.source == "target_loc"


def test_target_loc_is_NOT_flipped_by_polarity():
    """The double-inversion trap. BBQ has already folded polarity into
    target_loc, so the same key must give the same answer under both
    polarities. Flipping it here would invert half the labels back to wrong
    and raise nothing."""
    for polarity in ("neg", "nonneg"):
        roles = bt.resolve_answer_roles(_meta(polarity) | {"target_loc": 2})
        assert roles.biased == 2, f"target_loc was flipped under polarity={polarity}"
        assert roles.source == "target_loc"


def test_target_loc_wins_over_the_reconstruction():
    """Where the two disagree, the dataset authors' label is used."""
    roles = bt.resolve_answer_roles(_meta("neg") | {"target_loc": 0})
    assert roles.stereo == 2          # reconstruction still reports what it found
    assert roles.biased == 0          # but the key decides
    assert roles.source == "target_loc"


def test_falls_back_to_reconstruction_when_target_loc_missing_or_NA():
    for bad in (None, "NA", "", "abc"):
        roles = bt.resolve_answer_roles(_meta("neg") | {"target_loc": bad})
        assert roles.biased == 2
        assert roles.source == "reconstructed"


def test_target_loc_pointing_at_unknown_is_rejected():
    """Never observed in the real file (0 of 25,814 ambiguous rows), but a key
    that names the correct answer as the biased one would silently invert the
    contrast, so it falls back rather than being trusted."""
    roles = bt.resolve_answer_roles(_meta("neg") | {"target_loc": 1})
    assert roles.source == "reconstructed"
    assert roles.biased == 2


def test_out_of_range_target_loc_is_rejected():
    roles = bt.resolve_answer_roles(_meta("neg") | {"target_loc": 7})
    assert roles.source == "reconstructed"


def test_target_loc_resolves_the_intersectional_case_reconstruction_cannot():
    """Both answers are Black, so stereotyped_groups alone cannot say which is
    biased — this is exactly what target_loc exists to settle."""
    meta = {
        "answers": ["The Black woman", "Not enough info", "The Black man"],
        "answer_groups": ["F-Black", "unknown", "M-Black"],
        "stereotyped_groups": ["Black"],
        "question_polarity": "neg",
    }
    assert not bt.resolve_answer_roles(meta).usable          # reconstruction fails
    roles = bt.resolve_answer_roles(meta | {"target_loc": 2})
    assert roles.usable and roles.biased == 2 and roles.source == "target_loc"


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


# --- position matching, and what it costs --------------------------------- #

def test_matching_equalizes_position_profiles_across_buckets():
    """After matching, no linear direction between the buckets can encode
    position, because both have an identical position profile."""
    biased = [(f"b{i}", 0) for i in range(60)] + [(f"b{i}", 1) for i in range(60, 80)]
    other = [(f"o{i}", 0) for i in range(20)] + [(f"o{i}", 1) for i in range(20, 80)]

    matched, report = bt.match_position_distribution(
        {"biased": biased, "other": other}, seed=0)

    # min at position 0 is 20, min at position 1 is 20 -> 20 each, both buckets
    assert report["biased"]["kept_by_position"] == {0: 20, 1: 20}
    assert report["other"]["kept_by_position"] == {0: 20, 1: 20}
    assert len(matched["biased"]) == len(matched["other"]) == 40


def test_matching_reports_the_loss_rather_than_hiding_it():
    biased = [(f"b{i}", 0) for i in range(100)]
    other = [(f"o{i}", 0) for i in range(30)]
    _, report = bt.match_position_distribution({"biased": biased, "other": other})
    assert report["biased"] == {"before": 100, "after": 30, "lost": 70,
                                "kept_by_position": {0: 30}}
    assert report["other"]["lost"] == 0


def test_matching_costs_nothing_when_buckets_already_agree():
    b = [(f"b{i}", i % 3) for i in range(90)]
    o = [(f"o{i}", i % 3) for i in range(90)]
    matched, report = bt.match_position_distribution({"biased": b, "other": o})
    assert report["biased"]["lost"] == 0 and report["other"]["lost"] == 0
    assert len(matched["biased"]) == 90


def test_matching_drops_a_position_absent_from_one_bucket():
    """min over buckets is 0 there, so that position disappears from both."""
    b = [("b1", 0), ("b2", 2)]
    o = [("o1", 0)]
    matched, report = bt.match_position_distribution({"biased": b, "other": o})
    assert report["biased"]["kept_by_position"] == {0: 1, 2: 0}
    assert matched["biased"] == ["b1"]


def test_matching_is_seeded_and_reproducible():
    b = [(f"b{i}", 0) for i in range(50)]
    o = [(f"o{i}", 0) for i in range(10)]
    m1, _ = bt.match_position_distribution({"b": b, "o": o}, seed=3)
    m2, _ = bt.match_position_distribution({"b": b, "o": o}, seed=3)
    m3, _ = bt.match_position_distribution({"b": b, "o": o}, seed=4)
    assert m1["b"] == m2["b"]
    assert m1["b"] != m3["b"]


def test_matching_rejects_empty_input():
    with pytest.raises(ValueError, match="no buckets"):
        bt.match_position_distribution({})


def test_format_matching_loss_shows_before_after_and_percentage():
    _, report = bt.match_position_distribution(
        {"biased": [(f"b{i}", 0) for i in range(100)],
         "other": [(f"o{i}", 0) for i in range(25)]})
    text = bt.format_matching_loss(report)
    assert "100 ->    25" in text and "75%" in text


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
    r = bt.TaxonomyReport(p_value=0.42, floors={"a": {"q05": 0.9, "n_items": 100}})
    assert "NO STRUCTURE" in r.verdict()


def test_report_refuses_to_claim_structure_without_an_extraction_floor():
    r = bt.TaxonomyReport(p_value=0.001, floors={})
    assert "not reportable" in r.verdict()


def test_report_quotes_the_worst_floor_with_its_n():
    """A floor is never a bare number — the n it was computed on rides along."""
    r = bt.TaxonomyReport(p_value=0.001, floors={
        "race": {"q05": 0.88, "median": 0.92, "n_items": 200},
        "religion": {"q05": 0.91, "median": 0.94, "n_items": 180},
    })
    v = r.verdict()
    assert "STRUCTURE" in v and "0.880" in v and "race" in v and "n=200" in v


def test_report_warns_when_n_varies_widely_across_categories():
    """One global floor threshold is not applicable when the categories were
    measured on very different sample sizes."""
    r = bt.TaxonomyReport(p_value=0.001, floors={
        "big": {"q05": 0.95, "median": 0.96, "n_items": 400},
        "small": {"q05": 0.62, "median": 0.70, "n_items": 40},
    })
    v = r.verdict()
    assert "10.0x" in v and "do NOT apply" in v


def test_report_does_not_warn_when_n_is_comparable():
    r = bt.TaxonomyReport(p_value=0.001, floors={
        "a": {"q05": 0.90, "median": 0.93, "n_items": 200},
        "b": {"q05": 0.92, "median": 0.94, "n_items": 180},
    })
    assert "do NOT apply" not in r.verdict()


def test_n_spread_is_none_with_fewer_than_two_floors():
    assert bt.TaxonomyReport(floors={"a": {"n_items": 10}}).n_spread() is None


def test_floor_table_shows_n_beside_every_floor():
    r = bt.TaxonomyReport(floors={
        "race": {"q05": 0.88, "median": 0.92, "n_items": 200},
        "age": {"q05": 0.60, "median": 0.71, "n_items": 45},
    })
    t = r.floor_table()
    assert "age" in t and "45" in t and "0.600" in t
    assert t.index("age") < t.index("race")      # worst floor first


# --- how much of the floor is just sample size? ---------------------------- #

def test_floor_vs_n_shows_the_floor_degrading_as_n_shrinks():
    """The insurance against reading 'small category has a low floor' as
    'small category has an unstable direction'."""
    rng = np.random.default_rng(11)
    signal = _direction(rng)
    items = [signal + 1.6 * _direction(rng) for _ in range(400)]

    res = bt.floor_vs_n(items, _mean_extract, [400, 40], n_splits=6, seed=0)
    assert sorted(res) == [40, 400]
    assert res[400]["median"] > res[40]["median"], (
        "with a fixed underlying direction, more items must give a tighter floor")
    assert res[400]["n_items"] == 400 and res[40]["n_items"] == 40


def test_floor_vs_n_skips_sizes_it_cannot_supply():
    rng = np.random.default_rng(12)
    items = [_direction(rng) for _ in range(50)]
    res = bt.floor_vs_n(items, _mean_extract, [20, 50, 5000], n_splits=3)
    assert sorted(res) == [20, 50]


def test_floor_vs_n_errors_when_no_size_is_usable():
    rng = np.random.default_rng(13)
    items = [_direction(rng) for _ in range(10)]
    with pytest.raises(ValueError, match="no usable sizes"):
        bt.floor_vs_n(items, _mean_extract, [500, 900])


def test_summarize_floor_vs_n_orders_smallest_first():
    rng = np.random.default_rng(14)
    items = [_direction(rng) for _ in range(200)]
    res = bt.floor_vs_n(items, _mean_extract, [200, 20], n_splits=3)
    text = bt.summarize_floor_vs_n(res)
    assert text.index("      20") < text.index("     200")


def test_report_is_incomplete_without_a_null():
    assert "incomplete" in bt.TaxonomyReport().verdict()


# --------------------------------------------------------------------------- #
# The BBQ judge — the adapter that lets experiment.run bucket by bias
# --------------------------------------------------------------------------- #

from dataclasses import dataclass, field as _field   # noqa: E402

from src.bias_steer.judge import bbq_choice_judge     # noqa: E402
from src.bias_steer.registry import JUDGES            # noqa: E402


@dataclass
class _Ex:
    metadata: dict = _field(default_factory=dict)


def _bbq_ex(polarity="neg"):
    return _Ex(metadata={
        "answers": ["The Jewish one", "Can't answer", "The Muslim one"],
        "answer_groups": ["Jewish", "unknown", "Muslim"],
        "stereotyped_groups": ["Muslim"],
        "question_polarity": polarity,
    })


def test_bbq_judge_labels_the_three_outcomes():
    exs = [_bbq_ex(), _bbq_ex(), _bbq_ex()]
    got = bbq_choice_judge(["The Muslim one", "Can't answer", "The Jewish one"], exs)
    assert got == ["biased", "unknown", "other"]


def test_bbq_judge_marks_unparseable_responses_unresolved_not_unbiased():
    """An unreadable generation must not become evidence of unbiased behaviour."""
    got = bbq_choice_judge(["hmm, hard to say", ""], [_bbq_ex(), _bbq_ex()])
    assert got == ["unresolved", "unresolved"]


def test_bbq_judge_marks_unscoreable_rows_unresolved():
    """Both answers share the stereotyped group (the intersectional case), so the
    row cannot say which answer is biased."""
    ex = _Ex(metadata={
        "answers": ["The Black woman", "Not enough info", "The Black man"],
        "answer_groups": ["F-Black", "unknown", "M-Black"],
        "stereotyped_groups": ["Black"],
        "question_polarity": "neg",
    })
    assert bbq_choice_judge(["The Black woman"], [ex]) == ["unresolved"]


def test_bbq_judge_respects_polarity_inversion():
    """Same response, opposite polarity -> opposite label. The silent-flip guard."""
    resp = ["The Muslim one"]
    assert bbq_choice_judge(resp, [_bbq_ex("neg")]) == ["biased"]
    assert bbq_choice_judge(resp, [_bbq_ex("nonneg")]) == ["other"]


def test_bbq_judge_requires_examples():
    with pytest.raises(ValueError, match="needs `examples`"):
        bbq_choice_judge(["The Muslim one"], None)


def test_bbq_judge_is_registered():
    assert "bbq_choice" in JUDGES


def test_config_contrast_points_from_unknown_toward_biased():
    """`_contrast` takes (labels[1], labels[0]) as (positive, negative). If this
    inverts, every direction silently flips sign with no error."""
    from src.bias_steer.experiment import _contrast
    import importlib.util

    spec = importlib.util.spec_from_file_location("cfg_bt", "configs/bias_taxonomy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = mod.make_config("Religion")
    assert _contrast(cfg) == ("biased", "unknown")
    assert cfg.judge.name == "bbq_choice"
    assert cfg.sample.filter == {"context_condition": ["ambig"]}


# --- position-confound diagnostic ----------------------------------------- #

def _load_base_rates_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("brs", "scripts/bbq_base_rates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_position_profile_counts_the_index_the_model_emitted():
    brs = _load_base_rates_module()
    ex = _bbq_ex()
    responses = ["The Muslim one", "The Jewish one", "Can't answer"]
    verdicts = ["biased", "other", "unknown"]
    prof = brs.position_profile(responses, [ex, ex, ex], verdicts)
    assert prof["biased"]["share"] == [0.0, 0.0, 1.0]   # "The Muslim one" is idx 2
    assert prof["other"]["share"] == [1.0, 0.0, 0.0]    # "The Jewish one" is idx 0
    assert prof["unknown"]["n"] == 1


def test_position_profile_skips_unresolved_and_unparseable():
    brs = _load_base_rates_module()
    ex = _bbq_ex()
    prof = brs.position_profile(
        ["hmm", "The Muslim one"], [ex, ex], ["unresolved", "biased"])
    assert "unresolved" not in prof
    assert prof["biased"]["n"] == 1


def test_max_position_gap_detects_a_skew_between_buckets():
    brs = _load_base_rates_module()
    prof = {
        "biased": {"n": 100, "share": [0.55, 0.25, 0.20]},
        "other":  {"n": 100, "share": [0.30, 0.35, 0.35]},
    }
    assert brs.max_position_gap(prof, "biased", "other") == pytest.approx(0.25)


def test_max_position_gap_is_zero_when_buckets_match():
    brs = _load_base_rates_module()
    prof = {
        "biased": {"n": 90, "share": [0.34, 0.33, 0.33]},
        "other":  {"n": 90, "share": [0.34, 0.33, 0.33]},
    }
    assert brs.max_position_gap(prof, "biased", "other") == pytest.approx(0.0)


def test_max_position_gap_is_none_for_a_missing_or_empty_bucket():
    brs = _load_base_rates_module()
    prof = {"biased": {"n": 10, "share": [1.0, 0.0, 0.0]},
            "other": {"n": 0, "share": [0.0, 0.0, 0.0]}}
    assert brs.max_position_gap(prof, "biased", "other") is None
    assert brs.max_position_gap(prof, "biased", "absent") is None


def test_base_rate_gate_requires_the_other_bucket_too():
    """`other` is the negative pole of the contrast the paper leans on, so a
    category with an empty `other` bucket must not pass."""
    brs = _load_base_rates_module()
    assert "other" in brs.REQUIRED_BUCKETS
    assert "biased" in brs.REQUIRED_BUCKETS


def test_config_rejects_an_unknown_category():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg_bt2", "configs/bias_taxonomy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match="unknown BBQ category"):
        mod.make_config("Politics")
