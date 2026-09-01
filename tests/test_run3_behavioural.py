"""Run 3 regression tests — the behavioural contrast, its judge, and its controls.

Every test here encodes a failure that was actually hit while building run 3, or
a control that must DISCRIMINATE rather than merely run. The distinction matters:
`notes/11` §6 — a control that has not been shown to fire when it should is a
control that failed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.pilot import behavioural as bh, llm_judge as J, pairing
from scripts import run3_behavioural_contrast as r3


# --------------------------------------------------------------------------- #
# The adapter bug: resolve_answer_roles takes assembled metadata, not a raw row
# --------------------------------------------------------------------------- #

def _rows(category="Religion", n=40, condition="ambig"):
    rows = [r for r in pairing.load_category(category)
            if r["context_condition"] == condition][:n]
    assert rows, f"no {condition} rows in {category}"
    return rows


def test_row_metadata_makes_a_raw_bbq_row_usable():
    """Passing a raw row to `resolve_answer_roles` yields usable=False for every
    item, so the whole category buckets as `unparsed` and the direction is built
    from nothing -- silently. `row_metadata` is the adapter that prevents it."""
    from src.bias_steer.bias_taxonomy import resolve_answer_roles

    rows = _rows()
    raw_usable = sum(bool(resolve_answer_roles(r).usable) for r in rows)
    adapted_usable = sum(bool(resolve_answer_roles(bh.row_metadata(r)).usable)
                         for r in rows)
    assert raw_usable == 0, "a raw row should NOT resolve — that was the bug"
    assert adapted_usable == len(rows)


def test_buckets_are_not_all_unparsed_on_real_bbq_rows():
    rows = _rows()
    resp = [f"{r['ans0']}." if i % 2 else "Cannot be determined."
            for i, r in enumerate(rows)]
    bk = bh.bucket_responses(rows, resp, min_bucket=1)
    assert bk["n_unparsed"] == 0
    assert bk["n_biased"] + bk["n_refusal"] == len(rows)


# --------------------------------------------------------------------------- #
# M1: a small bucket is UNTESTABLE, never a negative
# --------------------------------------------------------------------------- #

def test_small_bucket_is_untestable_not_negative():
    rows = _rows(n=10)
    bk = bh.bucket_responses(rows, ["Cannot be determined."] * len(rows))
    assert bk["status"] == "UNTESTABLE"
    assert "M1" in bk["untestable_reason"]


# --------------------------------------------------------------------------- #
# N6: the position-bias control must DISCRIMINATE, and the cheap one must not
# --------------------------------------------------------------------------- #

def test_option_order_invariance_is_vacuous_by_construction():
    """It scores ~1.0 whatever the labeller does, because reordering the option
    LIST does not change which name appears first in the RESPONSE text. Kept as a
    sanity check; it must never be reported as the position-bias control."""
    rows = _rows(n=20)
    resp = [f"It's not {r['ans1']}, it's {r['ans0']}." for r in rows]
    out = bh.option_order_invariance(resp, rows)
    assert out["order_invariance"] == pytest.approx(1.0)
    assert out["is_the_position_bias_control"] is False


def test_person_swap_consistency_separates_a_stable_labeller_from_a_positional_one():
    rows = _rows(n=60)
    stable = ["Cannot be determined." if i % 2 else f"{r['ans0']}."
              for i, r in enumerate(rows)]
    bk_a = bh.bucket_responses(rows, stable, min_bucket=1)
    good = bh.person_swap_consistency(bk_a, bk_a, rows)
    assert good["consistency"] == pytest.approx(1.0)
    assert good["usable"]

    # A labeller that follows whichever person is named first flips under swap.
    flipped = [f"{r['ans1']}." if i % 2 else f"{r['ans0']}."
               for i, r in enumerate(rows)]
    bk_b = bh.bucket_responses(rows, flipped, min_bucket=1)
    bad = bh.person_swap_consistency(bk_a, bk_b, rows)
    assert bad["consistency"] < 0.9
    assert not bad["usable"]


# --------------------------------------------------------------------------- #
# The floor recovers a planted direction; the shuffled control does not
# --------------------------------------------------------------------------- #

def _planted(rows, *, refusal_weight=0.0, seed=0, n_layers=6, d_model=32):
    rng = np.random.default_rng(seed)
    v_bias = rng.normal(size=(n_layers, d_model))
    v_ref = rng.normal(size=(n_layers, d_model))
    resp = [f"{r['ans0']}." if i % 3 == 0 else "Cannot be determined."
            for i, r in enumerate(rows)]
    bk = bh.bucket_responses(rows, resp, min_bucket=1)
    R = rng.normal(size=(len(rows), n_layers, d_model)) * 0.5
    for i in bk["biased_idx"]:
        R[i] += v_bias + refusal_weight * v_ref
    for i in bk["refusal_idx"]:
        R[i] -= v_bias + refusal_weight * v_ref
    return R, bk, v_bias, v_ref


def test_floor_recovers_a_planted_direction_and_the_control_does_not():
    rows = _rows(n=120)
    R, bk, _, _ = _planted(rows)
    f = bh.bucket_floor(R, bk, n_splits=20)
    n = bh.shuffled_bucket_control(R, bk, n_splits=20, n_shuffles=5)
    assert f["mean"] > 0.8
    assert abs(n["mean"]) < 0.4
    from scripts.pilot import analysis
    assert analysis.reproduces(f, n) == "YES"


def test_split_is_stratified_by_bucket():
    """N5 for unequal buckets: cutting a pooled list blind lets a half drift to
    one arm, adding variance to the floor that is not about reproducibility."""
    bk = {"biased_idx": list(range(40)), "refusal_idx": list(range(40, 130))}
    A, B = bh.split_buckets(bk, seed=3)
    assert len(A["biased_idx"]) + len(B["biased_idx"]) == 40
    assert len(A["refusal_idx"]) + len(B["refusal_idx"]) == 90
    for half in (A, B):
        assert half["biased_idx"] and half["refusal_idx"]


# --------------------------------------------------------------------------- #
# The refusal control — the one that decides whether a taxonomy claim survives
# --------------------------------------------------------------------------- #

def test_refusal_decoupling_flags_a_planted_refusal_direction():
    rows = _rows(n=120)
    R, bk, _, v_ref = _planted(rows, refusal_weight=12.0)
    d = bh.behavioural_direction(R, bk)
    f = bh.bucket_floor(R, bk, n_splits=15)
    dec = bh.refusal_decoupling({"C": d}, {"C": f}, v_ref, {"ci_lo": 0.95})
    assert dec["per_category"]["C"]["verdict"] == "REFUSAL-DOMINATED"


def test_refusal_decoupling_is_quiet_when_no_refusal_component_is_planted():
    rows = _rows(n=120)
    R, bk, _, v_ref = _planted(rows, refusal_weight=0.0)
    d = bh.behavioural_direction(R, bk)
    f = bh.bucket_floor(R, bk, n_splits=15)
    dec = bh.refusal_decoupling({"C": d}, {"C": f}, v_ref, {"ci_lo": 0.95})
    assert dec["per_category"]["C"]["verdict"] == "BIAS-SPECIFIC"


def test_refusal_control_is_reported_vacuous_rather_than_passing_without_a_floor():
    rows = _rows(n=60)
    R, bk, _, v_ref = _planted(rows, refusal_weight=12.0)
    d = bh.behavioural_direction(R, bk)
    f = bh.bucket_floor(R, bk, n_splits=10)
    dec = bh.refusal_decoupling({"C": d}, {"C": f}, v_ref, {"ci_lo": None})
    assert dec["refusal_floor_usable"] is False


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

def test_directive_labels_map_from_positional_verdicts():
    m = lambda ch: J.to_directive_label(ch, target_loc=0, unknown_idx=2)
    assert m("OPTION_1") == J.BIASED_TARGET
    assert m("OPTION_2") == J.BIASED_DISTRACTOR
    assert m("OPTION_3") == J.REFUSAL          # BBQ's unknown option
    assert m("REFUSED") == J.REFUSAL
    assert m("UNCLEAR") == J.UNKNOWN
    assert m(None) == J.UNKNOWN                # format failure, never a class


def test_unlabelled_row_never_guesses_a_bias_label():
    assert J.to_directive_label("OPTION_1", target_loc=None,
                                unknown_idx=2) == J.UNKNOWN


def test_distractor_is_in_neither_arm_by_default():
    labels = [J.BIASED_TARGET] * 40 + [J.REFUSAL] * 40 + [J.BIASED_DISTRACTOR] * 20
    bk = J.buckets_from_labels(labels, min_bucket=5)
    assert bk["n_biased"] == 40 and bk["n_refusal"] == 40
    assert bk["n_distractor"] == 20
    assert len(bk["biased_idx"]) + len(bk["refusal_idx"]) == 80

    folded = J.buckets_from_labels(labels, min_bucket=5,
                                   include_distractor_in_refusal=True)
    assert folded["n_refusal"] == 60


def test_judge_qualification_catches_a_position_biased_judge():
    items = [{"scenario": "s", "options": ["a", "b", "c"], "response": "r"}] * 40
    ok = J.qualify(items, client=lambda p: "REFUSED", n_sample=20)
    assert ok["order_agreement"] == pytest.approx(1.0) and ok["qualified"]

    bad = J.qualify(items, client=lambda p: "OPTION_1", n_sample=20)
    assert bad["order_agreement"] == pytest.approx(0.0) and not bad["qualified"]


def test_batch_capable_judge_client_is_used_in_one_call():
    n = {"calls": 0}

    def cli(p):
        n["calls"] += 1
        return "REFUSED"

    def batch(ps):
        n["calls"] += 1
        return ["OPTION_1"] * len(ps)

    cli.batch = batch
    items = [{"scenario": "s", "options": ["a", "b", "c"], "response": "r"}] * 25
    out = J.judge_batch(items, client=cli)
    assert n["calls"] == 1 and set(out) == {"OPTION_1"}


def test_local_verdict_tokens_are_distinct():
    """The local judge is an argmax over these; two sharing a first token would
    make the verdict meaningless."""
    assert len(set(J.LOCAL_TOKENS)) == len(J.LOCAL_TOKENS) == 5


# --------------------------------------------------------------------------- #
# Steering dose
# --------------------------------------------------------------------------- #

def test_dose_is_dimensionless_across_a_large_per_layer_norm_spread():
    """Per-layer residual norms span 600-1391x on these models. A fixed alpha
    without this scaling injects a wildly different perturbation per layer."""
    rng = np.random.default_rng(0)
    scale = np.array([1, 3, 10, 30, 100, 300, 600, 1000], dtype=float)
    resid = rng.normal(size=(40, 8, 64)) * scale[None, :, None]
    D = rng.normal(size=(8, 64)) * 7.0
    for alpha in (0.25, 1.0, 2.0):
        v = r3.dose_vector(D, resid, alpha)
        ratio = np.linalg.norm(v, axis=1) / np.linalg.norm(resid, axis=2).mean(axis=0)
        assert np.allclose(ratio, alpha, rtol=1e-5)


def test_random_control_matches_the_per_layer_norm_profile():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(8, 64)) * np.arange(1, 9)[:, None]
    R = r3._matched_random(D, seed=0)
    assert np.allclose(np.linalg.norm(R, axis=1), np.linalg.norm(D, axis=1), rtol=1e-5)


# --------------------------------------------------------------------------- #
# The generation prompt must carry the option list
# --------------------------------------------------------------------------- #

def test_generation_prompt_includes_the_options_and_swap_reorders_only_the_named_pair():
    """The residual must come from the SAME string the completion came from.
    `capture_arm` builds a prompt with no option list, which is right for R1 and
    wrong here -- capturing a different prompt than the one that produced the
    bucket label breaks the correspondence the contrast rests on."""
    r = _rows(n=1)[0]
    p = r3._prompt_with_options(r)
    s = r3._prompt_with_options(r, swap=True)
    for a in (r["ans0"], r["ans1"], r["ans2"]):
        assert a in p and a in s
    assert p.index(r["ans0"]) < p.index(r["ans1"])
    assert s.index(r["ans1"]) < s.index(r["ans0"])
    assert p.split("Pick one of three options:")[1] != \
        s.split("Pick one of three options:")[1]


# --------------------------------------------------------------------------- #
# GATE 2 — the positive control's own logic
# --------------------------------------------------------------------------- #

def test_topic_control_buckets_recover_a_planted_topic_direction():
    """Gate 2 builds its arms from two CATEGORIES and then runs the real
    estimator unchanged. If this cannot recover a planted topic separation, a
    bias null downstream says nothing about bias."""
    from scripts.pilot import analysis

    n, L, D = 60, 6, 32
    rng = np.random.default_rng(4)
    topic = rng.normal(size=(L, D))
    R = rng.normal(size=(2 * n, L, D)) * 0.5
    R[:n] += topic
    R[n:] -= topic
    buckets = {"biased_idx": list(range(n)), "refusal_idx": list(range(n, 2 * n)),
               "n_biased": n, "n_refusal": n, "n_total": 2 * n}
    fl = bh.bucket_floor(R, buckets, n_splits=15)
    ng = bh.shuffled_bucket_control(R, buckets, n_splits=15, n_shuffles=5)
    assert fl["mean"] > 0.8
    assert analysis.reproduces(fl, ng) == "YES"


def test_topic_control_reports_no_when_there_is_nothing_to_find():
    """The gate must fail on noise, or it is decoration rather than a control."""
    from scripts.pilot import analysis

    n, L, D = 60, 6, 32
    R = np.random.default_rng(5).normal(size=(2 * n, L, D))
    buckets = {"biased_idx": list(range(n)), "refusal_idx": list(range(n, 2 * n)),
               "n_biased": n, "n_refusal": n, "n_total": 2 * n}
    fl = bh.bucket_floor(R, buckets, n_splits=15)
    ng = bh.shuffled_bucket_control(R, buckets, n_splits=15, n_shuffles=5)
    assert analysis.reproduces(fl, ng) != "YES"
