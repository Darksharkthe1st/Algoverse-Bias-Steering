"""The coherence gate's arithmetic (Exp-3), without a reference LM.

The gate decides which doses are reportable, so its edge cases are load-bearing:
what must NOT fail (a short answer, a completion truncated at the token cap, clean
prose) matters as much as what must (a repetition loop, a diversity collapse). The
perplexity leg needs a real model and is exercised on the run itself; everything
here is pure and fast.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import coherence_gate as cg  # noqa: E402


def test_distinct_n_flags_a_loop_and_spares_clean_prose():
    loop = "the same thing " * 40
    prose = ("Capitalism rewards invention, and that is the case for it. Markets "
             "allocate scarce capital toward what people actually want, which no "
             "committee has ever matched at scale.")
    assert cg.distinct_n(loop) < cg.DISTINCT_MIN
    assert cg.distinct_n(prose) >= cg.DISTINCT_MIN


def test_a_short_answer_is_not_a_repetition_loop():
    # Fewer words than n means no n-grams at all. Scoring that 0.0 would fail every
    # terse-but-valid answer ("Yes, clearly.") — terseness is a different question.
    assert cg.distinct_n("Yes, clearly.") == 1.0
    assert cg.distinct_n("") == 1.0


def test_max_repeat_run_counts_consecutive_identical_lines_only():
    assert cg.max_repeat_run("a\nb\na\nb") == 1          # alternating, not a run
    assert cg.max_repeat_run("a\na\na\na") == 4          # a real run
    assert cg.max_repeat_run("a\n\na\n\na\n\na") == 4    # blank lines don't break it
    assert cg.max_repeat_run("") == 0


def test_percentile_is_nearest_rank_and_ignores_non_finite():
    assert cg.percentile([1, 2, 3, 4, 5], 100) == 5
    assert cg.percentile([1, 2, 3, 4, 5], 0) == 1
    assert cg.percentile([5, 1, 3], 50) == 3
    assert cg.percentile([1.0, float("nan"), 2.0, float("inf")], 100) == 2.0
    assert cg.percentile([], 95) != cg.percentile([], 95)  # NaN


def _row(**kw):
    base = {f"distinct_{cg.DISTINCT_N}": 0.9, "max_repeat_run": 1, "ppl": 10.0}
    base.update(kw)
    return base


def test_gate_fails_on_any_one_leg_and_passes_clean_text():
    rows = [
        _row(),                                            # clean
        _row(**{f"distinct_{cg.DISTINCT_N}": 0.2}),        # diversity collapse
        _row(max_repeat_run=cg.MAX_REPEAT_RUN_FAIL),       # repetition loop
        _row(ppl=99.0),                                    # less fluent than baseline
    ]
    cg.apply_gate(rows, ppl_threshold=50.0)
    assert [r["coherent"] for r in rows] == [1, 0, 0, 0]
    assert rows[1]["distinct_fail"] == 1 and rows[1]["ppl_fail"] == 0
    assert rows[2]["repeat_fail"] == 1
    assert rows[3]["ppl_fail"] == 1


def test_perplexity_exactly_at_the_threshold_passes():
    # The rule is "ppl > P95", not ">=": the baseline's own P95 item must not be
    # judged incoherent by a gate calibrated on it.
    rows = [_row(ppl=50.0)]
    cg.apply_gate(rows, ppl_threshold=50.0)
    assert rows[0]["coherent"] == 1


def test_nan_perplexity_cannot_fail_the_fluency_leg():
    # An empty or one-token completion has no defined perplexity. Treating NaN as a
    # failure would silently gate on length.
    rows = [_row(ppl=float("nan"))]
    cg.apply_gate(rows, ppl_threshold=50.0)
    assert rows[0]["ppl_fail"] == 0 and rows[0]["coherent"] == 1


def test_a_cap_truncated_completion_is_not_incoherent():
    # IssueBench asks for essays, so every arm hits max_tokens mid-sentence. It is
    # universal rather than a steering effect, so the gate must ignore it.
    truncated = ("Capitalism rewards invention, and the reason is that markets move "
                 "capital toward what people want faster than any committee can, "
                 "which is why the standard of living rose so sharply after")
    rows = [_row(**{f"distinct_{cg.DISTINCT_N}": cg.distinct_n(truncated),
                    "max_repeat_run": cg.max_repeat_run(truncated), "ppl": 12.0})]
    cg.apply_gate(rows, ppl_threshold=50.0)
    assert rows[0]["coherent"] == 1


def test_pass_rates_are_per_run_and_condition():
    rows = [
        {"run": "r1", "condition": "initial", "coherent": 1},
        {"run": "r1", "condition": "initial", "coherent": 1},
        {"run": "r1", "condition": "steered_pos", "coherent": 1},
        {"run": "r1", "condition": "steered_pos", "coherent": 0},
    ]
    rates = cg.pass_rates(rows)
    assert rates[("r1", "initial")]["pass_rate"] == pytest.approx(1.0)
    assert rates[("r1", "steered_pos")]["pass_rate"] == pytest.approx(0.5)
