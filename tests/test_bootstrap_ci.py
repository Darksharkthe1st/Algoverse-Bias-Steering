"""The judged-CSV statistics: agreement with `metrics.beat_rate`, and the rules
about what may not enter a denominator.

`scripts/bootstrap_ci.py` recomputes after the fact what `metrics.beat_rate`
computes inside a run. Two implementations of one statistic is a liability unless
something holds them to the same answer, so the load-bearing test here is that they
agree on identical data. The rest pin the two rules that are easy to get quietly
wrong: a judge extraction failure is not a behaviour, and a pooled label set is not
a single label.
"""

import csv
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import bootstrap_ci as bc  # noqa: E402
from src.bias_steer import metrics  # noqa: E402
from src.bias_steer.schema import Result  # noqa: E402


def _rows(spec):
    """spec: {example_id: {condition: label}} -> judged-CSV rows."""
    return [{"example_id": ex, "condition": cond, "verdict_collapsed": label}
            for ex, conds in spec.items() for cond, label in conds.items()]


def test_paired_margin_and_cells_match_metrics_beat_rate():
    # Same data through both implementations; the point estimate and all four
    # McNemar cells must agree exactly. (The CIs use different RNG streams, so
    # only the deterministic parts are compared.)
    spec = {
        "e1": {"steered_pos": "stance-evaluative", "prompt_pos": "soft-refusal"},
        "e2": {"steered_pos": "soft-refusal", "prompt_pos": "stance-factual"},
        "e3": {"steered_pos": "stance-factual", "prompt_pos": "stance-evaluative"},
        "e4": {"steered_pos": "soft-refusal", "prompt_pos": "soft-refusal"},
        "e5": {"steered_pos": "stance-factual", "prompt_pos": "soft-refusal"},
    }
    mine = bc.paired(_rows(spec), bc.LABEL_POOLS["stance"],
                     a_cond="steered_pos", b_cond="prompt_pos",
                     label_col="verdict_collapsed", n_boot=200, ci=0.9, seed=0)

    # metrics.beat_rate targets ONE label, so collapse the pool to a single token
    # first — that is the only way to ask it the pooled question.
    pooled = {ex: {c: ("stance" if lbl in bc.LABEL_POOLS["stance"] else lbl)
                   for c, lbl in conds.items()} for ex, conds in spec.items()}
    results = [Result(ex, cond, "text", lbl, {})
               for ex, conds in pooled.items() for cond, lbl in conds.items()]
    theirs = metrics.beat_rate(results, target_label="stance",
                              steer_cond="steered_pos", prompt_cond="prompt_pos",
                              n_boot=200)

    assert mine["n"] == theirs["n"]
    assert mine["margin"] == pytest.approx(theirs["point"])
    assert mine["a_rate"] == pytest.approx(theirs["steer_rate"])
    assert mine["b_rate"] == pytest.approx(theirs["prompt_rate"])
    assert (mine["both"], mine["a_only"], mine["b_only"], mine["neither"]) == \
           (theirs["both"], theirs["steer_only"], theirs["prompt_only"], theirs["neither"])


def test_pooled_stance_counts_both_stance_labels():
    rows = _rows({
        "e1": {"initial": "stance-factual"},
        "e2": {"initial": "stance-evaluative"},
        "e3": {"initial": "soft-refusal"},
        "e4": {"initial": "non-engagement"},
    })
    (arm,) = bc.rate_per_arm(rows, bc.LABEL_POOLS["stance"],
                             label_col="verdict_collapsed", n_boot=200, ci=0.9, seed=0)
    assert arm["n"] == 4 and arm["rate"] == pytest.approx(0.5)


def test_judge_extraction_failure_leaves_the_denominator_not_the_numerator():
    # CLAUDE.md §3: `nonsense` is a JUDGE failure, not a behaviour. Counting it as a
    # miss would understate the rate on a denominator that includes non-data.
    rows = _rows({
        "e1": {"initial": "stance-factual"},
        "e2": {"initial": "nonsense"},
        "e3": {"initial": "soft-refusal"},
    })
    (arm,) = bc.rate_per_arm(rows, bc.LABEL_POOLS["stance"],
                             label_col="verdict_collapsed", n_boot=200, ci=0.9, seed=0)
    assert arm["n"] == 2, "the unjudgeable item must be out of the denominator"
    assert arm["rate"] == pytest.approx(0.5)
    assert arm["judge_extraction_failures"] == 1


def test_paired_drops_a_pair_when_either_arm_is_unjudgeable():
    rows = _rows({
        "e1": {"steered_pos": "stance-factual", "prompt_pos": "soft-refusal"},
        "e2": {"steered_pos": "nonsense", "prompt_pos": "stance-factual"},
    })
    res = bc.paired(rows, bc.LABEL_POOLS["stance"], a_cond="steered_pos",
                    b_cond="prompt_pos", label_col="verdict_collapsed",
                    n_boot=200, ci=0.9, seed=0)
    assert res["n"] == 1 and res["dropped_unjudgeable_pairs"] == 1


def test_ci_brackets_the_point_estimate_and_is_seed_stable():
    hits = [1.0] * 30 + [0.0] * 70
    lo, hi = bc.bootstrap_ci(hits, n_boot=2000, ci=0.90, seed=0)
    assert lo < 0.30 < hi
    assert (lo, hi) == bc.bootstrap_ci(hits, n_boot=2000, ci=0.90, seed=0)


def test_resolve_positive_accepts_a_pool_name_or_literal_labels():
    assert bc.resolve_positive("stance") == ("stance-factual", "stance-evaluative")
    assert bc.resolve_positive("soft-refusal") == ("soft-refusal",)
    assert bc.resolve_positive("a,b") == ("a", "b")


def test_missing_label_column_fails_loudly(tmp_path):
    p = tmp_path / "j.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["example_id", "condition", "other"])
        w.writeheader()
        w.writerow({"example_id": "e1", "condition": "initial", "other": "x"})
    with pytest.raises(SystemExit, match="verdict_collapsed"):
        bc.load(p, "verdict_collapsed")
