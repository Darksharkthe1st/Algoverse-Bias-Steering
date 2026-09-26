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


def test_load_many_tags_arms_by_run_so_two_runs_are_not_pooled(tmp_path):
    # Both runs have a `steered_pos`. Pooling them would average a treatment with its
    # control and report the mean as if it were one arm.
    paths = []
    for run, label in (("exp1", "stance-factual"), ("exp2", "soft-refusal")):
        p = tmp_path / f"{run}.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "example_id", "condition",
                                              "verdict_collapsed"])
            w.writeheader()
            w.writerow({"run": run, "example_id": "e1", "condition": "steered_pos",
                        "verdict_collapsed": label})
        paths.append(p)

    rows = bc.load_many(paths, "verdict_collapsed")
    assert sorted(r["condition"] for r in rows) == ["exp1:steered_pos", "exp2:steered_pos"]

    arms = {r["condition"]: r for r in bc.rate_per_arm(
        rows, bc.LABEL_POOLS["stance"], label_col="verdict_collapsed",
        n_boot=100, ci=0.9, seed=0)}
    assert arms["exp1:steered_pos"]["rate"] == 1.0
    assert arms["exp2:steered_pos"]["rate"] == 0.0


def test_load_many_leaves_a_single_run_untagged(tmp_path):
    p = tmp_path / "one.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "example_id", "condition",
                                          "verdict_collapsed"])
        w.writeheader()
        w.writerow({"run": "exp1", "example_id": "e1", "condition": "initial",
                    "verdict_collapsed": "soft-refusal"})
    rows = bc.load_many([p], "verdict_collapsed")
    assert rows[0]["condition"] == "initial"


# --- the coherence filter -----------------------------------------------------
# Additive steering degrades text before it stops working, so a judge can read a
# confident stance out of a repetition loop. These pin the two rules that keep that
# from inflating a rate.

def _coh(flags):
    """{(example_id, condition): coherent} -> the (run, example_id, condition) keys
    `load_coherence` produces, with run=None as `_rows` leaves it."""
    return {(None, ex, cond): v for (ex, cond), v in flags.items()}


def test_coherence_filter_drops_failing_completions_and_reports_the_count():
    rows = _rows({"e1": {"initial": "stance-factual"},
                  "e2": {"initial": "stance-factual"},
                  "e3": {"initial": "soft-refusal"}})
    coherence = _coh({("e1", "initial"): 1, ("e2", "initial"): 0,
                      ("e3", "initial"): 1})
    kept, report = bc.apply_coherence_filter(rows, coherence, tagged=False)
    assert [r["example_id"] for r in kept] == ["e1", "e3"]
    assert report["dropped_incoherent"] == 1 and report["unmatched"] == 0


def test_a_completion_with_no_coherence_row_is_excluded_and_counted():
    # Silently keeping it would report a rate over data the gate never saw; silently
    # dropping it would hide that the two inputs disagree. So: excluded AND counted.
    rows = _rows({"e1": {"initial": "stance-factual"}, "e2": {"initial": "stance-factual"}})
    kept, report = bc.apply_coherence_filter(rows, _coh({("e1", "initial"): 1}),
                                             tagged=False)
    assert [r["example_id"] for r in kept] == ["e1"]
    assert report["unmatched"] == 1


def test_paired_filter_drops_the_item_unless_both_arms_are_coherent():
    # Per-row filtering could keep steered_pos and drop prompt_pos for the same item,
    # which would silently unpair it. The pair is the unit.
    rows = _rows({
        "e1": {"steered_pos": "stance-factual", "prompt_pos": "soft-refusal"},
        "e2": {"steered_pos": "stance-factual", "prompt_pos": "soft-refusal"},
    })
    coherence = _coh({
        ("e1", "steered_pos"): 1, ("e1", "prompt_pos"): 1,
        ("e2", "steered_pos"): 1, ("e2", "prompt_pos"): 0,   # partner broken
    })
    kept = bc.restrict_to_pairs_coherent_in_both(
        rows, coherence, a_cond="steered_pos", b_cond="prompt_pos", tagged=False)
    assert {r["example_id"] for r in kept} == {"e1"}


def test_coherence_lookup_recovers_the_untagged_condition_for_multi_run_rows():
    # load_many rewrites conditions to <run>:<condition>, but coherence.csv stores the
    # bare condition — without recovering it every lookup would miss.
    rows = [{"run": "expA", "example_id": "e1", "condition": "expA:steered_pos",
             "verdict_collapsed": "stance-factual"}]
    coherence = {("expA", "e1", "steered_pos"): 1}
    kept, report = bc.apply_coherence_filter(rows, coherence, tagged=True)
    assert len(kept) == 1 and report["unmatched"] == 0


def test_load_coherence_rejects_a_csv_that_is_not_a_coherence_csv(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("run,example_id,condition\nr,e1,initial\n")
    with pytest.raises(SystemExit, match="coherent"):
        bc.load_coherence([p])
