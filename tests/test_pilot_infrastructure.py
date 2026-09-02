"""Negative-path tests for the pilot's queue runner and verifier.

A verifier that has only ever passed is a verifier that has never been tested.
`notes/11` §6 states the rule in its own terms — *a control that has not been run
is a control that failed* — and the queue runner and verifier are exactly the
controls that were missing in run 1: `run()` returned 0 unconditionally, so a
killed step and a successful one were indistinguishable.

So every check below makes something go WRONG and asserts it is caught.
"""

import json
import os

import numpy as np
import pytest

from scripts.pilot import verifier
from scripts.pilot.queue import Step, run_queue


# --------------------------------------------------------------------------- #
# The queue runner
# --------------------------------------------------------------------------- #

def test_queue_records_a_real_exit_code_when_a_step_raises(tmp_path):
    """The run-1 failure, directly: a step that dies must not report success."""
    def boom():
        raise RuntimeError("the model did not load")

    steps = [Step(name="explodes", fn=boom)]
    m = run_queue(steps, out_dir=str(tmp_path))

    assert m["all_ok"] is False
    assert m["steps"][0]["status"] == "FAILED"
    assert m["steps"][0]["exit_code"] == 1
    assert "the model did not load" in m["steps"][0]["error"]


def test_queue_marks_a_step_incomplete_when_it_exits_0_without_its_outputs(tmp_path):
    """Exit code 0 is not evidence of success.

    This is the failure mode `produces` exists for: a step can complete cleanly
    and still write nothing, and run 1 had no way to tell.
    """
    steps = [Step(name="quiet_liar",
                  produces=[str(tmp_path / "never_written.json")],
                  fn=lambda: None)]
    m = run_queue(steps, out_dir=str(tmp_path))

    assert m["steps"][0]["exit_code"] == 0        # it "succeeded"
    assert m["steps"][0]["status"] == "INCOMPLETE"  # and is still a failure
    assert m["steps"][0]["missing_outputs"] == [str(tmp_path / "never_written.json")]
    assert m["all_ok"] is False


def test_queue_stops_at_the_first_failure_and_does_not_run_later_steps(tmp_path):
    ran = []
    steps = [
        Step(name="ok", fn=lambda: ran.append("ok")),
        Step(name="bad", fn=lambda: (_ for _ in ()).throw(ValueError("nope"))),
        Step(name="never", fn=lambda: ran.append("never")),
    ]
    run_queue(steps, out_dir=str(tmp_path))
    assert ran == ["ok"]
    assert steps[2].status == "PENDING"


def test_queue_writes_the_manifest_after_every_step_not_only_at_the_end(tmp_path):
    """A queue that dies mid-run must still leave an accurate record.

    Same reasoning as the 10-minute continuous sync: state that exists in exactly
    one volatile place is the root cause behind S5.
    """
    seen = []

    def peek():
        p = tmp_path / "queue_manifest.json"
        seen.append(json.load(open(p, encoding="utf-8"))["steps"][0]["status"])

    run_queue([Step(name="first", fn=lambda: None), Step(name="second", fn=peek)],
              out_dir=str(tmp_path))
    assert seen == ["OK"], "the first step's status was not on disk before the second ran"


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #

def _minimal_good_run(d):
    """The smallest artifact set the verifier should accept."""
    os.makedirs(d / "residuals", exist_ok=True)
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    np.save(d / "residuals" / "Cat__a.npy", arr)
    json.dump({"category": "Cat", "arm": "a",
               "item_ids": ["Cat:0", "Cat:1", "Cat:2"], "n_items": 3,
               "n_layers": 4, "d_model": 5, "dtype": "float32",
               "capture_site": "resid_pre, token -1"},
              open(d / "residuals" / "Cat__a.json", "w", encoding="utf-8"))
    for name in ("prompts.jsonl", "responses.jsonl"):
        open(d / name, "w", encoding="utf-8").write(json.dumps({"item_id": "Cat:0"}) + "\n")
    json.dump({"steps": [], "all_ok": True},
              open(d / "queue_manifest.json", "w", encoding="utf-8"))


def test_verifier_passes_a_well_formed_run(tmp_path):
    _minimal_good_run(tmp_path)
    c = verifier.verify(str(tmp_path))
    assert c.passed, c.failures
    assert c.checked > 0


def test_verifier_catches_residuals_that_were_never_persisted(tmp_path):
    """S5, recurring.  The single most important thing the verifier looks for."""
    _minimal_good_run(tmp_path)
    os.remove(tmp_path / "residuals" / "Cat__a.npy")
    os.remove(tmp_path / "residuals" / "Cat__a.json")
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("no residual arrays were persisted" in f for f in c.failures)


def test_verifier_catches_a_sidecar_whose_ids_do_not_match_the_array(tmp_path):
    """A silent row misalignment makes every downstream direction wrong while
    every file still parses.  Nothing else in the pipeline would notice."""
    _minimal_good_run(tmp_path)
    p = tmp_path / "residuals" / "Cat__a.json"
    meta = json.load(open(p, encoding="utf-8"))
    meta["item_ids"] = ["Cat:0", "Cat:1"]          # 2 ids, 3 rows
    json.dump(meta, open(p, "w", encoding="utf-8"))
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("sidecar lists 2 item ids, array has 3 rows" in f for f in c.failures)


def test_verifier_catches_a_missing_capture_site(tmp_path):
    """The mandatory pre-registration field (notes/11 §4, incident I-5)."""
    _minimal_good_run(tmp_path)
    p = tmp_path / "residuals" / "Cat__a.json"
    meta = json.load(open(p, encoding="utf-8"))
    del meta["capture_site"]
    json.dump(meta, open(p, "w", encoding="utf-8"))
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("capture_site" in f for f in c.failures)


def test_verifier_catches_missing_verbatim_responses(tmp_path):
    """N6's lesson: without the raw text, a finished analysis cannot be audited."""
    _minimal_good_run(tmp_path)
    os.remove(tmp_path / "responses.jsonl")
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("responses.jsonl" in f for f in c.failures)


def test_verifier_catches_an_empty_responses_file(tmp_path):
    """Present-but-empty is the harder case: the file exists, so an
    existence-only check passes it."""
    _minimal_good_run(tmp_path)
    open(tmp_path / "responses.jsonl", "w", encoding="utf-8").write("")
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("present but empty" in f for f in c.failures)


def test_verifier_catches_corrupt_json(tmp_path):
    _minimal_good_run(tmp_path)
    open(tmp_path / "report.json", "w", encoding="utf-8").write("{not json")
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("unparseable json" in f for f in c.failures)


def test_verifier_reports_a_failed_step_from_the_manifest(tmp_path):
    _minimal_good_run(tmp_path)
    json.dump({"steps": [{"name": "s1", "status": "FAILED", "exit_code": 1,
                          "produces": [], "missing_outputs": [], "error": "boom"}],
               "all_ok": False},
              open(tmp_path / "queue_manifest.json", "w", encoding="utf-8"))
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("step s1" in f for f in c.failures)


def test_verifier_fails_loudly_when_there_is_no_manifest_at_all(tmp_path):
    c = verifier.verify(str(tmp_path))
    assert not c.passed
    assert any("missing manifest" in f for f in c.failures)


# --------------------------------------------------------------------------- #
# The pairing — the defect that would have produced a cross product
# --------------------------------------------------------------------------- #

def test_scenario_key_pairs_one_to_one_where_question_index_does_not():
    """notes/19 §6.3.  `question_index` takes 25-50 values per category, so a
    literal join on it yields a cross product rather than a pairing."""
    from scripts.pilot import pairing

    rows = pairing.load_category("Sexual_orientation")
    pairs = pairing.build_pairs(rows, "Sexual_orientation")
    rep = pairing.pairing_report(rows, pairs)

    assert rep["pairs"] == 432
    assert rep["naive_question_index_join"] == 7680      # what the spec's wording gives
    assert rep["arms_balanced"]
    assert rep["rows_dropped_unpaired"] == 0
    # BBQ ships the two arms of a scenario adjacently; a useful cross-check.
    assert rep["pairs_with_consecutive_example_ids"] == rep["pairs"]


def test_item_key_is_unique_across_categories():
    """`example_id` restarts at 0 in every file, so it is unique only within one."""
    from scripts.pilot import pairing

    a = pairing.load_category("Religion")[0]
    b = pairing.load_category("Age")[0]
    assert a["example_id"] == b["example_id"] == 0
    assert pairing.item_key(a) != pairing.item_key(b)


def test_split_by_pair_keeps_both_arms_of_a_scenario_together():
    """N5 one level up: if a scenario's two arms land in different halves, the
    half-directions are estimated from different scenarios."""
    from scripts.pilot import pairing

    rows = pairing.load_category("Religion")
    pairs = pairing.build_pairs(rows, "Religion")[:40]
    A, B = pairing.split_pairs(pairs, seed=0)

    assert len(A) + len(B) == len(pairs)
    assert not ({p.pair_id for p in A} & {p.pair_id for p in B})
    for half in (A, B):
        a_rows, b_rows = pairing.arms(half)
        assert len(a_rows) == len(b_rows)          # arm balance, free
        assert all(r["context_condition"] == "ambig" for r in a_rows)
        assert all(r["context_condition"] == "disambig" for r in b_rows)


def test_polarity_contrast_is_length_matched_where_the_primary_is_not():
    """The measurement behind notes/19 §3.3 A-3, as an executable assertion."""
    from scripts.pilot import pairing

    rows = pairing.load_category("Religion")
    pol = pairing.build_pairs(rows, "Religion", contrast="polarity")
    assert pol, "no polarity pairs built"
    assert all(p.a["context"] == p.b["context"] for p in pol)
    assert all((p.a["ans0"], p.a["ans1"], p.a["ans2"])
               == (p.b["ans0"], p.b["ans1"], p.b["ans2"]) for p in pol)

    ctx = pairing.build_pairs(rows, "Religion", contrast="context")
    assert not any(p.a["context"] == p.b["context"] for p in ctx)
    # and the disambiguated arm is the ambiguous one plus an appended clause
    assert all(p.b["context"].startswith(p.a["context"]) for p in ctx)


@pytest.mark.parametrize("category,expected_pairs", [
    ("Sexual_orientation", 432),
    ("Religion", 600),
    ("Disability_status", 778),
])
def test_matched_pair_counts_match_the_planning_documents(category, expected_pairs):
    from scripts.pilot import pairing
    rows = pairing.load_category(category)
    assert len(pairing.build_pairs(rows, category)) == expected_pairs
