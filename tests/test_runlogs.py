"""Reading back the raw completions from `logs/by_condition/<cond>.txt`.

The point of these tests is the adversarial case: the logs hold model prose, so a
completion can contain anything the format uses as a delimiter. A parser that
splits on the first `=== ... ===` it sees will silently turn one record into two,
and a silent miscount is the failure mode CLAUDE.md §3 exists to prevent — so the
forged-delimiter tests are the load-bearing ones, not the happy path.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.bias_steer import runlogs  # noqa: E402
from src.bias_steer.logs import RunLogger  # noqa: E402
from src.bias_steer.schema import Example, Result  # noqa: E402


def _write(tmp_path, cond, body):
    d = tmp_path / "logs" / "by_condition"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{cond}.txt"
    p.write_text(body, encoding="utf-8")
    return p


def test_parses_a_record_written_by_the_logger():
    # The writer and the reader must agree; exercise them against each other rather
    # than against a hand-typed string that could drift from logs.py.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        log = RunLogger(run)
        ex = Example(id="ex-1", prompt="Is X good?", metadata={})
        log.eval(ex, [Result("ex-1", "initial", "Yes, clearly.", "opinionated", {})])

        recs = runlogs.read_run(run)
        assert len(recs) == 1
        r = recs[0]
        assert (r.example_id, r.condition) == ("ex-1", "initial")
        assert r.prompt == "Is X good?"
        assert r.response == "Yes, clearly."
        assert r.logged_verdict == "opinionated"
        assert r.degenerate_flagged is False


def test_multiline_response_survives_including_blank_lines(tmp_path):
    # Essays have paragraph breaks. A blank line is content, not a record separator;
    # only the trailing separator blank may be dropped.
    body = (
        "=== ex-1 ===\n"
        "PROMPT:  Write an essay.\n"
        "VERDICT: opinionated\n"
        "OUTPUT:  Para one.\n"
        "\n"
        "Para two.\n"
        "\n"
        "=== ex-2 ===\n"
        "PROMPT:  Another.\n"
        "VERDICT: neutral\n"
        "OUTPUT:  Short.\n"
        "\n"
    )
    _write(tmp_path, "initial", body)
    recs = runlogs.read_run(tmp_path)
    assert [r.example_id for r in recs] == ["ex-1", "ex-2"]
    assert recs[0].response == "Para one.\n\nPara two."
    assert recs[1].response == "Short."


def test_a_forged_header_inside_a_response_does_not_split_the_record(tmp_path):
    # The adversarial case: the model writes a line that looks exactly like a record
    # header. It is not followed by a PROMPT line, so it stays part of the response.
    body = (
        "=== ex-1 ===\n"
        "PROMPT:  Explain the log format.\n"
        "VERDICT: opinionated\n"
        "OUTPUT:  Records look like this:\n"
        "=== ex-999 ===\n"
        "and that is all.\n"
        "\n"
    )
    _write(tmp_path, "steered_pos", body)
    recs = runlogs.read_run(tmp_path)
    assert len(recs) == 1, "a forged header split one record into two"
    assert "=== ex-999 ===" in recs[0].response


def test_a_forged_header_followed_by_a_prompt_line_is_reported_not_guessed(tmp_path):
    # Pathological: prose forges BOTH the header and a following PROMPT line. That is
    # indistinguishable from a real record by format alone, so the parser takes it as
    # one -- but the resulting block has no OUTPUT line, so it fails loudly instead of
    # returning a truncated response.
    body = (
        "=== ex-1 ===\n"
        "PROMPT:  Q.\n"
        "VERDICT: neutral\n"
        "OUTPUT:  Here is a sample log:\n"
        "=== ex-2 ===\n"
        "PROMPT:  a quoted prompt\n"
        "\n"
    )
    _write(tmp_path, "initial", body)
    with pytest.raises(runlogs.RunLogFormatError):
        runlogs.read_run(tmp_path)


def test_degenerate_flag_is_stripped_from_the_verdict(tmp_path):
    # logs.py may append a human-readability hint to the VERDICT line. It is not part
    # of the label, and folding it in would corrupt every verdict comparison.
    body = (
        "=== ex-1 ===\n"
        "PROMPT:  Q.\n"
        "VERDICT: opinionated  [!! repeat-loop]\n"
        "OUTPUT:  yes yes yes yes\n"
        "\n"
    )
    _write(tmp_path, "steered_pos", body)
    r = runlogs.read_run(tmp_path)[0]
    assert r.logged_verdict == "opinionated"
    assert r.degenerate_flagged is True


def test_multiline_prompt_is_kept_whole(tmp_path):
    # IssueBench templates contain newlines ("Take a state of researcher\n\nHere is
    # the general Topic:..."), so the prompt runs until the VERDICT line.
    body = (
        "=== ex-1 ===\n"
        "PROMPT:  line one\n"
        "line two\n"
        "VERDICT: neutral\n"
        "OUTPUT:  ok\n"
        "\n"
    )
    _write(tmp_path, "initial", body)
    r = runlogs.read_run(tmp_path)[0]
    assert r.prompt == "line one\nline two"
    assert r.response == "ok"


def test_read_run_can_restrict_to_named_arms(tmp_path):
    for cond in ("initial", "steered_pos", "prompt_pos"):
        _write(tmp_path, cond,
               f"=== ex-1 ===\nPROMPT:  Q.\nVERDICT: neutral\nOUTPUT:  {cond}\n\n")
    recs = runlogs.read_run(tmp_path, conditions=["initial", "prompt_pos"])
    assert sorted(r.condition for r in recs) == ["initial", "prompt_pos"]


def test_missing_by_condition_dir_is_an_explicit_error(tmp_path):
    with pytest.raises(runlogs.RunLogFormatError, match="raw completions"):
        runlogs.read_run(tmp_path)


def test_assert_matches_results_accepts_an_agreeing_pair(tmp_path):
    _write(tmp_path, "initial",
           "=== ex-1 ===\nPROMPT:  Q.\nVERDICT: neutral\nOUTPUT:  a\n\n"
           "=== ex-2 ===\nPROMPT:  Q2.\nVERDICT: opinionated\nOUTPUT:  b\n\n")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "example_id,condition,verdict\nex-1,initial,neutral\nex-2,initial,opinionated\n")
    rep = runlogs.assert_matches_results(runlogs.read_run(tmp_path), csv_path)
    assert rep["n"] == 2 and rep["verdict_disagreements"] == 0


def test_assert_matches_results_rejects_a_dropped_record(tmp_path):
    # If the parse silently lost a record, every rate computed from it would be wrong
    # on a denominator nobody checked.
    _write(tmp_path, "initial", "=== ex-1 ===\nPROMPT:  Q.\nVERDICT: neutral\nOUTPUT:  a\n\n")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "example_id,condition,verdict\nex-1,initial,neutral\nex-2,initial,neutral\n")
    with pytest.raises(runlogs.RunLogFormatError, match="mismatch"):
        runlogs.assert_matches_results(runlogs.read_run(tmp_path), csv_path)
