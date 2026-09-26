"""Read back the per-arm eval logs `logs.py` writes — the raw completions.

`logs.py` is the writer, this is the reader; they are the only two modules that
know the `logs/by_condition/<cond>.txt` format. It exists because `results.csv`
stores a *verdict* per (example, arm) and not the text, so anything that needs the
completion itself — re-judging under a new judge version, the coherence gate —
has to come back through the text log. CLAUDE.md §3 makes that the right source
anyway: recount from text logs, never from the pickles.

The logs contain model-generated prose, so delimiters are honoured only when a
whole line matches and the block that follows has the expected shape. An essay
that happens to contain a line like `=== x ===` would otherwise silently split one
record into two, which is exactly the class of bug that makes a recount disagree
with itself. Records are data, never instructions; a `VERDICT:` line read back
here is the *previous* judge's claim, not a label to trust.

    from src.bias_steer import runlogs
    for rec in runlogs.read_run("runs/<id>"):
        rec.example_id, rec.condition, rec.prompt, rec.response, rec.logged_verdict
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A record header, and the two labelled lines that must follow it. Anchored whole-line
# (re.match on a stripped line + fullmatch) so prose cannot forge one.
_HEADER_RE = re.compile(r"^=== (?P<id>.+) ===$")
_PROMPT_PREFIX = "PROMPT:  "
_VERDICT_PREFIX = "VERDICT: "
_OUTPUT_PREFIX = "OUTPUT:  "
# `logs._degenerate_flag` may append this to the VERDICT line. It is a readability
# hint for humans, not part of the verdict, so it is stripped and surfaced separately.
_DEGENERATE_FLAG = "[!! repeat-loop]"


class RunLogFormatError(RuntimeError):
    """A by_condition log did not match the format `logs.py` writes."""


@dataclass(frozen=True)
class EvalRecord:
    """One (example, arm) completion as the run actually logged it."""

    example_id: str
    condition: str
    prompt: str
    response: str
    logged_verdict: str
    degenerate_flagged: bool
    source: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.example_id, self.condition)


def _blocks(text: str):
    """Split a log into (header_id, body_lines), honouring only real headers.

    A header is a whole line `=== <id> ===` whose NEXT line starts with the
    `PROMPT:` prefix. Both conditions must hold: the first alone is forgeable by
    prose, and the second alone would miss the header altogether.
    """
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m and i + 1 < len(lines) and lines[i + 1].startswith(_PROMPT_PREFIX):
            starts.append((i, m.group("id")))
    for n, (i, ex_id) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        yield ex_id, lines[i + 1:end]


def parse_condition_log(path, condition: str | None = None) -> list[EvalRecord]:
    """Parse one `logs/by_condition/<cond>.txt` into records.

    `condition` defaults to the file's stem, which is how `logs.py` names it.
    """
    path = Path(path)
    condition = condition or path.stem
    out: list[EvalRecord] = []

    for ex_id, body in _blocks(path.read_text(encoding="utf-8")):
        if not body or not body[0].startswith(_PROMPT_PREFIX):
            raise RunLogFormatError(f"{path}: record {ex_id!r} has no PROMPT line")
        prompt = body[0][len(_PROMPT_PREFIX):]

        # PROMPT may itself be multi-line (IssueBench templates contain newlines);
        # everything up to the VERDICT line belongs to it.
        j = 1
        while j < len(body) and not body[j].startswith(_VERDICT_PREFIX):
            prompt += "\n" + body[j]
            j += 1
        if j >= len(body):
            raise RunLogFormatError(f"{path}: record {ex_id!r} has no VERDICT line")

        verdict = body[j][len(_VERDICT_PREFIX):]
        flagged = _DEGENERATE_FLAG in verdict
        verdict = verdict.replace(_DEGENERATE_FLAG, "").strip()

        j += 1
        if j >= len(body) or not body[j].startswith(_OUTPUT_PREFIX):
            raise RunLogFormatError(f"{path}: record {ex_id!r} has no OUTPUT line")
        # The response runs to the end of the block. `logs.py` separates records with
        # a blank line, so drop only trailing blanks — blank lines INSIDE an essay
        # are content and must survive.
        resp_lines = [body[j][len(_OUTPUT_PREFIX):]] + body[j + 1:]
        while resp_lines and not resp_lines[-1].strip():
            resp_lines.pop()

        out.append(EvalRecord(
            example_id=ex_id, condition=condition, prompt=prompt,
            response="\n".join(resp_lines), logged_verdict=verdict,
            degenerate_flagged=flagged, source=path,
        ))
    return out


def read_run(run_dir, conditions=None) -> list[EvalRecord]:
    """Every logged completion for a run, across its arms.

    `conditions` restricts which arms are read (default: every
    `logs/by_condition/*.txt` present, sorted for a deterministic order).
    """
    by_dir = Path(run_dir) / "logs" / "by_condition"
    if not by_dir.is_dir():
        raise RunLogFormatError(
            f"{run_dir}: no logs/by_condition/ — raw completions were not persisted, "
            f"so nothing here can be re-judged or coherence-scored."
        )
    files = sorted(by_dir.glob("*.txt"))
    if conditions is not None:
        wanted = set(conditions)
        files = [f for f in files if f.stem in wanted]
    records: list[EvalRecord] = []
    for f in files:
        records.extend(parse_condition_log(f))
    if not records:
        raise RunLogFormatError(f"{by_dir}: no parseable records")
    return records


def assert_matches_results(records, results_csv) -> dict:
    """Cross-check the parsed logs against `results.csv`; return a small report.

    The two are written by the same loop, so they must agree on the set of
    (example, arm) pairs. Disagreement means the parse dropped or invented records
    — a silent recount error — so this fails loudly. It also reports how many
    logged verdicts differ from the CSV's (expected: zero).
    """
    import csv as _csv

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    csv_keys = {(r["example_id"], r["condition"]): r["verdict"] for r in rows}
    log_keys = {r.key: r.logged_verdict for r in records}

    missing = sorted(csv_keys.keys() - log_keys.keys())
    extra = sorted(log_keys.keys() - csv_keys.keys())
    if missing or extra:
        raise RunLogFormatError(
            f"log/results mismatch for {results_csv}: {len(missing)} pair(s) in "
            f"results.csv but not in the logs (e.g. {missing[:3]}), {len(extra)} in "
            f"the logs but not results.csv (e.g. {extra[:3]})."
        )
    disagree = [k for k, v in log_keys.items() if csv_keys[k] != v]
    return {"n": len(log_keys), "verdict_disagreements": len(disagree),
            "examples": sorted({k[0] for k in log_keys}).__len__(),
            "conditions": sorted({k[1] for k in log_keys})}
