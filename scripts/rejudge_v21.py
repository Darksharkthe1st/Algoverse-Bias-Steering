"""Re-judge a run's persisted completions under pinned judge v2.1.

Lane B item 1 (`docs/HANDOFF_lane_a.md`), and the source of every *reportable*
number in Lane A: so that one judge version scores every headline count. CLAUDE.md
§4 — a judged number carries its judge version, and two versions never share a
table. Judge v1 (the 2025 binary rubric that scored factual decisiveness as
opinionation) is retired; the runs' inline `neutrality` verdicts are kept only as a
comparator, never mixed in.

Why offline rather than in the pipeline: the eval loop's contrast machinery is built
on a 2-label judge (`experiment._contrast` takes `judge.labels[1]` as the positive
pole, which with v2.1's 9 labels would silently become `incoherent`). So the run
generates with the binary judge inline, and the fine-grained rubric is applied here,
to the completions the run persisted.

    python scripts/rejudge_v21.py runs/<id>
    python scripts/rejudge_v21.py runs/<id> --limit 20      # smoke-test first

Writes `runs/<id>/judged_v2.1.csv` (one row per completion) and
`runs/<id>/judged_v2.1.json` (the judge's provenance: model, seed, temperature, and
the SHA-256 of the exact rubric string that produced these labels — the "judge
version" a table has to cite).

Both views are persisted per `docs/judges/judge_v2.1.md`: the raw 9-way
`verdict_v21`, and `verdict_collapsed` where the four non-behavioural labels fold to
`ignored`. `UNMATCHED` ("nonsense" — the judge's reply named no label) is an
extraction failure on the JUDGE's side, and passes through unchanged into both
columns: folding it into `ignored`, or into any behaviour class, is exactly the
error CLAUDE.md §3 forbids. Count them; never bury them.

Needs OPENAI_API_KEY. No GPU.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.bias_steer import runlogs  # noqa: E402
from src.bias_steer.judges import UNMATCHED, collapse, judge_v2_spec, neutrality_judge  # noqa: E402
from src.bias_steer.schema import Example  # noqa: E402
from src.bias_steer.tracking import git_sha  # noqa: E402


def rubric_sha256(spec) -> str:
    """The judge version's fingerprint: a hash of the exact rubric string used.

    A rubric edit (e.g. the 2026-09-26 identity-decline -> non-engagement patch) MUST
    change the labels it produces, so numbers from before and after are not the same
    judge. Naming "v2.1" is not enough to establish that two tables agree; this is.
    """
    return hashlib.sha256(spec.rubric.encode("utf-8")).hexdigest()


def rejudge(run_dir: Path, *, model: str, seed: int, limit: int | None = None,
            conditions=None) -> tuple[list[dict], dict]:
    records = runlogs.read_run(run_dir, conditions=conditions)
    records.sort(key=lambda r: (r.condition, r.example_id))
    if limit:
        records = records[:limit]

    spec = judge_v2_spec(model=model, seed=seed)
    examples = [Example(id=r.example_id, prompt=r.prompt, metadata={}) for r in records]
    responses = [r.response for r in records]

    print(f"judging {len(responses)} completions under v2.1 ({model}, seed={seed}) ...")
    verdicts = neutrality_judge(responses, examples, spec)

    rows = [{
        "run": run_dir.name,
        "example_id": rec.example_id,
        "condition": rec.condition,
        "verdict_v21": v,
        "verdict_collapsed": collapse(v),
        # The inline binary judge's label, side by side but never merged: it is a
        # DIFFERENT judge version and exists here only as a comparator.
        "verdict_inline_binary": rec.logged_verdict,
        "n_words": len(rec.response.split()),
    } for rec, v in zip(records, verdicts)]

    provenance = {
        "judge_version": "v2.1",
        "judge_rubric_sha256": rubric_sha256(spec),
        "judge_rubric_owner": "src/bias_steer/judges/v2.py::V21_RUBRIC",
        "judge_spec_doc": "docs/judges/judge_v2.1.md",
        "rubric_amendment": "2026-09-26 capability/identity declines -> non-engagement",
        "model": spec.model, "seed": spec.seed, "temperature": spec.temperature,
        "labels": list(spec.labels),
        "n_judged": len(rows),
        "run": run_dir.name,
        "git_sha": git_sha(_REPO)[0], "git_dirty": git_sha(_REPO)[1],
        "counts_v21": dict(Counter(r["verdict_v21"] for r in rows)),
        "counts_collapsed": dict(Counter(r["verdict_collapsed"] for r in rows)),
        # Judge-side extraction failures, reported as their own number so they can
        # never hide inside `ignored` or a behaviour class.
        "n_judge_extraction_failures": sum(1 for r in rows if r["verdict_v21"] == UNMATCHED),
    }
    return rows, provenance


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--model", default="gpt-4o-mini", help="judge model (pinned)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N completions (smoke test)")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated arms to judge (default: all)")
    ap.add_argument("--out-name", default="judged_v2.1.csv")
    a = ap.parse_args(argv)

    conditions = a.conditions.split(",") if a.conditions else None

    for run in a.runs:
        run = run if run.is_absolute() else _REPO / run
        rows, prov = rejudge(run, model=a.model, seed=a.seed, limit=a.limit,
                             conditions=conditions)

        out = run / a.out_name
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (run / a.out_name.replace(".csv", ".json")).write_text(
            json.dumps(prov, indent=2) + "\n")
        print(f"wrote {out}")
        print(f"  rubric sha256 {prov['judge_rubric_sha256'][:16]}…  "
              f"judge extraction failures: {prov['n_judge_extraction_failures']}")
        for label, n in sorted(prov["counts_v21"].items(), key=lambda kv: -kv[1]):
            print(f"    {label:20s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
