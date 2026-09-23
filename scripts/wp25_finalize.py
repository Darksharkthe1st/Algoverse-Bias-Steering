#!/usr/bin/env python3
"""Merge the WP-25 annotator sheets, validate, and write the adjudication notes.

Run it the moment the filled sheets come back:

    py scripts/wp25_finalize.py --dir wp25

It discovers every ``sheet_*.csv`` in that directory, checks each one against
the master, merges the labels, applies the frozen merge rule (majority wins,
ties resolve to S3), applies the DV-exclusion list, computes the agreement
statistics, and writes ``adjudication_notes.md``.

Nothing is written unless every check passes. To see where things stand while
sheets are still trickling in:

    py scripts/wp25_finalize.py --dir wp25 --check

which validates and reports progress without touching a file.

The merge rule and the S1 regex both live in docs/PREREG.md §3.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kappa_from_csv import report_many  # noqa: E402

S1_VERBS = re.compile(r"do you prefer|do you like more|do you enjoy more", re.IGNORECASE)
VALID = {"S2", "S3"}
BASE_COLS = ["item_id", "item_text", "s1_auto", "unnameable_candidate"]
TAIL_COLS = ["majority_stratum", "exposed", "dv_exclude"]


class Fail(Exception):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_master(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise Fail(f"{path.name} is empty")
    for r in rows:
        if not (r.get("item_id") or "").strip():
            raise Fail(f"{path.name} has a row with no item_id")
    ids = [r["item_id"].strip() for r in rows]
    if len(set(ids)) != len(ids):
        dupe = [i for i, n in Counter(ids).items() if n > 1]
        raise Fail(f"{path.name} has duplicate item_id: {dupe[:5]}")
    return rows


def find_sheets(d: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(d.glob("sheet_*.csv")):
        out[re.sub(r"^sheet_", "", p.stem)] = p
    return out


def label_col(rows: list[dict[str, str]], path: Path) -> str:
    names = list(rows[0].keys())
    for c in ("stratum_S2_or_S3", "stratum", "label"):
        if c in names:
            return c
    raise Fail(f"{path.name}: no label column found in {names}")


def validate_sheet(
    name: str, path: Path, master_text: dict[str, str], adjudicated: set[str]
) -> tuple[dict[str, str], list[str]]:
    """Returns (item_id -> label) and a list of problems."""
    rows = read_csv(path)
    if not rows:
        raise Fail(f"{path.name} is empty")
    col = label_col(rows, path)
    problems: list[str] = []
    labels: dict[str, str] = {}
    seen: list[str] = []

    for i, r in enumerate(rows, start=2):
        iid = (r.get("item_id") or "").strip()
        text = (r.get("item_text") or "").strip()
        raw = (r.get(col) or "").strip()
        if not iid:
            problems.append(f"row {i}: blank item_id")
            continue
        seen.append(iid)
        if iid not in master_text:
            problems.append(f"row {i}: {iid} is not in the master")
            continue
        if text and text != master_text[iid]:
            problems.append(
                f"row {i}: {iid} text does not match master "
                f"(sheet={text[:40]!r} master={master_text[iid][:40]!r})"
            )
        if not raw:
            continue
        lab = raw.upper().replace(" ", "")
        if lab in {"2", "S2"}:
            lab = "S2"
        elif lab in {"3", "S3"}:
            lab = "S3"
        else:
            problems.append(f"row {i}: {iid} has label {raw!r}, expected S2 or S3")
            continue
        labels[iid] = lab

    dupes = [i for i, n in Counter(seen).items() if n > 1]
    if dupes:
        problems.append(f"duplicate item_id in sheet: {dupes[:5]}")
    missing = adjudicated - set(seen)
    if missing:
        problems.append(f"{len(missing)} adjudicated items absent from sheet, e.g. {sorted(missing)[:3]}")
    extra = set(seen) - adjudicated
    if extra:
        problems.append(f"{len(extra)} items present that should not be labelled, e.g. {sorted(extra)[:3]}")
    return labels, problems


def majority(votes: list[str]) -> str:
    """Majority wins; ties resolve to S3 (frozen, docs/PREREG.md §3)."""
    if not votes:
        return ""
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(l for l, c in counts.items() if c == top)
    if len(winners) == 1:
        return winners[0]
    return "S3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("wp25"))
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    d = args.dir
    master_path = d / "battery_stratified_MASTER.csv"
    if not master_path.exists():
        raise Fail(f"no master at {master_path}")

    master = load_master(master_path)
    master_text = {r["item_id"].strip(): (r.get("item_text") or "").strip() for r in master}

    s1_ids = {i for i, t in master_text.items() if S1_VERBS.search(t)}
    adjudicated = set(master_text) - s1_ids

    excl_path = d / "dv_exclusions.csv"
    excluded = set()
    if excl_path.exists():
        excluded = {r["item_id"].strip() for r in read_csv(excl_path)}

    print("=" * 64)
    print("WP-25 finalize")
    print("=" * 64)
    print(f"master              : {len(master)} rows")
    print(f"S1 (auto-excluded)  : {len(s1_ids)}")
    print(f"to adjudicate       : {len(adjudicated)}")
    print(f"DV exclusions       : {len(excluded)}"
          + ("" if excl_path.exists() else "   [dv_exclusions.csv not found]"))

    if len(s1_ids) + len(adjudicated) != len(master):
        raise Fail("reconciliation failed: S1 + adjudicated != master rows")
    bad_excl = excluded - set(master_text)
    if bad_excl:
        raise Fail(f"dv_exclusions references unknown item_id: {sorted(bad_excl)[:5]}")
    overlap = excluded & s1_ids
    if overlap:
        raise Fail(f"dv_exclusions contains S1 items (should never happen): {sorted(overlap)}")

    sheets = find_sheets(d)
    if not sheets:
        raise Fail(f"no sheet_*.csv found in {d}")

    print()
    print(f"sheets found        : {len(sheets)}  ({', '.join(sorted(sheets))})")
    print("-" * 64)

    raters: dict[str, dict[str, str]] = {}
    all_problems: dict[str, list[str]] = {}
    for name, path in sheets.items():
        labels, problems = validate_sheet(name, path, master_text, adjudicated)
        raters[name] = labels
        all_problems[name] = problems
        done = len(labels)
        if done == len(adjudicated):
            status, extra = "COMPLETE", f"  {done}/{len(adjudicated)}"
        elif done == 0:
            status, extra = "not started", "  ignored"
        else:
            status, extra = "IN PROGRESS", f"  {done}/{len(adjudicated)}"
        flag = "  <-- PROBLEMS" if problems else ""
        print(f"  {name:16s} {status:>12s}{extra}{flag}")
        for p in problems:
            print(f"      ! {p}")

    blocking = {n: p for n, p in all_problems.items() if p}
    if blocking:
        print()
        print("STOPPED — fix the problems above and rerun. Nothing was written.")
        return 1

    # A sheet that is started but unfinished means somebody is mid-task. Merging
    # around it would produce a result that looks final but silently drops them.
    partial = {n: len(l) for n, l in raters.items() if 0 < len(l) < len(adjudicated)}
    if partial:
        print()
        for n, got in sorted(partial.items()):
            print(f"  ! {n} is partly filled ({got}/{len(adjudicated)}) — still in progress")
        print()
        print("STOPPED — a partly filled sheet means someone is still labelling.")
        print("Wait for them, or empty that sheet's label column if they have")
        print("dropped out. Nothing was written. Use --check for progress only.")
        return 1

    complete = {n: l for n, l in raters.items() if len(l) == len(adjudicated)}
    print()
    if len(complete) < 2:
        print(f"{len(complete)} of {len(sheets)} sheets complete — need at least 2 to merge.")
        print("Nothing written. Rerun when more come back.")
        return 0
    if len(complete) < 3:
        print(f"WARNING: only {len(complete)} complete sheets. PREREG §3 expects 3.")
        print("         Merging anyway; the annotator count is recorded in the notes.")

    if args.check:
        print("--check given: validation passed, nothing written.")
        return 0

    # ---- merge -----------------------------------------------------------
    names = sorted(complete)
    for r in master:
        iid = r["item_id"].strip()
        r["s1_auto"] = "S1" if iid in s1_ids else ""
        for n in names:
            r[f"label_{n}"] = complete[n].get(iid, "")
        votes = [complete[n][iid] for n in names if iid in complete[n]]
        r["majority_stratum"] = "S1" if iid in s1_ids else majority(votes)
        r["unnameable_candidate"] = "Y" if iid in excluded else ""
        r["dv_exclude"] = "Y" if iid in excluded else ""
        r.setdefault("exposed", "")

    cols = BASE_COLS + [f"label_{n}" for n in names] + TAIL_COLS
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(master)
    master_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {master_path}")

    strata = Counter(r["majority_stratum"] for r in master)
    n_s2, n_s3 = strata.get("S2", 0), strata.get("S3", 0)
    if n_s2 + n_s3 + strata.get("S1", 0) != len(master):
        raise Fail("post-merge reconciliation failed: strata do not sum to master rows")

    s2_primary = sum(
        1 for r in master if r["majority_stratum"] == "S2" and r["dv_exclude"] != "Y"
    )

    print()
    print("-" * 64)
    print(f"S1 {strata.get('S1', 0):4d}   S2 {n_s2:4d}   S3 {n_s3:4d}   total {len(master)}")
    print(f"S2 in the primary DV after exclusions: {s2_primary}")
    print("-" * 64)
    print()

    # ---- agreement -------------------------------------------------------
    out = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = out
    try:
        stats = report_many(complete, args.boot, args.seed)
    finally:
        sys.stdout = real_stdout
    agreement_text = out.getvalue()
    print(agreement_text)

    # ---- notes -----------------------------------------------------------
    today = datetime.date.today().isoformat()
    notes = f"""# WP-25 adjudication notes

Generated by `scripts/wp25_finalize.py` on {today}. Rerun to reproduce:

    py scripts/wp25_finalize.py --dir {d}

## Counts

| stage | n |
|---|---|
| source rows (`datasets/GPT_Prompts/comparison_questions_200.csv`) | 296 |
| exact duplicates removed | 3 |
| unique items | {len(master)} |
| S1, auto-excluded by regex | {strata.get('S1', 0)} |
| hand-adjudicated | {len(adjudicated)} |
| **S2** | **{n_s2}** |
| **S3** | **{n_s3}** |
| DV exclusions (no nameable side B) | {len(excluded)} |
| **S2 entering the primary DV** | **{s2_primary}** |

## Method

Annotators: {', '.join(names)} ({len(names)}), labelling independently and
blind to model output. Each labelled all {len(adjudicated)} non-S1 items S2 or S3
against the definitions in `docs/PREREG.md` §3.

S1 is deterministic, not hand-labelled: the regex
`do you prefer|do you like more|do you enjoy more` matched {len(s1_ids)} items,
all excluded from analysis. Regex and match list are committed alongside.

Merge rule, frozen: majority wins, ties resolve to S3 — conservative, since S3
is the stratum where hedging is the correct behaviour.

DV exclusions are derived deterministically by `scripts/wp25_dv_exclusions.py`,
not hand-picked; see `dv_exclusions_review.txt` for the rule and the borderline
cases it deliberately leaves in.

## Agreement

```
{agreement_text.strip()}
```

Fleiss κ and Gwet's AC1 are both reported. κ is depressed when one category
dominates the marginals, which it does here; AC1 is robust to that skew. The
disagreement rate is a number this paper carries, not a gate it has to pass.

## Known limitations

- **Annotator pool.** The pool named in `docs/HANDOFF_WP25.md` (Farhan,
  Aryaman, Jeremiah) was not available: Aryaman left the team and Farhan was
  assigned elsewhere. The construct is explicitly "a competent adult would
  agree one alternative is correct", so labellers outside the research team are
  appropriate to it, but the substitution is recorded here rather than implied.
- **Exposure flag.** `HANDOFF_WP25.md` asks for an `exposed` flag marking items
  that overlap the ~30-item calibration sheet. That sheet is not in the
  repository, and it carries no item ids that map onto this battery, so the
  flag could not be computed. The column is present and empty.
- **The "18".** `RESEARCH_CONTRACT.md` states 18 DV exclusions. No list or
  script in the repository reproduces that number. It has been replaced with a
  stated rule yielding {len(excluded)}, derived reproducibly. Cite {len(excluded)}.
- **Near-duplicates.** Only exact duplicates were removed. Semantic
  near-duplicates remain (two seatbelt items, two helmet items, two
  fire-extinguisher items, two shower items, two ignorance items).
"""
    notes_path = d / "adjudication_notes.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"wrote {notes_path}")
    print()
    if len(names) >= 3:
        print("WP-25 complete. Commit wp25/ and open the PR — this unblocks WP-04.")
    else:
        print(f"Merged with {len(names)} annotators, not 3. Usable, but the")
        print("annotator count is a limitation the paper has to state. Commit")
        print("wp25/ and open the PR — this unblocks WP-04.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        raise SystemExit(2)
