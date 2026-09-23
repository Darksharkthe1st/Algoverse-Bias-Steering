#!/usr/bin/env python3
"""Derive the WP-25 DV-exclusion set deterministically.

Some battery items pit a thing against its own absence — *"a seatbelt or no
seatbelt"*. The primary DV (``named_a_side``) asks whether a response committed
to one of the two named alternatives, but side B here has no nameable surface
form, so commitment to it can never be extracted. Those items are dropped from
the primary DV.

``RESEARCH_CONTRACT.md`` states this set has 18 members. That number appears
nowhere else in the repository: no list, no script, and no artifact reproduces
it. This script replaces the unreproducible count with a stated rule, its
output, and a review file for the cases the rule deliberately leaves in.

THE RULE (strict). Split ``Which <PROP>: <A> or <B>?`` on its final " or ".
Exclude the item when <B> opens with a negator — no / none / not / nothing /
neither / without / no one. The test is whether the extractor has any string to
match on side B, so a side B that names a *thing* stays in even when that thing
is an absence in spirit ("ignorance" is matchable; "no seatbelt" is not).

S1 items are never in the set: they are excluded from all analysis already.

  py scripts/wp25_dv_exclusions.py --master wp25/battery_stratified_MASTER.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and cannot encode the em dashes used here.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# side B opens with one of these -> no nameable surface form
NEGATOR = re.compile(
    r"^(?:no|none|not|nothing|neither|without|no\s+one)\b",
    re.IGNORECASE,
)

# side B names a thing, but that thing denotes an absence or omission.
# Deliberately NOT excluded — kept here so the judgement call is auditable.
BORDERLINE = re.compile(
    r"^(?:ignorance|ignoring|skipping|avoiding|going\s+without|never)\b",
    re.IGNORECASE,
)

S1_VERBS = re.compile(r"do you prefer|do you like more|do you enjoy more", re.IGNORECASE)


def split_alternatives(text: str) -> tuple[str, str] | None:
    """'Which is safer: a seatbelt or no seatbelt?' -> ('a seatbelt', 'no seatbelt')."""
    if ":" not in text:
        return None
    body = text.split(":", 1)[1].strip().rstrip("?").strip()
    idx = body.lower().rfind(" or ")
    if idx < 0:
        return None
    return body[:idx].strip(), body[idx + 4 :].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <master dir>/dv_exclusions.csv")
    ap.add_argument("--review", type=Path, default=None, help="default: <master dir>/dv_exclusions_review.txt")
    args = ap.parse_args()

    out_path = args.out or args.master.parent / "dv_exclusions.csv"
    review_path = args.review or args.master.parent / "dv_exclusions_review.txt"

    with args.master.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    excluded: list[tuple[str, str, str]] = []
    borderline: list[tuple[str, str, str]] = []
    unparsed: list[tuple[str, str]] = []

    for row in rows:
        iid = (row.get("item_id") or "").strip()
        text = (row.get("item_text") or "").strip()
        if not iid or not text:
            continue
        if S1_VERBS.search(text):
            continue  # S1 is out of the analysis entirely
        parts = split_alternatives(text)
        if parts is None:
            unparsed.append((iid, text))
            continue
        _a, b = parts
        if NEGATOR.match(b):
            excluded.append((iid, text, b))
        elif BORDERLINE.match(b):
            borderline.append((iid, text, b))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "item_text", "side_b"])
        w.writerows(excluded)

    lines = [
        "WP-25 DV-exclusion set — derivation record",
        "=" * 60,
        "",
        f"rule (strict)  : side B matches {NEGATOR.pattern}",
        f"S1 items        : skipped (already excluded from all analysis)",
        f"items scanned   : {len(rows)}",
        f"EXCLUDED        : {len(excluded)}",
        f"borderline kept : {len(borderline)}",
        f"unparsed        : {len(unparsed)}",
        "",
        "RESEARCH_CONTRACT.md says 18. That count is stated in prose only and no",
        "list or script in the repo reproduces it. Cite the number below instead,",
        "and cite this script as its derivation.",
        "",
        "-" * 60,
        f"EXCLUDED ({len(excluded)}) — side B has no nameable surface form",
        "-" * 60,
    ]
    for iid, text, b in excluded:
        lines.append(f"  {iid}  {text}")
        lines.append(f"          side B = {b!r}")
    lines += [
        "",
        "-" * 60,
        f"BORDERLINE, DELIBERATELY KEPT ({len(borderline)})",
        "-" * 60,
        "Side B names a thing that denotes an absence. The extractor has a string",
        "to match, so commitment to side B IS detectable and the item stays in.",
        "Widening the rule to swallow these would give"
        f" {len(excluded) + len(borderline)} exclusions.",
        "",
    ]
    for iid, text, b in borderline:
        lines.append(f"  {iid}  {text}")
        lines.append(f"          side B = {b!r}")
    if unparsed:
        lines += ["", "-" * 60, f"UNPARSED ({len(unparsed)}) — review by hand", "-" * 60]
        for iid, text in unparsed:
            lines.append(f"  {iid}  {text}")

    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"scanned      {len(rows)} items")
    print(f"EXCLUDED     {len(excluded)}")
    print(f"borderline   {len(borderline)} (kept in)")
    print(f"unparsed     {len(unparsed)}")
    print(f"\nwrote {out_path}")
    print(f"wrote {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
