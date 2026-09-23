"""Fetch BBQ's supplemental answer key (`target_loc`) from the upstream repo.

    python scripts/fetch_bbq_metadata.py

BBQ's `.jsonl` files under `datasets/BBQ_Prompt_Sets/` do NOT say which answer
counts as the biased one. They give `stereotyped_groups` (which group the
stereotype targets) and `question_polarity`, and leave you to work it out. That
reconstruction is doable but fiddly, and it fails outright on the intersectional
sets where both answers share the stereotyped race.

The authors ship the answer directly, in a file that is not part of the per-
category jsonl release: `supplemental/additional_metadata.csv` carries a
**`target_loc`** column giving the index (0/1/2) of the biased answer for each
`(category, example_id)`.

Two properties of `target_loc` that matter, both verified against the local
files before this script was written:

1. **Polarity is already folded in.** For `Race_ethnicity` example 0 ("Who
   likely committed a crime?", negative) it points at the stereotyped group; for
   example 2, same context, "Who was likely innocent?" (non-negative), it points
   at the OTHER group. So it means "which answer is biased", not "where the
   stereotyped group is" — and a polarity flip must NOT be applied on top of it.
2. **It never points at the unknown option** on an ambiguous row (0 of 25,814).

Downloads to `third_party/bbq/additional_metadata.csv`, mirroring how
`scripts/fetch_refusal_artifacts.py` stages third-party artifacts outside
`datasets/`. Re-running is safe; pass --force to re-download.
"""

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/nyu-mll/BBQ/main/"
       "supplemental/additional_metadata.csv")

DEST = Path(__file__).resolve().parents[1] / "third_party" / "bbq" / "additional_metadata.csv"

EXPECTED_COLUMNS = {"category", "example_id", "target_loc"}


def fetch(dest: Path = DEST, url: str = URL, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"already present: {dest}  (--force to re-download)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as r:
        data = r.read()
    text = data.decode("utf-8", "replace")

    # Validate before writing: a truncated or redirected download that lands as
    # HTML would otherwise sit there looking like a valid answer key.
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise SystemExit("downloaded file has no rows — refusing to write")
    missing = EXPECTED_COLUMNS - set(rows[0])
    if missing:
        raise SystemExit(
            f"downloaded file is missing expected columns {sorted(missing)}; "
            f"got {sorted(rows[0])}. Refusing to write."
        )

    dest.write_text(text, encoding="utf-8")
    na = sum(1 for r in rows if str(r.get("target_loc", "")).strip().upper() == "NA")
    print(f"wrote {dest}")
    print(f"  {len(rows):,} rows, {len({r['category'] for r in rows})} categories, "
          f"{na} rows with target_loc=NA")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()
    fetch(Path(args.dest), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
