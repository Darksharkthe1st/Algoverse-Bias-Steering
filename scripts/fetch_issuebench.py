#!/usr/bin/env python3
"""Fetch the IssueBench prompt splits (Röttger et al., 2025; arXiv:2502.08395).

These are third-party parquet files from `Paul/IssueBench` on the HuggingFace
Hub (CC-BY-4.0): the pre-materialised writing-assistance prompts (template ×
issue) plus the per-issue metadata. They are NOT committed to this repo — this
script downloads them, driven entirely by `manifest.json`, which pins the exact
source commit and the expected byte size of every file.

By default only the small `debug` split (150 prompts) and the issue metadata are
fetched — enough for CI and a boundary-check pilot. `--split sample|full` adds
the larger splits.

Usage:
    python scripts/fetch_issuebench.py                 # debug split + issue metadata
    python scripts/fetch_issuebench.py --split sample  # + the 636k subsample
    python scripts/fetch_issuebench.py --split full    # + the whole 2.49m bench
    python scripts/fetch_issuebench.py --force         # re-download everything

Stdlib only (urllib) — importing pandas/the ML stack is not required to fetch;
only the `issuebench` dataset loader needs pandas to read the parquet.
"""

import argparse
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "third_party" / "issuebench" / "manifest.json"

# `meta` is always fetched (small, and the loader may join on it); the prompt
# splits are opt-in and cumulative: "sample" implies "debug", "full" implies both.
_SPLIT_ORDER = ["debug", "sample", "full"]


def _dest_for(src_path: str, manifest: dict) -> Path:
    """Map a source repo path to its local destination under the repo root."""
    repo_root = MANIFEST.parent.parent.parent  # third_party/issuebench/ -> repo root
    return repo_root / manifest["dest_root"] / src_path


def _wanted_splits(requested: str) -> set:
    """Which manifest `split` tags to fetch for a requested level.

    `meta` is always included. A requested level pulls itself and everything
    cheaper than it (debug ⊂ sample ⊂ full), so `--split full` gives a folder
    that also satisfies the debug loader path.
    """
    upto = _SPLIT_ORDER[: _SPLIT_ORDER.index(requested) + 1]
    return {"meta", *upto}


def _to_jsonl(parquet_path: Path, *, force: bool) -> Path | None:
    """Write a sibling `.jsonl` next to a fetched parquet — human-readable,
    grep-able, diff-able. Needs pandas (lazy import, so the default fetch stays
    stdlib-only). Returns the jsonl path, or None if an up-to-date one exists.

    NOTE: the `issuebench` loader still reads the parquet; the jsonl is purely for
    inspection and is git-ignored alongside it. For the full split this file can be
    ~1-2 GB (JSONL repeats every string per row — see README on why parquet ships).
    """
    jsonl = parquet_path.with_suffix(".jsonl")
    if jsonl.exists() and not force and jsonl.stat().st_mtime >= parquet_path.stat().st_mtime:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)  # raises ImportError if no parquet engine
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "--to-jsonl needs pandas and a parquet engine (pyarrow).\n"
            "  Install with:  pip install pandas pyarrow\n"
            f"  (underlying import error: {e})"
        ) from e
    df.to_json(jsonl, orient="records", lines=True, force_ascii=False)
    return jsonl


def _download(url: str, dest: Path, expected_bytes: int | None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"HTTP {e.code} fetching {url}\n"
            f"  If this is a 401/403 the dataset may be gated — accept its terms at "
            f"https://huggingface.co/datasets/Paul/IssueBench and set HF_TOKEN."
        ) from e
    got = tmp.stat().st_size
    if expected_bytes is not None and got != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"size mismatch for {dest.name}: got {got}, expected {expected_bytes} "
            f"(is the pinned commit still {url.split('/resolve/')[-1].split('/')[0][:10]}?)"
        )
    tmp.replace(dest)
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="debug", choices=_SPLIT_ORDER,
                    help="largest prompt split to fetch (cumulative; default: debug)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--to-jsonl", action="store_true",
                    help="also write a sibling .jsonl next to each parquet (human-readable; needs pandas)")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    base = manifest["raw_base_url"]
    wanted = _wanted_splits(args.split)
    files = [f for f in manifest["files"] if f.get("split") in wanted]

    print(f"source: {manifest['source_repo']}@{manifest['source_commit'][:10]}  "
          f"(split={args.split}: {len(files)} files)")
    fetched = skipped = 0
    for entry in files:
        src, want = entry["path"], entry.get("bytes")
        dest = _dest_for(src, manifest)
        if dest.exists() and not args.force and (want is None or dest.stat().st_size == want):
            skipped += 1
            continue
        n = _download(base + src, dest, want)
        fetched += 1
        print(f"  + {dest.relative_to(MANIFEST.parent.parent.parent)}  ({n/1e6:.1f} MB)")

    print(f"done: {fetched} fetched, {skipped} already present -> {manifest['dest_root']}")

    if args.to_jsonl:
        repo_root = MANIFEST.parent.parent.parent
        converted = 0
        for entry in files:
            dest = _dest_for(entry["path"], manifest)
            if not dest.exists():
                continue
            out = _to_jsonl(dest, force=args.force)
            if out is not None:
                converted += 1
                print(f"  ~ {out.relative_to(repo_root)}  ({out.stat().st_size/1e6:.1f} MB jsonl)")
        print(f"jsonl: {converted} written, {len(files) - converted} up-to-date/absent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
