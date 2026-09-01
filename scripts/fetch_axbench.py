#!/usr/bin/env python3
"""Fetch AxBench Concept500 data (Wu et al., 2025; arXiv:2501.17148).

Third-party parquet from `pyvene/axbench-concept500` on the HuggingFace Hub
(CC-BY-4.0): per-concept labelled `positive` / `negative` instruction+response
pairs — the contrast our difference-of-means steering is built from. NOT
committed; this script downloads them, driven by `manifest.json`, which pins the
exact source commit and the expected byte size of every file.

AxBench shipped one copy of the data per (extraction model, layer); `--variant`
selects which copy of the labelled text to read (default 2b/l20). Both the train
and test split of a variant are fetched.

Usage:
    python scripts/fetch_axbench.py                  # default variant (2b/l20)
    python scripts/fetch_axbench.py --variant 9b/l31 # a different released copy
    python scripts/fetch_axbench.py --force          # re-download

Stdlib only (urllib) — only the `axbench` dataset loader needs pandas.
"""

import argparse
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "third_party" / "axbench" / "manifest.json"


def _dest_for(src_path: str, manifest: dict) -> Path:
    repo_root = MANIFEST.parent.parent.parent  # third_party/axbench/ -> repo root
    return repo_root / manifest["dest_root"] / src_path


def _to_jsonl(parquet_path: Path, *, force: bool) -> Path | None:
    """Write a sibling `.jsonl` next to a fetched parquet — human-readable,
    grep-able, diff-able. Needs pandas (lazy import, so the default fetch stays
    stdlib-only). Returns the jsonl path, or None if an up-to-date one exists.

    NOTE: the `axbench` loader still reads the parquet; the jsonl is purely for
    inspection and is git-ignored alongside it. Each row carries the full `output`
    text, so the jsonl is larger than the prompts alone.
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
            f"https://huggingface.co/datasets/pyvene/axbench-concept500 and set HF_TOKEN."
        ) from e
    got = tmp.stat().st_size
    if expected_bytes is not None and got != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"size mismatch for {dest.name}: got {got}, expected {expected_bytes}")
    tmp.replace(dest)
    return got


def main(argv=None) -> int:
    manifest = json.loads(MANIFEST.read_text())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=manifest["default_variant"],
                    choices=manifest["variants"],
                    help=f"which released copy to read (default: {manifest['default_variant']})")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--to-jsonl", action="store_true",
                    help="also write a sibling .jsonl next to each parquet (human-readable; needs pandas)")
    args = ap.parse_args(argv)

    base = manifest["raw_base_url"]
    files = [f for f in manifest["files"] if f.get("variant") == args.variant]

    print(f"source: {manifest['source_repo']}@{manifest['source_commit'][:10]}  "
          f"(variant={args.variant}: {len(files)} files)")
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
