#!/usr/bin/env python3
"""Fetch the pre-computed refusal-direction artifacts (Arditi et al., 2024).

These are third-party files from `andyrdt/refusal_direction` (Apache-2.0): the
selected refusal `direction.pt` per model, its `direction_metadata.json`
(source layer + token position), the full `mean_diffs.pt` candidate grid, and
the paper's committed completions/evaluations JSON (used as the ground truth we
diff our reproduction against). They are NOT committed to this repo — this
script downloads them, driven entirely by `manifest.json`, which pins the exact
source commit and the expected byte size of every file.

Usage:
    python scripts/fetch_refusal_artifacts.py            # fetch all, skip existing
    python scripts/fetch_refusal_artifacts.py --force    # re-download everything
    python scripts/fetch_refusal_artifacts.py --model qwen-1_8b-chat   # one model

Stdlib only (urllib) — importing torch/the ML stack is not required to fetch.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "third_party" / "refusal_direction" / "manifest.json"


def _dest_for(src_path: str, manifest: dict) -> Path:
    """Map a source repo path to its local destination under the repo root.

    `pipeline/runs/<model>/x` -> `<dest_root>/runs/<model>/x` (strip_prefix removed).
    """
    strip = manifest.get("strip_prefix", "")
    rel = src_path[len(strip):] if strip and src_path.startswith(strip) else src_path
    repo_root = MANIFEST.parent.parent.parent  # third_party/refusal_direction/ -> repo root
    return repo_root / manifest["dest_root"] / rel


def _download(url: str, dest: Path, expected_bytes: int | None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} fetching {url}") from e
    got = tmp.stat().st_size
    if expected_bytes is not None and got != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"size mismatch for {dest.name}: got {got}, expected {expected_bytes}")
    tmp.replace(dest)
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--model", default=None, help="only fetch files for this run dir (e.g. qwen-1_8b-chat)")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    base = manifest["raw_base_url"]
    files = manifest["files"]
    if args.model:
        files = [f for f in files if f"/runs/{args.model}/" in f["path"]]
        if not files:
            raise SystemExit(f"no manifest files for model {args.model!r}; known: {manifest['models']}")

    print(f"source: {manifest['source_repo']}@{manifest['source_commit'][:10]}  ({len(files)} files)")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
