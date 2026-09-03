"""Collect a finished run into ONE self-contained folder under `results/`.

    python3 -m scripts.collect_run3 --out results/run3_2026-09-01

The standing rule for this project: **every piece of data is saved**, and a
finished experiment lives in one folder that a reader can open without knowing
where anything was originally written.  The project's worst historical defect
(S5) was exactly this failure -- residual tensors were never persisted, so every
follow-up question became another GPU rental.

WHAT GOES IN THE REPO, AND WHAT CANNOT
--------------------------------------
Everything except the residual tensors goes in verbatim: prompts, every model
completion, every judge verdict, the extracted direction vectors, every reported
number, and every run manifest.

Residual tensors cannot: they run to ~9 GB across five models and individual
files exceed GitHub's 100 MiB hard limit.  Dropping them is not the answer, so
this writes `RESIDUALS_MANIFEST.json` -- every file with its sha256, shape and
size -- next to the rest.  The collection is then complete and verifiable: a
reader can tell exactly which tensors existed and check that a copy fetched from
external storage is the one this analysis used.

Large `.jsonl` files are gzipped.  `steering_responses.jsonl` alone reaches
~95 MB on qwen-14b once cross-application has run, which is inside GitHub's limit
but past its warning threshold; gzip takes it to single-digit MB and `pandas`,
`jq` and `zcat` all read it without unpacking.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Copied verbatim; gzipped if over `GZIP_OVER`.
PER_RUN_FILES = [
    "capture_site.json", "prompts.jsonl", "responses.jsonl",
    "judge_qualification.json", "judge_labels.jsonl", "judge_labels_alt.jsonl",
    "queue_manifest.json", "report_behavioural.json", "report_steering.json",
    "steering_responses.jsonl", "positive_control.json",
]
GZIP_OVER = 8 * 1024 * 1024          # 8 MB


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _copy(src: str, dst_dir: str) -> dict:
    """Copy one file, gzipping if large.  Returns its record for the index."""
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.basename(src)
    size = os.path.getsize(src)
    digest = sha256(src)
    if size > GZIP_OVER and name.endswith((".jsonl", ".json")):
        dst = os.path.join(dst_dir, name + ".gz")
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo)
    else:
        dst = os.path.join(dst_dir, name)
        shutil.copy2(src, dst)
    return {"file": os.path.relpath(dst, dst_dir), "source": src,
            "bytes_original": size, "bytes_stored": os.path.getsize(dst),
            "sha256_original": digest,
            "gzipped": dst.endswith(".gz")}


def collect_residual_manifest(run_dir: str) -> list:
    """Record every residual tensor without copying it.

    Shape and dtype are read from the .npy header via mmap, so a 300 MB file is
    described without being loaded.
    """
    import numpy as np                                          # noqa: PLC0415

    d = os.path.join(run_dir, "residuals")
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npy"):
            continue
        p = os.path.join(d, fn)
        a = np.load(p, mmap_mode="r")
        out.append({"file": f"residuals/{fn}", "bytes": os.path.getsize(p),
                    "shape": list(a.shape), "dtype": str(a.dtype),
                    "sha256": sha256(p)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the one folder, e.g. results/run3_2026-09-01")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run directories; default = every runs/r3_* plus the controls")
    ap.add_argument("--docs", nargs="*", default=[
        "docs/EXPERIMENT_PLAN_RUN3.md", "docs/judges/v2-bbq-choice-llm.md"])
    args = ap.parse_args(argv)

    runs = args.runs
    if runs is None:
        rd = os.path.join(ROOT, "runs")
        runs = sorted(os.path.join("runs", n) for n in os.listdir(rd)
                      if n.startswith(("r3_", "_control_r3", "_smoke_r3",
                                       "r1_annotation", "_control_r1")))

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    index = {"collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "runs": {}, "docs": [], "residuals_external": {}}

    # --- the written procedure travels with the results --------------------- #
    dd = os.path.join(out, "procedure")
    for rel in args.docs:
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            index["docs"].append(_copy(src, dd))

    # --- code, so the folder is self-contained and auditable ---------------- #
    cd = os.path.join(out, "code")
    for rel in ("scripts/run3_behavioural_contrast.py", "scripts/pilot/behavioural.py",
                "scripts/pilot/llm_judge.py", "scripts/pilot/analysis.py",
                "scripts/pilot/pairing.py", "scripts/pilot/verifier.py",
                "scripts/overnight_queue.sh", "scripts/collect_run3.py"):
        src = os.path.join(ROOT, rel)
        if os.path.exists(src):
            _copy(src, cd)

    # --- per-run artifacts --------------------------------------------------- #
    total_stored = 0
    for rel in runs:
        src_dir = os.path.join(ROOT, rel)
        if not os.path.isdir(src_dir):
            continue
        name = os.path.basename(rel.rstrip("/\\"))
        dst_dir = os.path.join(out, "runs", name)
        files, missing = [], []
        for fn in PER_RUN_FILES:
            p = os.path.join(src_dir, fn)
            if os.path.exists(p):
                rec = _copy(p, dst_dir)
                files.append(rec)
                total_stored += rec["bytes_stored"]
            else:
                missing.append(fn)

        # direction vectors are small and are the headline object -- always in
        for sub in ("directions", "residuals"):
            sd = os.path.join(src_dir, sub)
            if sub == "directions" and os.path.isdir(sd):
                for fn in sorted(os.listdir(sd)):
                    rec = _copy(os.path.join(sd, fn), os.path.join(dst_dir, "directions"))
                    files.append(rec)
                    total_stored += rec["bytes_stored"]

        resid = collect_residual_manifest(src_dir)
        if resid:
            with open(os.path.join(dst_dir, "RESIDUALS_MANIFEST.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"note": "tensors are NOT in the repo -- too large for "
                                   "GitHub. Stored externally; verify with sha256.",
                           "n_files": len(resid),
                           "bytes_total": sum(r["bytes"] for r in resid),
                           "files": resid}, f, indent=2)
            index["residuals_external"][name] = {
                "n_files": len(resid), "bytes_total": sum(r["bytes"] for r in resid)}
        index["runs"][name] = {"files": files, "missing": missing,
                               "n_residual_tensors": len(resid)}

    with open(os.path.join(out, "INDEX.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    # --- a human entry point ------------------------------------------------- #
    lines = ["# Run 3 — complete collected results", "",
             f"Collected {index['collected_at']}.", "",
             "Everything this experiment produced, in one place. "
             "`INDEX.json` lists every file with its original size and sha256.", "",
             "| folder | contents |", "|---|---|",
             "| `procedure/` | the experimental plan and the judge version record |",
             "| `code/` | the exact scripts that produced these results |",
             "| `runs/<run>/` | prompts, every completion, every judge verdict, "
             "extracted vectors, and every reported number |", "",
             "## Residual tensors", "",
             "Not in the repo — several GB, and individual files exceed GitHub's "
             "100 MiB limit. Each run folder carries `RESIDUALS_MANIFEST.json` "
             "listing every tensor with shape, size and sha256, so a copy fetched "
             "from external storage can be verified as the one used here.", "",
             "| run | tensors | bytes |", "|---|---|---|"]
    for k, v in index["residuals_external"].items():
        lines.append(f"| `{k}` | {v['n_files']} | {v['bytes_total']:,} |")
    lines += ["", "## Files that were expected and are absent", ""]
    any_missing = False
    for k, v in index["runs"].items():
        if v["missing"]:
            any_missing = True
            lines.append(f"- `{k}`: {', '.join(v['missing'])}")
    if not any_missing:
        lines.append("None — every expected artifact is present.")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"collected {len(index['runs'])} run(s) into {out}")
    print(f"  stored {total_stored / 1e6:.1f} MB in the repo")
    for k, v in index["residuals_external"].items():
        print(f"  {k}: {v['n_files']} tensors, {v['bytes_total'] / 1e9:.2f} GB external")
    big = [f for r in index["runs"].values() for f in r["files"]
           if f["bytes_stored"] > 100 * 1024 * 1024]
    if big:
        print("\n  *** OVER GITHUB'S 100 MiB LIMIT — will be rejected on push:")
        for f in big:
            print(f"      {f['file']}  {f['bytes_stored'] / 1e6:.1f} MB")
        return 1
    for k, v in index["runs"].items():
        if v["missing"]:
            print(f"  note: {k} missing {', '.join(v['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
