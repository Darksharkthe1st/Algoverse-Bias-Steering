"""Post-run verifier — notes/14 §3 item 4.  The termination gate.

It asserts, over the manifest:

  * every step exited 0 and produced its declared outputs;
  * every `.json` parses;
  * every `.npy` has a valid header AND the expected shape;
  * every residual file's item ids match its sidecar, and its sidecar's count
    matches the array's first dimension.

It exits non-zero and prints a diff if anything fails.  It runs against the
LAPTOP copy, never the box (notes/14 §1.3), because a verifier that reads the
machine it is verifying proves nothing about what survived the transfer.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


class Check:
    def __init__(self):
        self.failures, self.checked = [], 0

    def ok(self, condition, message):
        self.checked += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    @property
    def passed(self) -> bool:
        return not self.failures


def _verify_npy(c: Check, path: str) -> np.ndarray | None:
    """Open with mmap, so a 13 GB residual file is validated without loading it.

    The largest single array in the real run is Race_x_gender on qwen-14b:
    15,960 x 40 x 5120 x 4 B = 13.1 GB.  A verifier that loaded that into RAM
    would OOM on the laptop it is supposed to be verifying (notes/19 §2.2).
    """
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as e:
        c.ok(False, f"{path}: unreadable .npy ({type(e).__name__}: {e})")
        return None
    c.ok(arr.ndim == 3, f"{path}: expected 3-D (n_items, n_layers, d_model), got {arr.shape}")
    c.ok(arr.dtype == np.float32, f"{path}: expected float32, got {arr.dtype}")
    c.ok(arr.size > 0, f"{path}: empty array")
    return arr


def verify(out_dir: str, *, manifest_name: str = "queue_manifest.json") -> Check:
    c = Check()
    manifest_path = os.path.join(out_dir, manifest_name)

    if not c.ok(os.path.exists(manifest_path), f"missing manifest: {manifest_path}"):
        return c
    try:
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    except Exception as e:
        c.ok(False, f"{manifest_path}: unparseable ({e})")
        return c

    # --- every step exited 0 and produced what it declared ----------------- #
    for s in manifest.get("steps", []):
        c.ok(s["status"] == "OK",
             f"step {s['name']}: status={s['status']} exit={s['exit_code']} "
             f"missing={s.get('missing_outputs')}"
             + (f"\n{s['error']}" if s.get("error") else ""))
        for p in s.get("produces", []):
            c.ok(os.path.exists(p), f"step {s['name']}: declared output missing: {p}")

    # --- every json parses, every npy is well-formed ----------------------- #
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.endswith(".json"):
                try:
                    json.load(open(path, encoding="utf-8"))
                except Exception as e:
                    c.ok(False, f"{path}: unparseable json ({e})")
            elif fn.endswith(".jsonl"):
                try:
                    with open(path, encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if line.strip():
                                json.loads(line)
                except Exception as e:
                    c.ok(False, f"{path}:{i}: unparseable jsonl ({e})")
            elif fn.endswith(".npy"):
                _verify_npy(c, path)

    # --- residuals: ids match the sidecar, count matches the array --------- #
    resid_dir = os.path.join(out_dir, "residuals")
    if os.path.isdir(resid_dir):
        npys = sorted(f for f in os.listdir(resid_dir) if f.endswith(".npy"))
        c.ok(bool(npys), f"{resid_dir}: no residual arrays were persisted "
                         f"(this is S5 recurring — stop, do not proceed)")
        for fn in npys:
            npy = os.path.join(resid_dir, fn)
            side = npy[:-4] + ".json"
            if not c.ok(os.path.exists(side), f"{npy}: no sidecar {os.path.basename(side)}"):
                continue
            meta = json.load(open(side, encoding="utf-8"))
            arr = _verify_npy(c, npy)
            if arr is None:
                continue
            ids = meta.get("item_ids", [])
            c.ok(len(ids) == arr.shape[0],
                 f"{npy}: sidecar lists {len(ids)} item ids, array has {arr.shape[0]} rows")
            c.ok(len(set(ids)) == len(ids), f"{npy}: duplicate item ids in sidecar")
            c.ok(meta.get("n_layers") == arr.shape[1],
                 f"{npy}: sidecar n_layers={meta.get('n_layers')} vs array {arr.shape[1]}")
            c.ok(meta.get("d_model") == arr.shape[2],
                 f"{npy}: sidecar d_model={meta.get('d_model')} vs array {arr.shape[2]}")
            c.ok(meta.get("capture_site") is not None,
                 f"{npy}: sidecar records no capture_site — the mandatory "
                 f"pre-registration field (notes/11 §4, incident I-5)")

    # --- prompts and responses were persisted verbatim --------------------- #
    # N6 (notes/18) requires raw RESPONSE text to be saved because the
    # generation-era misparse rate became unmeasurable without it. A
    # capture-only run (run2_annotation_contrast) generates no text at all,
    # so there is nothing N6 could apply to: capture_site.json marks such
    # runs, and responses.jsonl is required only when it is absent.
    capture_only = os.path.exists(os.path.join(out_dir, "capture_site.json"))
    required = [("prompts.jsonl", "notes/13 §13 — every prompt, verbatim")]
    if not capture_only:
        required.append(("responses.jsonl",
                         "notes/18 — N6 is why this is non-negotiable"))
    for name, why in required:
        path = os.path.join(out_dir, name)
        if c.ok(os.path.exists(path), f"missing {name} ({why})"):
            n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            c.ok(n > 0, f"{path}: present but empty ({why})")

    return c


def main(argv: list) -> int:
    if len(argv) != 2:
        print("usage: python -m scripts.pilot.verifier <out_dir>", file=sys.stderr)
        return 2
    c = verify(argv[1])
    print(f"verifier: {c.checked} checks, {len(c.failures)} failures")
    for f in c.failures:
        print(f"  FAIL  {f}")
    if c.passed:
        print("VERIFIER PASSED — this is the termination gate (notes/14 §3).")
    else:
        print("VERIFIER FAILED — do not terminate the box, do not report numbers.")
    return 0 if c.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
