#!/usr/bin/env python3
"""Build disjoint calibration + scored annotation pools from archived text logs.

Gate 1 hygiene:
  - strip chat scaffolds / control tokens for display
  - blind to arm (arm stored only in a sealed key file)
  - calibration and scored sets are disjoint by content hash
  - never trusts historical judge labels as gold (they are v1, retired)

Usage:
  python3 scripts/build_annotation_pools.py \\
      --out-dir experiments/annotation_pools \\
      --n-calibrate 30 \\
      --n-scored 150 \\
      --seed 20260810
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.textlog_parse import ARM_NAMES, parse_steered_log

# Historical control / chat debris — strip for human judges.
SCAFFOLD_RE = re.compile(
    r"<\|im_start\|>.*?(?:<\|im_end\|>|$)"
    r"|<\|im_end\|>"
    r"|</?s>"
    r"|\[/?INST\]"
    r"|<<SYS>>.*?<</SYS>>"
    r"|^\s*PROMPT:\s*.*$"
    r"|^\s*OUTPUT:\s*",
    re.I | re.M | re.S,
)
MULTI_NL = re.compile(r"\n{3,}")


def clean_response(text: str) -> str:
    t = SCAFFOLD_RE.sub(" ", text or "")
    # drop role headers left after partial cleans
    t = re.sub(r"^(system|user|assistant)\s*$", "", t, flags=re.I | re.M)
    t = MULTI_NL.sub("\n\n", t)
    return t.strip()


def content_hash(prompt: str, response: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.strip().encode())
    h.update(b"\0")
    h.update(response.strip().encode())
    return h.hexdigest()[:16]


def iter_items(log_paths: list[Path]) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for path in log_paths:
        try:
            parsed = parse_steered_log(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        for i, rec in enumerate(parsed.records):
            prompt = (rec.prompt or "").strip()
            if not prompt:
                continue
            for arm in ARM_NAMES:
                arm_resp = rec.arm(arm)
                if arm_resp is None or not (arm_resp.text or "").strip():
                    continue
                cleaned = clean_response(arm_resp.text)
                if len(cleaned) < 20:
                    continue
                ch = content_hash(prompt, cleaned)
                if ch in seen:
                    continue
                seen.add(ch)
                items.append(
                    {
                        "content_hash": ch,
                        "source_log": str(path),
                        "record_index": i,
                        "arm": arm,  # sealed later
                        "legacy_v1_label": arm_resp.label,  # not gold
                        "prompt": prompt,
                        "response": cleaned,
                        "response_raw_len": len(arm_resp.text or ""),
                    }
                )
    return items


def write_annotator_csv(path: Path, rows: list[dict], *, include_arm: bool) -> None:
    fields = [
        "item_id",
        "prompt",
        "response",
        "label",  # blank for annotator
        "notes",
        "dual_frame_hard_soft",  # y/n optional
        "safety_relevant",  # y/n/u optional axis
    ]
    if include_arm:
        fields = [
            "item_id",
            "content_hash",
            "arm",
            "legacy_v1_label",
            "source_log",
            "record_index",
            "prompt",
            "response",
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--logs-glob",
        default="experiments/past_logs/**/*_steered*.txt",
        help="Glob for archived steered text logs",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/annotation_pools"))
    ap.add_argument("--n-calibrate", type=int, default=30)
    ap.add_argument("--n-scored", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--max-logs", type=int, default=80, help="Cap logs scanned (speed)")
    args = ap.parse_args()

    root = Path.cwd()
    log_paths = sorted(root.glob(args.logs_glob))
    if not log_paths:
        # fallback: rglob from past_logs
        log_paths = sorted((root / "experiments/past_logs").rglob("*_steered*.txt"))
    if args.max_logs and len(log_paths) > args.max_logs:
        rng = random.Random(args.seed)
        log_paths = rng.sample(log_paths, args.max_logs)

    print(f"scanning {len(log_paths)} logs…")
    items = iter_items(log_paths)
    print(f"unique prompt×response items: {len(items)}")
    if len(items) < args.n_calibrate + args.n_scored:
        raise SystemExit(
            f"not enough items ({len(items)}) for calibrate={args.n_calibrate} "
            f"+ scored={args.n_scored}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(items)

    cal = items[: args.n_calibrate]
    scored = items[args.n_calibrate : args.n_calibrate + args.n_scored]
    # assign blind item ids
    for i, row in enumerate(cal, 1):
        row["item_id"] = f"cal_{i:03d}"
        row["label"] = ""
        row["notes"] = ""
        row["dual_frame_hard_soft"] = ""
        row["safety_relevant"] = ""
    for i, row in enumerate(scored, 1):
        row["item_id"] = f"score_{i:03d}"
        row["label"] = ""
        row["notes"] = ""
        row["dual_frame_hard_soft"] = ""
        row["safety_relevant"] = ""

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sealed = out / "sealed"
    sealed.mkdir(exist_ok=True)

    # Annotator-facing (no arm)
    write_annotator_csv(out / "calibration_blank.csv", cal, include_arm=False)
    write_annotator_csv(out / "scored_blank.csv", scored, include_arm=False)
    # Second annotator gets a reshuffled scored sheet (same ids, different order)
    scored_b = scored[:]
    rng.shuffle(scored_b)
    write_annotator_csv(out / "scored_blank_annotator_B_shuffled.csv", scored_b, include_arm=False)

    # Sealed keys (do not share with annotators during labeling)
    write_annotator_csv(sealed / "calibration_key.csv", cal, include_arm=True)
    write_annotator_csv(sealed / "scored_key.csv", scored, include_arm=True)

    manifest = {
        "seed": args.seed,
        "n_calibrate": len(cal),
        "n_scored": len(scored),
        "n_pool_scanned": len(items),
        "n_logs": len(log_paths),
        "rubric": "docs/RUBRIC_v2.md",
        "judge_version_for_scored_pass": "TBD — freeze hash into docs/PREREG.md first",
        "note": (
            "Historical v1 labels in sealed keys are NOT gold. "
            "Calibration may iterate the rubric; scored pass requires freeze."
        ),
        "calibration_ids": [r["item_id"] for r in cal],
        "scored_ids": [r["item_id"] for r in scored],
    }
    (out / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote pools → {out}")
    print("  calibration_blank.csv, scored_blank.csv (+ annotator B shuffle)")
    print("  sealed/*_key.csv (arm + legacy labels — keep offline during labeling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
