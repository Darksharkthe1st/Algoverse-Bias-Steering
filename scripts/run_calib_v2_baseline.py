#!/usr/bin/env python3
"""Baseline (UNSTEERED) qwen3-8b generations for the v2 calibration battery.

Produces the labeling-ready sheet that feeds the human + LLM-judge calibration
(Cohen's kappa, `scripts/kappa_from_csv.py`). One greedy response per prompt for
the 40 items in `datasets/Calibration/calibration_v2_prompts.csv`.

THIS SCRIPT LOADS A MODEL. Do not run it in the planning / CPU environment — run
it on a GPU box (see `docs/HANDOFF_calib_v2_inference.md`). It:

  1. loads `qwen3-8b` through the shared loader (`bias_steer.models.load_model`),
     so the pinned HuggingFace revision in `MODEL_CATALOG` (Qwen/Qwen3-8B @
     b968826d9c46) is what gets fetched — the manifest records exactly that.
  2. generates greedily (`do_sample=False`) on all 40 prompts, in batches, using
     the same code path (`bias_steer.models.generate`) the real INITIAL/baseline
     condition uses in `experiment.py`, under the project default system prompt.
  3. splits qwen3's `<think>...</think>` reasoning trace off the front: the
     judged/labeled `response` is the FINAL answer after `</think>`; the full raw
     generation is kept verbatim in `raw_generations.csv`. See the handoff for
     why (the reasoning trace is not the model's answer and must not be labeled).

Writes `runs/<ts>_calib-v2_qwen3-8b/` with a complete manifest (model id +
revision, decode params, seed, prompt-file hash), the raw generations, and the
labeling sheet `item_id,prompt,response`.

Usage (on the GPU box, from the repo root):

    python scripts/run_calib_v2_baseline.py
    python scripts/run_calib_v2_baseline.py --batch-size 4      # if OOM
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The exact columns the labeling sheet must carry, in order. The kappa pipeline
# (scripts/kappa_from_csv.py) reads `item_id`; annotators add a `label` column.
SHEET_COLUMNS = ("item_id", "prompt", "response")

# Everything the raw record keeps, so the run folder is self-contained and the
# think-trace split is auditable (raw vs final side by side).
RAW_COLUMNS = (
    "item_id", "bucket", "source_dataset", "prompt",
    "response", "raw_response", "think_complete", "n_response_chars",
)

# Marker qwen3 closes its reasoning trace with. The final answer is everything
# after the LAST occurrence (nested/echoed markers are rare but cheap to guard).
THINK_CLOSE = "</think>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    for col in ("item_id", "prompt"):
        if col not in rows[0]:
            raise SystemExit(f"{path} missing required column '{col}'")
    return rows


def split_think(raw: str) -> tuple[str, bool]:
    """Return (final_answer, think_complete).

    `think_complete` is True iff the generation closed its `</think>` block; the
    final answer is then the text after the last `</think>`, stripped. If the
    trace never closed (hit the token budget mid-think), there IS no final answer
    — we return the full text stripped and flag it False so the operator raises
    the budget or excludes the item rather than labeling a truncated ramble.
    """
    if THINK_CLOSE in raw:
        return raw.rsplit(THINK_CLOSE, 1)[1].strip(), True
    return raw.strip(), False


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-8b",
                    help="MODEL_CATALOG key (default qwen3-8b; pins the HF revision)")
    ap.add_argument("--prompts", type=Path,
                    default=_REPO_ROOT / "datasets" / "Calibration" / "calibration_v2_prompts.csv",
                    help="prompt CSV with columns item_id,bucket,source_dataset,prompt")
    ap.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / "runs")
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="generation budget. qwen3-8b emits a <think> trace before its "
                         "answer; 2048 lets the short calibration prompts finish thinking "
                         "AND produce a full final answer (128, used by the extract runs, "
                         "truncates mid-think — no answer at all).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per generate() call (8B fp16). Drop to 4 if OOM.")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded for provenance; greedy decode is deterministic regardless")
    ap.add_argument("--label", default="calib-v2",
                    help="run-id label component -> runs/<ts>_<label>_<model>/")
    args = ap.parse_args(argv)

    if not args.prompts.is_file():
        raise SystemExit(f"prompt file not found: {args.prompts}")

    import torch
    from src.bias_steer import models
    from src.bias_steer.config import DEFAULT_SYS
    from src.bias_steer.registry import MODELS
    from src.bias_steer.tracking import git_sha, make_run_id
    from src.utils import get_current_time_str

    if args.model not in MODELS:
        raise SystemExit(f"unknown model '{args.model}'; known: {sorted(MODELS)}")
    spec = MODELS[args.model]
    if not spec.revision:
        raise SystemExit(
            f"model '{args.model}' has no pinned revision in MODEL_CATALOG — a bare "
            f"repo name is not provenance (PREREG §3b). Refusing to run.")

    system_prompt = DEFAULT_SYS  # the baseline INITIAL condition's system prompt

    rows = read_prompts(args.prompts)
    prompts = [r["prompt"] for r in rows]
    prompt_hash = sha256_file(args.prompts)

    # Deterministic greedy decode. manual_seed is belt-and-suspenders (argmax is
    # already deterministic); recorded in the manifest either way.
    torch.manual_seed(args.seed)

    when = get_current_time_str()
    run_id = make_run_id(args.label, args.model, when)
    run_dir = Path(args.runs_dir) / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{get_current_time_str()}] {msg}"
        with (run_dir / "logs" / "run.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"run_id={run_id}")
    log(f"loading model {spec.hf_id} @ revision {spec.revision}")
    loaded = models.load_model(spec)
    log(f"loaded on device={loaded.device}; "
        f"n_layers={loaded.model.cfg.n_layers} d_model={loaded.model.cfg.d_model}")

    # --- generate (greedy), batched -------------------------------------------
    raw_responses: list[str] = []
    for bi, batch in enumerate(_batches(prompts, args.batch_size)):
        log(f"generating batch {bi} ({len(batch)} prompts), "
            f"max_new_tokens={args.max_new_tokens}")
        raw_responses.extend(
            models.generate(loaded, batch, args.max_new_tokens, system_prompt)
        )
    assert len(raw_responses) == len(rows), (
        f"got {len(raw_responses)} responses for {len(rows)} prompts")

    # --- split think trace, assemble records ----------------------------------
    raw_records, sheet_records = [], []
    n_incomplete, n_empty = 0, 0
    gen_txt = run_dir / "logs" / "generations.txt"
    with gen_txt.open("w", encoding="utf-8") as gf:
        for r, raw in zip(rows, raw_responses):
            final, complete = split_think(raw)
            if not complete:
                n_incomplete += 1
            if not final.strip():
                n_empty += 1
            raw_records.append({
                "item_id": r["item_id"],
                "bucket": r.get("bucket", ""),
                "source_dataset": r.get("source_dataset", ""),
                "prompt": r["prompt"],
                "response": final,
                "raw_response": raw,
                "think_complete": complete,
                "n_response_chars": len(final),
            })
            sheet_records.append({
                "item_id": r["item_id"],
                "prompt": r["prompt"],
                "response": final,
            })
            gf.write(f"=== {r['item_id']} (think_complete={complete}) ===\n"
                     f"PROMPT: {r['prompt']}\n"
                     f"RAW:\n{raw}\n"
                     f"FINAL (labeled response):\n{final}\n\n")

    # --- write CSVs -----------------------------------------------------------
    raw_csv = run_dir / "raw_generations.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(RAW_COLUMNS))
        w.writeheader()
        w.writerows(raw_records)

    sheet_csv = run_dir / "labeling_sheet.csv"
    with sheet_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SHEET_COLUMNS))
        w.writeheader()
        w.writerows(sheet_records)

    # --- manifest -------------------------------------------------------------
    sha, dirty = git_sha()
    manifest = {
        "run_id": run_id,
        "label": args.label,
        "kind": "baseline-unsteered-generation",
        "purpose": "v2 calibration battery -> human + LLM-judge Cohen's kappa",
        "model": args.model,
        "model_spec": {"name": spec.name, "hf_id": spec.hf_id,
                       "revision": spec.revision or None},
        "timestamp": when,
        "git": {"sha": sha, "dirty": dirty},
        "decode": {
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "system_prompt": system_prompt,
            "chat_template": spec.chat_template,
        },
        "think_handling": {
            "close_marker": THINK_CLOSE,
            "labeled_response": "text after last </think>, stripped (raw kept in raw_generations.csv)",
        },
        "prompt_file": {
            "path": str(args.prompts.relative_to(_REPO_ROOT))
                    if args.prompts.is_relative_to(_REPO_ROOT) else str(args.prompts),
            "sha256": prompt_hash,
            "n_items": len(rows),
        },
        "outputs": {
            "labeling_sheet": sheet_csv.name,
            "raw_generations": raw_csv.name,
            "columns": list(SHEET_COLUMNS),
        },
        "counts": {
            "n_items": len(rows),
            "n_responses": len(sheet_records),
            "n_think_incomplete": n_incomplete,
            "n_empty_response": n_empty,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"wrote {sheet_csv} ({len(sheet_records)} rows)")
    log(f"wrote {raw_csv}")
    log(f"think_incomplete={n_incomplete}  empty_response={n_empty}")
    if n_incomplete:
        log("WARNING: some generations never closed </think> — raise --max-new-tokens "
            "or exclude those item_ids before labeling.")
    if n_empty:
        log("WARNING: some labeled responses are EMPTY — inspect raw_generations.csv.")
    log("done")
    print(f"\nrun folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
