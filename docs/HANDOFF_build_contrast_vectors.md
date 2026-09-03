# Handoff — build the 3 contrast vectors (Phases 2–3) on a GPU box

Builds `V1/V2/V3.safetensors` (the judge-v2.1 steering vectors) from a prompt pool:
capture residuals, judge with judge v2.1, bucket + collapse + pool, gate on group
size, build, save. This is **first-class pipeline behavior**, not a script:

    python -m src.bias_steer vectors configs/contrast_vectors_qwen3.py

(`experiment.build_contrast_vectors`, same front half as `run()`.)

## Prerequisites

- **GPU box** that can load `Qwen/Qwen3-8B` via TransformerLens (the box the
  `20260901-*_qwen3-8b` runs used — known-good).
- **`OPENAI_API_KEY`** (judge v2.1) and **`HF_TOKEN`** (model download), in `.env`
  at the repo root or exported.
- This branch checked out (it has `experiment.build_contrast_vectors`,
  `src/bias_steer/contrasts.py`, and `configs/contrast_vectors_qwen3.py`).

## Assemble the pool (one-time data prep)

The config points `dataset.path` at a combined **`plain`** file (one prompt per
line). It must contain all the poles the three contrasts need:

| vector | poles | source of those responses |
|---|---|---|
| V1 | soft-refusal ← **hard-refusal** | safety/harm prompts (Do_Not_Answer) |
| V2 | stance ← soft-refusal | opinion comparisons, BBQ, issuebench |
| V3 | stance ← non-engagement | mixed controversial prompts |

**hard-refusal is the scarce pole** (calibration: 4/40), so make sure the pool has
plenty of Do_Not_Answer-style prompts. **Exclude the 40 calibration items**
(`datasets/Calibration/calibration_v2_prompts.csv`). Aim for ≥ 2×`n_floor` expected
per pole. Write the file to `datasets/Calibration/vector_pool.txt` (or edit the
config path).

## Run + knobs

```
python -m src.bias_steer vectors configs/contrast_vectors_qwen3.py \
    --n-floor 40                 # min per pole to build a vector (default 40)
    # --build-under-floor        # build even thin contrasts (default: skip them)
```

Config knobs: `dataset.train_split` (default 0.5 — half builds, half held out for
Phase 4), `max_tokens` (2048, room for qwen3 `<think>` + answer), `batch_size`,
`strip_reasoning=True` (judge sees the post-`</think>` answer; residuals are still
captured over the full response — see the open decision below).

## Two open methodological decisions (documented; defaults chosen)

1. **Capture span vs `<think>`.** Residuals are captured over the **full** response
   (`capture_mean`); the judge sees only the answer (`strip_reasoning`). If qwen3's
   reasoning trace dominates the token mean, the behavior signal dilutes — revisit
   by restricting capture to the answer span if the norm profiles look flat.
2. **Pool composition / `n_floor`.** Start at 40; the gate table says which
   contrasts cleared it. Enrich the pool rather than lowering the floor.

## Outputs (`runs/<ts>_<label>_<model>/`)

- `V1.safetensors`, `V2.safetensors`, `V3.safetensors` — only contrasts that cleared
  the floor (each shape-asserted `(n_layers, d_model)` by `save_vector`). The
  printed `vectors built: [...]` line says which.
- `test_split.csv` — the held-out items Phase 4 evaluates on.
- `manifest.json` + `logs/` — the standard run manifest (from `open_run`): model +
  pinned revision, dataset, split; and `logs/` records the per-bucket counts.

## Definition of done

- [ ] The `vectors built: [...]` line lists each intended vector (a skipped one was
      under the floor — enrich the pool if that wasn't intended).
- [ ] The expected `.safetensors` exist and are non-empty.
- [ ] `manifest.json` records the model revision + split; `logs/` shows the bucket counts.
- [ ] `test_split.csv` is non-empty (the held-out half — disjoint from TRAIN by construction).
- [ ] Commit the run folder + push to the branch (raw artifacts, no hand edits).

## Next

Phase 4 (`fk/phase4-coeff-sweep`) sweeps each vector's coefficient on
`test_split.csv`, re-judging with judge v2.1 — feed it these vectors and that split.
