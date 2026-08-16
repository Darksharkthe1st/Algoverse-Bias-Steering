# Refusal-direction GENERATION track (arXiv:2406.11717)

Reproduce the paper's `generate_directions` + `select_direction` stages *ourselves*
(vs loading their vectors, which the sibling "load" track does in `refusal.py`), and
validate against the committed artifacts already fetched into
`third_party/refusal_direction/`.

Two ground-truth checks anchor the track:
- **Extraction check** — our candidate grid vs their `mean_diffs.pt`, cosine per
  `(pos, layer)` cell. Cosine ≈ 1 proves the recipe (template, positions,
  formatting, filtering) is exactly right. Needs no selection and no refusal metric.
- **Selection check** — does our `select_direction` land on their published
  `(layer, pos)`?

## What runs where

**CPU-only (validated in this environment, no model):**
- `tools/refusal_grid_provenance.py` + `tests/test_refusal_grid_provenance.py` —
  Chunk A0. Decodes `mean_diffs.pt` / `direction.pt` layout from the pickle with the
  stdlib (no torch): grid is `(n_pos, n_layers, d_model)`, float64, C-contiguous, and
  `direction.pt` is the storage-view `mean_diffs[pos, layer]` named in the metadata.
- `src/bias_steer/datasets.py::load_refusal` + `tests/test_refusal_datasets.py` —
  Chunk B. Registered dataset `"refusal"`; loads `<label>_<split>.json`, tags each
  Example with `metadata["label"]`/`["split"]`.
- `src/bias_steer/steering.py::capture_prompt_positions` / `build_refusal_grid` and
  the `"refusal_extract"` method; `src/bias_steer/refusal_extract.py` templates +
  `load_and_sample_repro` + `run_extraction`; `tests/test_refusal_extract.py` —
  Chunks C/D. The torch-gated numeric tests self-skip on CPU and run on the GPU box.

**Deferred to the Lambda GPU box (model execution — DO NOT run in the CPU/planning
env):** `scripts/refusal_extract_check.py` (Chunk A) and
`scripts/refusal_select_direction.py` (Chunk E). Written and syntax-checked here;
they load a model, so a Claude instance ON Lambda runs them.

## Provenance decoded in Chunk A0 (per model)

| run dir | grid (n_pos, n_layers, d_model) | published (pos, layer) |
|---|---|---|
| qwen-1_8b-chat | (5, 24, 2048) | (-2, 15) |
| gemma-2b-it | (5, 18, 2048) | (-2, 10) |
| meta-llama-3-8b-instruct | (5, 32, 4096) | (-5, 12) |
| llama-2-7b-chat-hf | (6, 32, 4096) | (-1, 14) |
| yi-6b-chat | (6, 32, 4096) | (-5, 20) |

`n_pos` = number of end-of-instruction template tokens (model-specific). Position
axis index `i` maps to token position `i - n_pos`.

## The recipe (why the details matter for cosine ≈ 1)

- **Chat template, system=None.** Extraction formats each instruction with the
  upstream *literal* template (`refusal_extract.REFUSAL_TEMPLATES`), with **no system
  prompt**. This is NOT `models.render_prompts` (which injects `DEFAULT_SYS` as a
  system turn) and NOT HF `apply_chat_template` (which injects a default system
  prompt for Qwen). A system turn shifts every token position and breaks cosine.
- **Prompt activations, not response.** We read `hook_resid_pre` on the PROMPT at the
  last `n_pos` tokens, with no generation (`run_extraction`). The framework's
  `generate_with_cache` caches over the response — wrong for this.
- **Bucket by label, not judge verdict.** `experiment.run` buckets train residuals by
  judge verdict; `run_extraction` buckets by the known harmful/harmless label.
- **Exact instruction set.** `load_and_sample_repro` reproduces upstream's
  `random.seed(42); random.sample(...)` selection over the committed splits, and the
  Chunk A driver then applies the model-based `filter_train` (default True upstream):
  keep harmful with refusal_score > 0, harmless with refusal_score < 0. Skipping the
  filter (`--no-filter`) changes the mean and lowers cosine.
- **Precision.** Their grid is float64; the check casts both sides to float64. The
  main legitimate deviation is our fp16 forward vs their accumulation + padding.

## Running on Lambda

```bash
# 0. env: the transformer_lens + torch stack with CUDA; HF access for qwen-1.8b.
pip install -e .            # or the project's usual install
python scripts/fetch_refusal_artifacts.py            # grids, directions, splits (git-ignored)

# 1. Chunk A — extraction + cosine self-validation (anchor: qwen-1.8b, ~3.5GB dl)
python scripts/refusal_extract_check.py --model qwen-1.8b
#   Expect: cosine grid printed per (pos_index, layer); worst cell >= 0.999 = recipe
#   exact; >= 0.95 acceptable; < 0.95 = debug formatting/positions/filtering. The
#   selected cell (pos=-2, layer=15) is reported explicitly.

# 2. Chunk E — select_direction (heavy: ~n_pos x n_layers scored sweeps)
python scripts/refusal_select_direction.py --model qwen-1.8b
#   Expect: "selected: position=-2, layer=15" and "MATCH" against the published cell.
#   Defaults to selecting over their mean_diffs.pt (isolates selection from
#   extraction); add --use-extracted to select over our own grid.
```

Thresholds (per the approved plan): ≥ 0.999 per cell = recipe exactly right;
≥ 0.95 = acceptable, investigate outliers; < 0.95 = real problem.

## Selection metric note (important)

`select_direction` uses the paper's **logit-based** `refusal_score` (first-token
refusal-marker probability, `refusal_extract.get_refusal_scores`), NOT the substring
judge. The substring metric (`judge.is_refusal` on the sibling `fk/init-refusal-rewrite`
branch) is a downstream jailbreak-EVAL metric, used in a later chunk that reproduces
the paper's completions/evaluations — not in selection. So Chunk E does **not** import
`judge.is_refusal`; if/when the jailbreak-eval chunk is built it will, and if that
symbol is absent (branches not yet reconciled) it should be flagged, not reimplemented.
