# HANDOFF_calib_v2_inference — baseline qwen3-8b generations for the v2 calibration battery

**Operator handoff.** Everything here has been scoped and dry-checked on CPU
(the script compiles, the prompt file parses to 40 rows, the think-split logic
is unit-checked); **the model load and generation have NOT been run** — that is
your job on the GPU box. This doc may not redefine the calibration protocol; it
produces the input to it.

**What this produces:** one greedy, **unsteered** qwen3-8b response per prompt
for the 40 items in `datasets/Calibration/calibration_v2_prompts.csv`, written as
a labeling-ready sheet with columns exactly `item_id,prompt,response`. That sheet
feeds the human + LLM-judge calibration (Cohen's κ, `scripts/kappa_from_csv.py`):
annotators add a `label` column, they do **not** regenerate.

**Owner:** whoever has the GPU box. No coding required — one command.

---

## 0. Preconditions

- **GPU / VRAM.** Qwen3-8B in fp16 is ~16 GB of weights; with the KV cache for a
  batch of 8 at up to 2048 new tokens, budget **≥ 24 GB VRAM** (one A100-40GB,
  A6000, L4-24, or 4090-24 is comfortable). If you only have less, drop
  `--batch-size` (see §3). CPU-only is not viable (the load path is
  `float16` + CUDA/MPS).
- **Stack.** `transformer_lens` + `torch` must import and see CUDA:
  ```bash
  python -c "import transformer_lens, torch; print(transformer_lens.__version__, torch.__version__, torch.cuda.is_available())"
  ```
- **Qwen3 support is already demonstrated on this box.** The three
  `runs/20260901-*_qwen3-8b/` extraction runs loaded `Qwen/Qwen3-8B` under the
  installed TransformerLens and generated (see their `logs/run.log`). So the one
  precondition `docs/HANDOFF_G1.md` flags as unverified is, for this task,
  already known good. If the load nonetheless raises on an unsupported
  architecture, **stop and report it** — do not substitute another model
  (open-weight-only + provenance rules, CLAUDE.md §8).
- **No OpenAI key needed.** This is generation only; there is no judge in this
  step. (The κ pass that consumes the sheet is a separate, later step.)
- **HF access.** First run downloads the pinned Qwen3-8B revision; ensure
  `HF_TOKEN` is set if your environment gates it and that there is disk for the
  weights.

## 1. Get the branch

Pull the branch this was scoped on. It carries the script, the prompt file, and
this doc:

- `scripts/run_calib_v2_baseline.py`
- `datasets/Calibration/calibration_v2_prompts.csv`  (SHA256
  `a6bec92a66d8cbd780830f35e20c7755e72ff412c320a6a11e012cb1799757ba`, 40 items)
- `docs/HANDOFF_calib_v2_inference.md`  (this file)

## 2. The exact command

From the repo root, on the GPU box:

```bash
python scripts/run_calib_v2_baseline.py
```

That is the whole run. Defaults are the intended configuration:
`--model qwen3-8b`, `--prompts datasets/Calibration/calibration_v2_prompts.csv`,
`--max-new-tokens 2048`, `--batch-size 8`, `--seed 0`, writing under `runs/`.

## 3. Knobs (only if you need them)

- `--batch-size 4` (or `2`) if you OOM. Output is identical — greedy decode does
  not depend on batch composition here (left-padded, per-row token slicing).
- `--max-new-tokens N` to change the budget. See the think-trace note (§5) before
  lowering it; raise it if §7 reports any `think_incomplete`.
- Leave `--seed` at 0. Greedy (`do_sample=False`) is deterministic regardless;
  the seed is recorded for provenance only.

## 4. Expected runtime

40 prompts, greedy, batch 8, up to 2048 new tokens each. Short prompts finish and
emit EOS well before the cap, so real generation is far under the worst case.
Ballpark **5–15 minutes** wall-clock on a single modern GPU, plus one-time weight
download on a cold box. If it runs dramatically longer, a batch is being dragged
to the 2048 cap by a non-terminating item — check `logs/run.log`.

## 5. The think-trace decision (why `response` is the post-`</think>` answer)

qwen3-8b is a reasoning model: every generation is `<think> … </think>` followed
by the actual answer. **The reasoning trace is not the answer and must not be
what a human or the LLM judge labels for neutral-vs-opinionated.**

The decision, implemented in the script:

- **`response` = the text after the last `</think>`, stripped** — the model's
  final answer. This is what lands in `labeling_sheet.csv`.
- **The full raw generation is kept verbatim** in `raw_generations.csv`
  (`raw_response` column) alongside `response`, so the split is auditable and
  nothing is lost.

Why not match the existing `20260901-*` extraction runs, which judged the raw
text including `<think>`? Because those runs used `max_tokens=128`, and at that
budget qwen3-8b **never closes `</think>`** — the recorded "responses" are
truncated reasoning ramble with no answer at all (see any `logs/train.txt` /
`logs/eval.txt` there). That is an artifact of the 128-token cap, not a
deliberate think-handling convention worth inheriting for a calibration whose
entire point is labeling the model's *answer*. Hence the larger budget (§below)
plus the split.

**`max_new_tokens = 2048`** is chosen so the short calibration prompts finish the
`<think>` block **and** produce a complete final answer. Empirically the traces
in the existing runs were still going at 128 tokens; a few hundred tokens of
thinking plus a few hundred of answer fits comfortably under 2048, while bounding
worst-case runtime. If any item still fails to close `</think>`, the script flags
it (`think_complete=False`) rather than silently labeling a truncated trace.

**System prompt.** Generation uses the project default system prompt
(`config.DEFAULT_SYS`, "First give the clear, definitive answer and then explain
…") — the same prompt the real baseline/INITIAL condition uses in
`experiment.py`, so the calibrated judge is calibrated against the conditions it
will actually score. It is recorded in the manifest.

## 6. Outputs — what lands where

A single run folder `runs/<YYYYMMDD-HHMMSS>_calib-v2_qwen3-8b/` containing:

| File | Contents |
|---|---|
| `labeling_sheet.csv` | **the deliverable** — columns exactly `item_id,prompt,response` (post-`</think>` final answer), 40 rows |
| `raw_generations.csv` | `item_id,bucket,source_dataset,prompt,response,raw_response,think_complete,n_response_chars` — full raw generation + split, for audit |
| `manifest.json` | model id + pinned revision, decode params (greedy, max_new_tokens, batch_size, seed, system prompt), prompt-file path + SHA256 + n_items, git sha/dirty, think-handling record, counts |
| `logs/run.log` | event log (load, per-batch, warnings, done) |
| `logs/generations.txt` | human-readable prompt / raw / final per item |

## 7. Definition of done

The run is done — and the sheet is fit to hand to annotators — when **all** hold:

- [ ] `runs/<ts>_calib-v2_qwen3-8b/labeling_sheet.csv` exists with header exactly
      `item_id,prompt,response`.
- [ ] It has **40 data rows**, one per `cal2_001 … cal2_040`, item ids matching
      the prompt file.
- [ ] **No empty `response`** cells (`manifest.json → counts.n_empty_response == 0`).
- [ ] **No truncated reasoning** (`counts.n_think_incomplete == 0`). If nonzero,
      the `run.log` prints a warning — raise `--max-new-tokens` and re-run, or
      escalate before those items are labeled.
- [ ] `manifest.json` is complete: `model_spec.revision == "b968826d9c46"`,
      decode params present, `prompt_file.sha256 ==
      a6bec92a66d8cbd780830f35e20c7755e72ff412c320a6a11e012cb1799757ba`.
- [ ] Push the whole run folder to the branch as raw artifacts under `runs/`
      (CSVs + manifest + logs), following the `Log_N` / existing-run convention.
      No hand-edited responses, no hand-written conclusions.

## 8. Provenance / hygiene (CLAUDE.md)

- Open-weight model, revision pinned in `MODEL_CATALOG`
  (`Qwen/Qwen3-8B @ b968826d9c46`); the script refuses to run a model whose spec
  carries no revision (a bare name is not provenance, PREREG §3b).
- This is a plain generation record — no steering vector, no judged numbers, so
  none of the steering-claim-hygiene bar applies here. It exists to make the
  *judge* trustworthy before any steering number is read.
