# HANDOFF_prompt_control — the system-prompt control baseline on the Lambda box

**Canonical execution handoff.** It executes **needed-experiments §14** (the spec)
and the steering-hygiene bar in **AGENTS.md §5a** (a system-prompt baseline on the
same prompts is required before any steering result is reportable). It does **not**
redefine the paper, a metric, a model set, or a definition of done; if it disagrees
with the four control-plane files (`PROJECT_STATE.md`, `RESEARCH_CONTRACT.md`,
`WORK_LEDGER.md`, `docs/PREREG.md`), they win.

The **method** — the two behaviour-inducing system prompts — lives in
`src/bias_steer/config.py` (`DEFAULT_POS_SYS` / `DEFAULT_NEG_SYS`) and is recorded
verbatim in every run manifest, like a judge version. **Do not edit the prompt text
in a config file.** Changing it is a method change: edit the constants, note it
here, and treat prior runs as a different method.

---

## What this is

An `intervention` mode on `experiment.run` that scores the *same* eval prompts
under, per mode:

| `config.intervention` | Arms | Needs a vector? |
|---|---|---|
| `steer` (historical default) | INITIAL · STEERED_POS · STEERED_NEG | yes |
| `prompt` | INITIAL · PROMPT_POS · PROMPT_NEG | **no** |
| `both` | all five | yes |

`prompt` induces the target behaviour by system prompt instead of a vector.
`both` runs both interventions on the identical prompts under the identical judge,
so "did the direction beat simply *asking*?" is a **per-item paired comparison**
(`metrics.beat_rate`, item-bootstrap CI), not a difference of two marginals.

## Conventions already locked (do not re-decide)

- **§0.1 injection convention = farhan-batch-coeffs.** `steering.apply_resid_pre_add`
  already injects `coeff / n_layers × vector[layer]` at every layer, raw vector —
  the fixed-total-budget rule. Nothing to change; the steer arms use it as-is.
- **§0.2 (k-repeat judge) is intentionally NOT applied.** The judge is single-shot
  at `temperature=0, seed=0` (`JudgeSpec`). It drifts ~±1–2 labels per 100 on
  re-runs, so **do not over-read small per-item differences** — a Δ inside that band
  is not a result. This was a cost decision (k=3–5 would 3–5× the OpenAI spend);
  revisit before any headline number ships.

## Before you start

- `transformer_lens` is box-only, so anything touching `qwen3-8b` runs only on the
  Lambda box. The wiring is unit-tested off-box (`tests/test_prompt_control.py`,
  3/3); the model path is not.
- Needs `OPENAI_API_KEY` (the neutrality judge) and `HF_TOKEN`.

```bash
python -c "import transformer_lens, torch; print(transformer_lens.__version__, torch.__version__, torch.cuda.is_available())"
```

## 1. Pure prompt baseline (no vector, nothing to fetch)

The GPT opinion/comparison prompts are already in-repo
(`datasets/GPT_Prompts/all_data_1000_prompts.txt`, read by the `plain` loader).

```bash
python -m src.bias_steer run configs/prompt_baseline_opinion.py
```

Produces `runs/<id>/` with `results.csv` (INITIAL / PROMPT_POS / PROMPT_NEG per
example), `summary.md` (the "Prompt-baseline quality" block), `manifest.json` (with
the frozen system prompts), and `examples.csv`. **No `steering_vector.safetensors`
is written or required** — that is expected for a prompt-only run, not an incomplete
one.

**Sanity gate before trusting it:** in `summary.md`, the PROMPT_POS arm should be
mostly `opinionated` and PROMPT_NEG mostly `neutral`. If a prompt arm barely moves
off INITIAL, the *prompt* is weak — that is itself the finding (prompting is a weak
control here), not a bug to paper over.

## 2. Head-to-head: does the vector beat the prompt? (`both`)

Needs an existing opinion direction (a `steering_vector.safetensors` from an
`extract_*` run). Edit `configs/prompt_baseline_opinion.py`: set
`config.intervention = "both"`, then either pass `--vector` or set
`config.vector_path`.

```bash
python -m src.bias_steer run configs/prompt_baseline_opinion.py \
    --vector runs/<opinion_run>/steering_vector.safetensors
```

The supplied vector must be a `qwen3-8b (n_layers, d_model)` tensor — the shape
guard rejects a mismatch (this is the Log-213 scalar-broadcast class of bug;
AGENTS.md §6). `summary.md` gains a **"Steer vs prompt (per-item)"** section with
Δ and a 90% item-bootstrap CI per direction.

## What to log / definition of done

A run is done when its evidence exists and validates (`WORK_LEDGER.md`):
`results.csv` + `summary.md` + `manifest.json` on disk, both prompt arms
non-trivially populated (per the §1 gate), and — for `both` — the beat-rate Δ with
its CI recorded. Push the run folder under `runs/` following the `Log_N_*`
convention. Report the outcome honestly either way: **if single-direction additive
steering does not beat the prompt baseline, that IS the boundary result the
literature reports (FK-5) — log it as an honest negative, do not soften it.**
