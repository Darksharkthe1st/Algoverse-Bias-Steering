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
Δ and a 90% item-bootstrap CI per direction, plus the paired 2×2 cells (see §2.1).

### 2.1 Read the complementarity, not just Δ — the expected result here

**Prior (2025, unpushed) result on these opinion prompts: the prompt baseline was
roughly as good as the steering vector *on aggregate*, but the two methods won on
DIFFERENT items — vector steering flipped some prompts to the target that prompting
did not, and prompting flipped others that steering did not.** That is
complementarity, and it is the headline of this experiment, not a footnote.

The net Δ (`beat_rate.point`, mean of steer-hit − prompt-hit) **cannot show this** —
Δ≈0 is equally consistent with "both methods succeed on the same items" and "each
succeeds on a disjoint subset." So the summary's per-item line reports the paired
2×2 explicitly: `both · steer-only · prompt-only · neither`, with
`discordant = steer-only + prompt-only`. Read it this way:

- **large `discordant`, small Δ → complementary** (the expected 2025 finding): the
  methods are NOT interchangeable; report the split, not "no difference."
- **small `discordant` → concordant**: whichever wins Δ wins nearly item-for-item.

Do **not** collapse a complementary result into "prompting is just as good." The
claim the paper can make is per-item: *which* prompts each method moves, and how
much they overlap — that is what `steer-only`/`prompt-only` quantify. (Judge drift
is ±1–2/100 single-shot — §0.2 — so treat discordant counts near that band as noise.)

## What to log / definition of done

A run is done when its evidence exists and validates (`WORK_LEDGER.md`):
`results.csv` + `summary.md` + `manifest.json` on disk, both prompt arms
non-trivially populated (per the §1 gate), and — for `both` — the beat-rate Δ with
its CI **and the paired 2×2 cells (§2.1)** recorded. Report the outcome honestly
either way: **if single-direction additive steering does not beat the prompt
baseline on Δ, that IS the boundary result the literature reports (FK-5) — log it as
an honest negative, do not soften it — and if the two are complementary (§2.1), that
per-item split is the result, not a null.**

### Commit and push — non-negotiable (this run was lost once already)

**Every change and every artifact MUST be committed and pushed to GitHub, on a
branch, before the box is released.** The 2025 run of this exact experiment was
performed on the GPU box and **never pushed** — the results, and the finding above,
were lost with the box. Do not repeat it. Concretely, before you stop:

1. Push the run folder under `runs/` (the `Log_N_*` convention) — `results.csv`,
   `summary.md`, `manifest.json`, `examples.csv`.
2. Commit and push **any code, config, or doc change you made to get the run to
   work** — a fix is not done until it is on GitHub (AGENTS.md §7: push runs to
   branches; no hand-edited conclusions).
3. Launch long jobs **detached** from the session (see `fk/qwen3-8b-opinion-vector`)
   and push incrementally, so a dropped session cannot strand the outputs on the box.

A run whose evidence exists only on the box does **not** satisfy the definition of
done — it is indistinguishable from a run that never happened.
