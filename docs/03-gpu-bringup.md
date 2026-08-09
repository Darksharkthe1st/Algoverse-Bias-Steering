# GPU Bring-up & Handoff

For a Claude Code session (or a human) starting fresh on the Lambda box. It gets the
refactored pipeline running on real hardware, validates the two things that could not
be checked without a GPU, and hands off the experiment backlog mapped onto the new
architecture.

If you're a fresh Claude session: read [`00-overview.md`](./00-overview.md),
[`01-feature-roadmap.md`](./01-feature-roadmap.md), and
[`02-architecture-roadmap.md`](./02-architecture-roadmap.md) first — they are the design
of record. This doc is the *operational* on-ramp.

---

## 0. What this codebase is now

The notebook (`experiments/farhan-experimentation.ipynb`, kept **frozen** as a historical
record) was re-implemented as the `src/bias_steer/` package across four phases:

| Phase | Modules | What it gives you |
|---|---|---|
| 0 foundations | `schema`, `config`, `registry`, `tracking` | `Example`/`Result` contract, one `ExperimentConfig`, name→component registries, run IDs + `manifest.json` + `runs/index.csv` |
| 1 science | `datasets`, `models`, `steering`, `judge` | loaders→`Example`s + sampling, model load/generate, `mean_diff` steering, OpenAI neutrality judge |
| 2 wiring | `experiment`, `metrics`, `logs`, `artifacts`, `cli` | `run(config)` end-to-end, tidy `results.csv`, plaintext logs, safetensors, CLI |
| 3 payoff | `analysis/compare.py` + `stereoset` dataset + `last_token` method | standalone pandas analysis; proven plug-and-play |
| 4 batch | `coordinator` | `--queue`: route→checkout→run→commit/push per phase across branches |

Design invariants to preserve: the heavy stack (torch/transformer_lens/openai) is
**lazy-imported**, so the package imports on any machine; extend by adding a function +
one registry line (§3 of the arch roadmap); runs are **committed by default** (only
`runs/**/residuals.safetensors`, `runs/_discard/`, `_coordinator/` are gitignored).

### Verified vs. NOT verified

Everything below the ML stack was tested on a laptop (45 tests, all green). Two things
**could not run without a GPU/OpenAI and are the whole point of this bring-up**:

1. **Numerical parity with the notebook** (Phase 2 goal) — nothing has confirmed a real
   run reproduces a historical result.
2. **The live `--queue` subprocess stream** — the coordinator's per-phase commit logic is
   tested with a fake runner; the real `python -m src.bias_steer run` Popen/stdout path
   has never executed.

The torch/pandas-gated unit tests (steering math, analysis) also ran in *skip* mode on the
laptop — they execute for real here.

---

## 1. Environment

```bash
cd Algoverse-Bias-Steering
git pull                                  # get the refactor branch
python -m venv .venv && source .venv/bin/activate   # or conda; py>=3.12
pip install -e .                          # installs torch/transformer_lens/openai/safetensors/tqdm/pandas
huggingface-cli login                     # or: export HF_TOKEN=...   (gated models: gemma/llama need access)
export OPENAI_API_KEY=...                 # the neutrality judge calls gpt-4o-mini
nvidia-smi                                # confirm the GPU is visible
```

## 2. Sanity: run the test suite (now unskipped)

```bash
for t in 0 1 2 3 4; do python tests/test_phase$t.py; done   # or: pytest tests/
```
With torch + pandas present, the previously-skipped tests run for real: `capture_mean` /
`build_mean_difference` / `capture_last` math (Phase 1/3) and the pandas analysis (Phase 3).
All should pass. If a torch-gated test fails, that's a real numeric bug to fix before any run.

## 3. First real run (smoke) — exercises every untested link

```bash
python -m src.bias_steer run configs/example_bbq.py --runs-dir runs
```
This is the first time real model load, generation, residual capture, the OpenAI judge, and
safetensors saving all run together. Then inspect the output:

```bash
ls runs/*/                                # manifest.json results.csv steering_vector.safetensors logs/ summary.md
cat runs/*/summary.md
tail -f runs/*/logs/eval.txt              # live prompt-by-prompt view during a run
python -m analysis.compare runs/          # the cross-run table
```
Likely first-run friction to fix in place: a tokenizer with `bos_token=None` (guarded, but
watch it), a model whose `chat_template` flag is wrong (see §7), OOM (drop `batch_size` /
use a smaller model), or gated-model access (gemma/llama-3 need HF approval).

## 4. (highest value) Numerical parity vs. the notebook

The refactor ports the **`farhan-batch-coeffs` convention**: `coeff / n_layers` added to the
**raw** difference-of-means vector at **every** `blocks.{l}.hook_resid_pre`. (This is one of
three archived conventions — see `needed-experiments.md` §0.1.) So parity must be checked
against a run made with that convention — i.e., the frozen notebook, **not** the
`old-results` or `aryaman_*` normalized branches.

Two complementary checks:
- **Vector cosine.** Rebuild a steering vector with `run(config)` on the same model + prompt
  set a historical vector used, then cosine-compare against the archived
  `experiments/best_vecs/log_*_<model>_steer_vec.pkl` (load with torch; ≈1.0 per layer if the
  capture/build math matches).
- **Transition matrix.** Reproduce a historical `complete_test` config (same model, dataset,
  split, coeffs, `max_tokens`) and diff the `condition × verdict` counts (`metrics.condition_verdict_counts`)
  against the archived `Batched_Gen.csv` / `Log_N` for that run. Expect *close*, not identical
  — the judge (`gpt-4o-mini`, unpinned) is nondeterministic (that's exactly `needed-experiments.md`
  §0.2). A large systematic gap (e.g. mirror-imaged steering) is a real bug; the prime suspect
  is the coeff sign convention (`steered_pos=+opinion`, `steered_neg=−neutral`) or the
  response-only residual capture — both are faithful ports, documented in `models.py`/`experiment.py`.

Record the parity result (which log, cosine, count diff) in a new `docs/04-parity.md`.

## 5. `--queue` smoke (the coordinator)

```bash
mkdir -p _coordinator
cp configs/route.example.json _coordinator/route.json    # edit branch/configs as needed
python -m src.bias_steer run --queue
```
Verify: it checks out the route branch, and after each phase (`vector`, then `eval`) a commit
appears (`git log --oneline`), with a done-marker in `_coordinator/queue/done/`. `push`
soft-lands if there's no reachable `origin` (that's expected; local commits still persist).
`_coordinator/` is gitignored, so the route/queue/state never enters git — only `runs/` does.

---

## 6. Handoff: the experiment backlog → this architecture

[`needed-experiments.md`](./needed-experiments.md) (copied from Results-Retrieval; its
`../results_analysis/*` links point to *that* repo) is the backlog. The task "split up the
experiment concerns" is largely: **map each experiment onto a new registry component** — which
is cheap here because datasets/methods/judges are one-function-plus-one-line additions.

| Backlog item | Add to the architecture as… |
|---|---|
| §0.1 canonical injection / normalized+layer-band | a new **METHOD** (e.g. `norm_layerband`): override `apply` (unit-normalize per layer, single coeff, restrict to a layer band) — reuse `build`. Decide the canonical convention and record it. |
| §0.2 k=3–5 judging + majority + agreement | enhance the **judge** layer: call k times, majority-vote, record per-example agreement (extend `JudgeSpec` / `judge.py`). |
| §0.3 coherence gate | a new **judge/metric**: perplexity or a fluency yes/no pass per generation; void runs above an incoherence threshold. |
| §1 opinion spectrum (1–5) | a new **JUDGE** (`neutrality_spectrum`) with `labels=[1..5]` + the graded rubric. Metrics already tally arbitrary labels. |
| §3 BBQ bias score (ground truth) | a **non-LLM judge/metric**: an MC-answer parser + BBQ bias-score/accuracy over the `label`/`context_condition` already preserved in `Example.metadata`. |
| §4 rubric-v2 + κ gate | a new **JUDGE** (frozen rubric) + an annotation/κ step in `analysis/`. |
| §5 combined vs single dataset | orchestration: build vectors per dataset + combined, eval on one held-out set (loop over configs). |
| §6 normalization ablation | falls out of the §0.1 METHOD — run `mean_diff` vs `norm_layerband` on one eval set. |
| §7 coeff/layer ablation | a config sweep over coeff + layer band (METHOD param); log the coherence signal. |
| §8 cross-model transfer | `steering.apply` variant handling dim/layer mismatch across models. |
| §9 Grok questions | a **DATASET** loader (`grok`) over `datasets/Grok_Questions/*.txt`. |

**Do §0 first** (it blocks most comparisons): lock the injection convention, judge-reliability
(k-vote), and coherence gate — each is a small, isolated component in this design. The HIGH-priority
science (spectrum judge §1, CrowS completion §2, BBQ ground-truth §3) then plugs into the frozen
conventions.

## 7. Known gotchas (faithful ports — verify, don't assume)

- **Per-model `chat_template` flags** (`models.py:MODEL_CATALOG`) mirror the notebook: qwen/yi
  `True`, gemma/llama-3 `False` (no system prompt). Confirm that's still intended per model.
- **Residuals are captured over the *response* text only** (not prompt+response) — faithful to
  `batch_resids`. Commented in `models.py:generate_with_cache`; revisit if the science should differ.
- **Coeff signs**: `steered_pos = +opinion_coeff`, `steered_neg = −neutral_coeff`, vector =
  `mean(pos) − mean(neg)` with pos = `judge.labels[1]`. First suspect if steering looks mirrored.
- **`gpt-4o-mini` is an unpinned alias** — pin a dated version for any result you'll defend (§0.2).
- **Gated models** (gemma, llama-3) need HF access approval on the token you log in with.
