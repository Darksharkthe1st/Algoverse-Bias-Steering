# Refusal-direction reproduction (arXiv:2406.11717)

Reproducing *"Refusal in Language Models Is Mediated by a Single Direction"*
(Arditi et al., 2024) inside this framework. **This is a two-track effort and the
Lambda run is meant to execute BOTH tracks:**

1. **Refusal (load) track** — apply the paper's *published* direction vectors,
   run their interventions, score refusal, and diff against their committed
   numbers. Lives on branch **`fk/init-refusal-rewrite`** (this doc's branch).
2. **Generation track** — re-derive the direction vectors ourselves
   (extraction + selection) and validate against the published `mean_diffs.pt`.
   Lives on branch **`fk/refusal-generate`**; its own guide is
   **`docs/06-refusal-generation.md`** on that branch.

> The two branches were split for parallel development and share Chunks 0–2
> (fetch infra, direction loader, ablation/act-add methods). They must be
> **merged (or both checked out) on the Lambda box** so all commands below work
> from one tree. One reconciliation note: the generation track's
> `select_direction` uses the paper's **logit-based** `refusal_score`, NOT the
> substring metric from the load track — they are different metrics for
> different jobs (selection vs. jailbreak-eval). Don't unify them.

Model execution (both tracks) is intended for a GPU box (your Lambda instance);
everything else is CPU-only and already validated in CI-style standalone tests.

---

## 0. Environment (Lambda / GPU box)

```bash
pip install -e .                       # the package
pip install torch transformer_lens     # the ML stack (CUDA build)
export HF_TOKEN=...                     # for gated model downloads (Llama/Gemma)
# NO OpenAI key needed — refusal is scored by deterministic substring match.
```

Fetch the third-party artifacts (pinned to andyrdt/refusal_direction@9d852fa;
git-ignored, ~54 MB + dataset splits):

```bash
python scripts/fetch_refusal_artifacts.py            # directions, mean_diffs, completions, evals
# splits (generation track) come with the same script once its manifest entries are merged in
```

Sanity-check the fetch with the CPU-only tests:

```bash
python3 tests/test_refusal.py                        # load track (27 tests)
python3 tests/test_refusal_grid_provenance.py        # generation track (from fk/refusal-generate)
python3 tests/test_refusal_datasets.py
python3 tests/test_refusal_extract.py
```

---

## 1. Refusal (load) track — run this

Applies the published direction for each model and runs the paper's five arms
(harmful: baseline · ablation · act-add(−); harmless: baseline · act-add(+)),
scores refusal by substring match, and auto-diffs against the paper.

```bash
python -m src.bias_steer refuse configs/refusal_repro.py
```

Default config runs **qwen-1.8b**. To add models, edit `configs/refusal_repro.py`
(`models=[...]`) — `qwen-1.8b`, `yi-6b`, `llama-2-7b` are `chat_template=True` and
work directly; **gemma-2b and llama3-8b are `chat_template=False`** in the catalog
(a legacy quirk) and need per-model template handling before their numbers are
trustworthy — defer them until qwen is confirmed.

### What to expect

Output (and `runs/<run_id>/summary.md`) includes an our-vs-paper table. Target
refusal rates from the paper (what we should land within ±0.05 of):

| model | harmful/baseline | harmful/ablation | harmful/actadd | harmless/baseline | harmless/actadd |
|---|---|---|---|---|---|
| gemma-2b-it | 0.91 | 0.00 | 0.02 | 0.00 | 1.00 |
| llama-2-7b-chat | 0.97 | 0.07 | 0.03 | 0.01 | 1.00 |
| llama-3-8b-instruct | 0.95 | 0.00 | 0.01 | 0.01 | 1.00 |
| qwen-1.8b-chat | 0.70 | 0.01 | 0.03 | 0.03 | 0.98 |
| yi-6b-chat | 0.62 | 0.02 | 0.02 | 0.05 | 0.88 |

**Reading it:** ablation should collapse harmful refusal toward 0; act-add(+)
should push harmless refusal toward 1. If a cell is off, see §3.

---

## 2. Generation track — also run this

From `fk/refusal-generate` (see its `docs/06-refusal-generation.md` for detail):

```bash
# Anchor: extract our candidate grid and cosine-check vs the published mean_diffs.pt
python scripts/refusal_extract_check.py --model qwen-1_8b-chat
#   expect per-cell cosine ~0.999 (>=0.95 = investigate, <0.95 = debug formatting/positions).
#   NOTE: replicates upstream `filter_train` (default on) — a filtering forward pass runs first.

# Then reproduce select_direction and check it lands on the published cell (pos -2, layer 15 for qwen)
python scripts/refusal_select_direction.py --model qwen-1_8b-chat
```

If the cosine anchor passes, extraction faithfully reproduces the paper's recipe;
selection landing on the published `(layer, pos)` closes the loop.

---

## 3. Correctness guards & known fidelity caveats

**Silent-bug guards (already in code).** `steering.check_direction` runs before
any direction is applied and raises a clear error rather than silently
misbehaving when:
- a direction's length ≠ `model.cfg.d_model` (wrong model/direction pairing),
- a `(n_layers, d_model)` bias-steering stack or `(n_pos, n_layers, d_model)`
  grid is passed where a single `(d_model,)` direction is expected,
- the direction has NaN/Inf, or an act-add `layer` is out of range.

`refusal.load_refusal_direction` additionally rejects a malformed `direction.pt`
(non-1-D, empty, non-finite, or zero-norm) at load time.

**Basis assumption.** The published directions are raw-HF-residual-stream vectors;
we load models with TransformerLens `from_pretrained_no_processing`, which keeps
that same basis (no LN folding/centering). This is *why* the vectors transfer.
It can't be asserted from shape alone — the cosine anchor (§2) and the
refusal-rate match (§1) are the empirical confirmations.

**Chat template / no system turn.** The paper formats prompts with the model's
chat template and **no system turn**. `models.render_prompts` currently emits an
empty system turn (`system_prompt=""`), and the generation track uses hardcoded
per-model templates with `system=None`. If a §1 cell is off (especially
baselines), reconcile the eval-side template to omit the system turn — this is
the most likely source of a mismatch.

---

## 4. What "done" looks like

- **Load track:** all five arms per model within ±0.05 of the table in §1.
- **Generation track:** cosine ≥ ~0.999 vs `mean_diffs.pt`, and `select_direction`
  recovers the published `(layer, pos)` per model.

Report both tables. If the load track matches but generation cosine is low, the
issue is our extraction recipe; if generation matches but the load-track rates
are off, the issue is eval-side formatting (§3), not the vectors.

---

## 5. Results

Run outcomes live in [`docs/findings/`](./findings/), not here — this file is the
how-to-run guide and stays stable across runs.

- **2026-08-16, qwen-1.8b, both tracks** —
  [`findings/2026-08-16-refusal-repro-qwen-1.8b.md`](./findings/2026-08-16-refusal-repro-qwen-1.8b.md).
  Load track 4/5 arms in tolerance (harmful/baseline 0.380 vs 0.700); extraction
  cosine 0.900 vs the 0.999 target; `select_direction` recovers layer 15 but
  position −1 vs the published −2. Eight candidate causes eliminated with evidence.

  **Note for §3 readers:** the system-turn hypothesis in §3 was tested and is
  **not** the cause — it is worth only ~0.01 on the failing arm. Eval now uses the
  paper's literal template regardless (it is the correct formatting), but do not
  re-spend time there. See the findings doc for what remains.
