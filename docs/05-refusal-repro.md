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

## 5. First Lambda run — 2026-08-16 (qwen-1.8b)

**Not done yet.** Both tracks ran end to end; neither fully meets §4.

### Load track (`runs/20260816-011914_refusal-repro_qwen-1.8b`)

| arm | ours | paper | Δ | within ±0.05 |
|---|---|---|---|---|
| harmful/baseline | 0.380 | 0.700 | −0.320 | ✗ |
| harmful/ablation | 0.000 | 0.010 | −0.010 | ✓ |
| harmful/actadd | 0.000 | 0.030 | −0.030 | ✓ |
| harmless/baseline | 0.010 | 0.030 | −0.020 | ✓ |
| harmless/actadd | 0.950 | 0.980 | −0.030 | ✓ |

4/5. The paper's *qualitative* claim reproduces cleanly — ablation collapses
harmful refusal to 0.00, act-add(+) drives harmless refusal to 0.95. Only the
un-intervened harmful baseline is off, and it is off in the direction of the
model refusing *less* than the paper's.

### Generation track

- Extraction cosine at the published cell (pos −2, layer 15): **0.900**
  (0.892 with `--no-filter`). Target ≥0.999, acceptable ≥0.95 → **below threshold**.
  Mean over all cells 0.642 (0.675 excluding layer 0, which is identically zero in
  the paper's grid too — not a bug on our side).
- `select_direction` over *their* `mean_diffs.pt`: selected **(pos −1, layer 15)**
  vs published (pos −2, layer 15). Layer exact, position off by one.

### §3's hypothesis was wrong — do not re-try it

§3 named the eval-side system turn as the most likely cause of a baseline
mismatch. It was reconciled (commit `6db750a`): eval now prompts with the paper's
literal `REFUSAL_TEMPLATES` string and no system turn, verified token-identical to
the paper's 15-token prompt. **Measured effect: 0.370 → 0.380.** The remaining
0.32 is something else.

Causes eliminated, with evidence (scripts + logs under the run's `diagnostics/`):

| suspect | verdict | evidence |
|---|---|---|
| empty system turn | ruled out | fixed it; +0.010 |
| BOS token prepended | ruled out | Qwen `default_prepend_bos=False`; ids match paper exactly |
| batched left-padding | ruled out | batch=16 vs batch=1: **0/40** verdict flips |
| `max_tokens=128` | minor, real | truncating *the paper's own* completions to 128 tok costs 0.020 (0.700→0.680) |
| substring judge | ruled out | reproduces the paper's committed labels, 2500 labels, 0 mismatches |
| `filter_train` | ruled out | selected-cell cosine moves <0.01 |
| fp16 vs fp32 forward | ruled out | cosine 0.89165 vs 0.89164 — identical to 5 dp |
| position/layer axis misalignment | ruled out | every pos and every layer best-matches its own index |

**Strongest remaining lead.** Our mean-diff vectors are consistently **0.80–0.84×
the paper's norm at every layer** while pointing ~0.89 cosine the same way. Same
direction, diluted magnitude is the signature of averaging over a *different set
of instructions* — not of a numerics or formatting bug. Check
`load_and_sample_repro`'s selection against upstream's actual train set, and the
`refusal_score` filter, which keeps only 71/128 harmful (aggressive enough to be
worth verifying against upstream's kept count).

Note this cuts across §4's decision rule: *both* tracks are off, so the shared
cause sits upstream of both (which instructions/activations feed the mean), not in
eval-side formatting — that has now been largely eliminated.

Cheap fidelity fix worth making before the next run: raise `max_tokens` from 128
to the paper's 512 in `configs/refusal_repro.py` (worth ~0.02).
