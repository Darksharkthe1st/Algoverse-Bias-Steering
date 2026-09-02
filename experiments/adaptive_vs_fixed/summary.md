# Adaptive ablation vs fixed-coeff additive steering

**Scope:** `docs/SCOPE_adaptive_steering.md`, Definition of Done #4 — compare, on a
handful of prompts, the judge-label shift from **adaptive ablation**
(`adaptive_ablation`, `x ← x − (x·r̂_L) r̂_L`) vs **fixed-coeff additive steering**
(`apply_resid_pre_add`, `x ← x + (c/n_layers) r`), to see whether "remove a
direction" and "add −c·direction" behave differently.

Both operate on the **`(n_layers, d_model)`** per-layer stack; each layer uses its
own row as that layer's direction. This is deliberately kept separate from the
single `(d_model,)` refusal convention (CLAUDE.md §6). "**a** direction," not
"the" — removing a direction that steers behavior does not identify the
representation (non-identifiability, CLAUDE.md §5).

## What is computed here (mechanism — runs anywhere)

Reproduce: `python experiments/adaptive_vs_fixed/compare_adaptive_vs_fixed.py`
→ `mechanism.csv`, `mechanism_summary.json` (committed).

On a synthetic residual stream with a known per-layer direction (`n_layers=6`,
`d_model=32`, `c=8`, seed 0), tracking the projection of each token's residual
onto that layer's unit direction `r̂_L`, before vs after each method:

| quantity | adaptive ablation | fixed-coeff add |
|---|---|---|
| post-intervention projection onto `r̂_L` | **0.0** for every token | pre + a fixed per-layer shift |
| does the *effect size* adapt per token? | **yes** — removed amount varies across tokens within a layer (spread **3.94**) | **no** — within-layer shift across tokens is constant (spread **2e-6**, float noise) |

The two are structurally different operations, independent of any model:

- **Adaptive ablation** sets the component along `r̂_L` to **0** for *every* token —
  the coefficient is that token's own dot product `(x·r̂_L)`, computed in the hook,
  so it adapts. No dose, no coeff sweep.
- **Fixed-coeff add** shifts the projection by the **same amount** for every token
  at a layer — `(c/n_layers)·‖vector[L]‖`, independent of where the token started.
  A token already far along the direction is pushed further; ablation would instead
  zero it. This is the concrete sense in which "remove" ≠ "add −c·direction": add is
  a uniform translation of the component; ablate is a projection to zero.

## What is NOT computed here (judged label shift — PENDING)

DoD #4 also asks for the **judge-label** shift on real prompts. That needs a GPU
(to load a model) **and** an `OPENAI_API_KEY` (the `neutrality` judge calls
OpenAI) — neither was available in the authoring environment, so the judged table
below is **PENDING**, not reported. Per CLAUDE.md §3 no number is asserted from a
run that did not happen.

Produce it on a GPU box with a judge key:

```
export OPENAI_API_KEY=...
python experiments/adaptive_vs_fixed/compare_adaptive_vs_fixed.py \
    --run-model --model qwen-1.8b --coeff 8
# writes judged_result.json + judged_transitions.csv into this directory
```

The harness runs the same handful of prompts through **INITIAL**,
**adaptive-ablation**, and **fixed-add (−c)**, judges each, and writes a per-method
init→steered transition matrix plus one comparable per-prompt table. Every judged
output records the **judge version** — judge model id + a SHA-256 hash of the
rubric text (CLAUDE.md §4) — so the numbers cannot be silently mixed across judge
versions. Interpretation guidance for whoever runs it:

- Report the **3×3 transition counts** (per-example distribution), not just a mean
  rate (CLAUDE.md §5, steering-claim hygiene).
- Note a structural artifact of ablation: it has **no dose and no sign**, so a
  standard `run()` produces **identical `STEERED_POS` and `STEERED_NEG`** arms for
  `adaptive_ablation` (the registered method ignores its coeff). Compare ablation's
  single steered arm against fixed-add's `+c` / `−c` arms accordingly.

## Files

- `compare_adaptive_vs_fixed.py` — the harness (mechanism default; `--run-model` judged).
- `mechanism.csv` — per (layer, token): pre / post projections + shifts.
- `mechanism_summary.json` — the aggregate numbers in the table above.
- `judged_result.json`, `judged_transitions.csv` — written only by `--run-model` (PENDING).
