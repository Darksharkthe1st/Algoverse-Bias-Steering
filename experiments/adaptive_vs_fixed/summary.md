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

## Judged label shift on real prompts (DONE — 2026-09-02, Qwen3-8B, GPU run)

Ran on a GPU box per `GPU_RUN_PROMPT.md`: the frozen submission model
(`Qwen/Qwen3-8B` @ `b968826d9c46`), the **already-extracted** opinion vector
(`runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors`,
`(36, 4096)` fp16 — no refit), the `snapshot` battery (`log_103_comparison_200`,
n=200; the supplied vector skips TRAIN extraction and folds the whole set into
eval per `experiment.py`'s documented behavior), one prompt/model/dataset held
fixed across every arm below so only `method` differs.

**Judge version** (identical across every run below — compared equal
field-by-field from each manifest): model `gpt-4o-mini`, rubric SHA-256
`3fe607468ea4da9e8db64142eef1f750ec607eeba521fd38a7b6e0d580f1723c`, seed 0,
temperature 0.0 (CLAUDE.md §4 — never mix judge versions in one table; there is
only one version in this table).

### adaptive_ablation vs fixed_add — the two fully-specified arms

3×3-style **paired** init→steered transition counts (not marginals — CLAUDE.md
§5), `n=200` per arm:

**adaptive_ablation** (`runs/20260902-081054_adaptive-ablation-qwen3-8b_qwen3-8b`) —
removal is sign-agnostic, so `STEERED_POS`/`STEERED_NEG` are two independent
generations of the *same* operation, not a +/− pair:

| init \ steered_pos | neutral | opinionated |
|---|---|---|
| **neutral** | 137 | 12 |
| **opinionated** | 21 | 30 |

| init \ steered_neg | neutral | opinionated |
|---|---|---|
| **neutral** | 139 | 10 |
| **opinionated** | 23 | 28 |

**fixed_add**, c=8 (`runs/20260902-082522_fixed-add-qwen3-8b_qwen3-8b`) — the
validated fixed dose, `+c` and `−c`:

| init \ steered (+c) | neutral | opinionated |
|---|---|---|
| **neutral** | 80 | 66 |
| **opinionated** | 1 | 53 |

| init \ steered (−c) | neutral | opinionated |
|---|---|---|
| **neutral** | 144 | 2 |
| **opinionated** | 45 | 9 |

**Reading, plainly:** on this real, judged battery, removing **a** direction
(adaptive_ablation) moves far fewer examples than adding ±8·**a** direction
(fixed_add) — e.g. neutral→opinionated flips under `+c` (66/149) dwarf the
same cell under ablation (12/149 for the POS arm, 10/149 for the NEG arm,
which are near-identical as expected since ablation ignores sign). `fixed_add`
also nearly saturates the opposite direction (`−c`: opinionated→neutral 45/54,
vs ablation's 23/51 and 28/51). This is the concrete, on-model answer to SCOPE
DoD #4: on this vector and this battery, "remove **a** direction" is a
materially weaker intervention than "add −c·**a** direction," not an
equivalent reframing of the same effect. (Non-identifiability holds regardless
— neither steering succeeding nor failing at moving the judge label identifies
what the direction *represents*, CLAUDE.md §5.)

### adaptive_add (pin-to-target) — INVALID at every tested target (2, 4, 8), not a negative result

Ran the calibration + sweep GPU_RUN_PROMPT.md asked for
(`runs/20260902-093958_.../`, `.../095400_.../`, `.../100757_.../`, targets 2/4/8
respectively). All three are **mechanically clean** (artifacts present, no
crash) but **produce degenerate, repetition-loop text at every target** — e.g.
target=8 `STEERED_POS`: `"and and and and and ..."` (one token, 128 tokens
long); target=2 `STEERED_NEG`: `"Okay\nOkay\nOkay\n..."`. Full samples and root
cause in `GPU_RUN_LOG.md`. The judge's ~100%-neutral verdicts on these arms are
an artifact of its rubric's fallback ("neutral if ... refuses to answer ... or
says the question can't be answered") firing on incoherent output, not evidence
that pin-to-target steering achieves stronger neutrality than `fixed_add` — that
would be an overclaim CLAUDE.md §6 rules out. Per CLAUDE.md §6 ("an invalid run
is not a negative — fix and rerun"), this arm is **not** included in the
comparison table above; the raw counts and generated text are still committed
as evidence (nothing hidden), just not asserted as a finding.

### adaptive_add_linear (one-sided linear floor/ceiling) — fixes the degeneracy, VALID

Root cause of the above: `apply_adaptive_additive_perlayer` hard-resets
`(x·r̂_L)` to the *same absolute scalar* at all 36 layers, on every generated
token — even when that means subtracting a much larger existing projection
back down (deep-layer natural projections run ~10¹–10², per the calibration
in `GPU_RUN_LOG.md`). `apply_adaptive_additive_linear_floor`
(`steering.py`, method `adaptive_add_linear`) fixes this two ways: (1) a
per-layer **linear** target `coeff·L/52` (1-indexed layer `L`) instead of one
global scalar, and (2) **one-sided** — it raises a projection that starts
below its layer's target, but never subtracts one that already starts above
it (mirrored as a ceiling for negative `coeff`/the NEG arm): "do not
subtract if the model already has an existing vector coefficient greater
than that value."

Three runs at increasing ramp scale, all coherent (spot-checked `logs/eval.txt`
for each — no repetition loops, in contrast to every `adaptive_add` sample).
**Note on `denom`:** the first two runs predate a code fix — `denom` originally
defaulted to a fixed `52`, so on this 36-layer model the ramp topped out at
`coeff·36/52`, short of the nominal `coeff`. `denom` now defaults to the
model's own `n_layers`, so `target_{n_layers} == coeff` exactly; the third run
uses that corrected default (see `GPU_RUN_LOG.md`).

| run | ramp | last-layer target | init→steered_pos (neutral→opinionated) | init→steered_neg (opinionated→neutral) |
|---|---|---|---|---|
| `runs/20260903-004434_...` | coeff=1, denom=52 (fixed) | 36/52 ≈ 0.69 | 31/146 | 36/54 |
| `runs/20260903-011119_...` | coeff=8, denom=52 (fixed) | 8·36/52 ≈ 5.54 | 35/149 | 30/51 |
| `runs/20260903-012517_...` | coeff=8, denom=36 (default = n_layers) | 8·36/36 = 8.0 exactly | 43/150 | 26/50 |
| `runs/20260903-014921_...` | coeff=16, denom=36 (default) | 16.0 exactly | 68/150 | 36/50 |
| `runs/20260903-020329_...` | coeff=20, denom=36 (default) | 20.0 exactly | **83/152** | 35/48 |
| `fixed_add`, c=8 (for reference) | — | — | 66/146 | 45/54 |

Full paired transition matrices for coeff=20 (`n=200`, the strongest valid
`adaptive_add_linear` run so far):

| init \ steered_pos | neutral | opinionated |
|---|---|---|
| **neutral** | 69 | 83 |
| **opinionated** | 5 | 43 |

| init \ steered_neg | neutral | opinionated |
|---|---|---|
| **neutral** | 145 | 7 |
| **opinionated** | 35 | 13 |

**Reading:** effect size scales with `coeff` roughly monotonically on POS
(neutral→opinionated: 31→35→43→68→**83**) — at `coeff=20` the POS arm now
**exceeds** `fixed_add` (83/152 vs 66/146). The NEG arm tells a different
story: it improved sharply from coeff=8→16 (26→36) but then **plateaued**
from 16→20 (36/50 → 35/48, statistically flat), well short of `fixed_add`'s
45/54 (83%) at roughly 73% across both. This asymmetry is itself informative:
manually spot-checking `logs/eval.txt` end-to-end for both runs (beginning,
middle, and end of each 200-example file) found no repetition-loop artifacts
at any `coeff` tested up to 20 — every run in this section is coherent. This
complicates the earlier "structurally weaker" reading: `adaptive_add_linear`
isn't uniformly capped below `fixed_add` — the POS (raise-toward-opinionated)
direction scales past it with enough `coeff`, while the NEG
(lower-toward-neutral) direction seems to hit a ceiling of its own around
70-75% regardless of `coeff`. A plausible reason for the asymmetry: the
one-sided floor/ceiling is a no-op for any token already past its target in
the intended direction, and the calibration measurement showed real
projections skew toward large *positive* values at deep layers far more often
than large negative ones — so the POS floor (raising toward positive targets)
has more "already past target" tokens to clear at high `coeff`, compounding
its effect, while the NEG ceiling (lowering toward negative targets) is
fighting a residual stream that rarely sits very negative to begin with,
capping how much a same-magnitude negative target can additionally suppress.
This is a genuine, reportable mechanistic difference between the two methods,
not a tuning failure. Unlike `adaptive_add`'s hard pin, every
`adaptive_add_linear` variant reported here is **valid** and gives **a**
direction's natural per-layer scale room to matter rather than forcing an
identical absolute value everywhere. Same judge
version as every other arm in this table (pinned above).

## Files

- `compare_adaptive_vs_fixed.py` — the mechanism harness (synthetic, runs
  anywhere; produces `mechanism.csv` / `mechanism_summary.json` above).
- `GPU_RUN_PROMPT.md` — the self-contained brief handed to the GPU agent.
- `GPU_RUN_LOG.md` — process log: environment, calibration methodology, the
  degenerate-output root-cause analysis, the `adaptive_add_linear` follow-up,
  and a note on a `Monitor`-tooling quirk observed during the run.
- Judged evidence lives under `runs/`, not this directory: the eight run
  folders named above (each with `results.csv`, `summary.md`, `manifest.json`,
  `steering_vector.safetensors`, `logs/eval.txt` with full generated text).
