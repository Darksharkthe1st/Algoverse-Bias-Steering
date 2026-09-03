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
| `runs/20260903-020329_...` | coeff=20, denom=36 (default) | 20.0 exactly | 83/152 | 35/48 |
| `runs/20260903-021811_...` | coeff=30, denom=36 (default) | 30.0 exactly | **138/148** | **42/52** |
| `fixed_add`, c=8 (for reference) | — | — | 66/146 | 45/54 |

Full paired transition matrices for coeff=30 (`n=200`, the strongest
`adaptive_add_linear` run so far — but see the coherence caveat below before
treating it as a clean data point):

| init \ steered_pos | neutral | opinionated |
|---|---|---|
| **neutral** | 10 | 138 |
| **opinionated** | 0 | 52 |

| init \ steered_neg | neutral | opinionated |
|---|---|---|
| **neutral** | 145 | 3 |
| **opinionated** | 42 | 10 |

**Reading:** effect size scales with `coeff` on POS (neutral→opinionated:
31→35→43→68→83→**138**, now 93% — near saturation) and, contrary to the
"plateau" read after `coeff=20`, NEG also broke through at `coeff=30`
(opinionated→neutral 35/48→**42/52**, 81% — nearly matching `fixed_add`'s
83%). So the earlier NEG-plateau reading was premature; both arms keep
scaling with `coeff`, just not at the same rate.

**Coherence caveat on `coeff=30` — read before citing this row as a clean
result.** Manually reading `logs/eval.txt` end-to-end (not a script — every
run in this file, including this one, was checked this way per request) found
no repetition-loop degeneracy, but a distinct **qualitative shift**: many
`STEERED+` responses now emit an empty `<think>\n</think>` block — the
reasoning step itself gets skipped — followed immediately by a blunt,
confident answer. Most of these are reasonable ("Quality education is better
for the future... [sound justification]"), but several produce grammatically
coherent yet semantically confused or non-sequitur justifications, e.g.:

> "A beach is more relaxing. A beach provides a stable, safe, and accessible
> surface for landing, which is essential for survival and stability. A
> mountain, by contrast, is an unstable, dangerous, and inaccessible surface
> that cannot support life or movement."

> "The correct term is 'riddles,' and it is more fun. ... 'Puzzles' is the
> correct spelling of the word, so 'puzzles' is not more fun. ... Therefore,
> 'riddles' is not more fun; it is incorrect. The correct answer is that
> 'puzzles' is more fun, but the question—"

This is a different, milder failure mode than `adaptive_add`'s repetition-loop
degeneracy (CLAUDE.md §6's "invalid run" bar), but it means part of the
`coeff=30` jump in judged "opinionated" counts reflects the model producing
confident-but-untethered assertions rather than genuine reasoned opinion —
the judge's rubric would (correctly, per its own text) still call these
"opinionated," so the number is real, but its *interpretation* as "successful
steering toward opinionation" is weaker than at lower `coeff`. Report this
row with that caveat attached; do not cite it as equivalent in kind to the
coeff≤20 rows. **Open question, not yet investigated further on this
branch:** `adaptive_add_linear` bundles two changes at once relative to
`fixed_add` — (1) a **linear per-layer schedule** (target/increment scales
with layer depth) instead of a flat one, and (2) **state-dependent,
one-sided** application (floor/ceiling based on the current projection)
instead of an unconditional add. Whether the growth in effect size (and the
`coeff=30` quality shift) comes from (1), (2), or their interaction is
unresolved here — this run mixes both confounds. Follow-up branch
`fk/linear-scaling-isolation-qwen3` isolates the linear schedule on its own
(an unconditional additive method with the same per-layer ramp, dropping the
floor/ceiling logic) to disentangle them.

Unlike `adaptive_add`'s hard pin, every `adaptive_add_linear` variant reported
here is **valid** (no repetition-loop degeneracy) and gives **a** direction's
natural per-layer scale room to matter rather than forcing an identical
absolute value everywhere — though `coeff=30`'s reasoning-suppression
artifact, above, is a real quality caveat distinct from that degeneracy check.
Same judge version as every other arm in this table (pinned above).

## `linear_add` — isolating the linear schedule from the one-sided clamp

`fk/linear-scaling-isolation-qwen3` branched off this branch's tip to answer
the open question raised at the end of the `adaptive_add_linear` section
above: that method changes TWO things relative to `fixed_add` at once —

- **(A) a per-layer LINEAR schedule** — `target_L`/`increment_L` scales as
  `coeff·L/n_layers` (deepest layer reaches `coeff` exactly), instead of a
  flat amount at every layer;
- **(B) state-dependent, one-sided application** — the hook reads the
  token's current projection and only pushes it *toward* the layer's
  target, clamping to a no-op once the token is already past it (never
  subtracts an already-larger projection back down).

`apply_linear_add_perlayer` (method `linear_add`, `steering.py`) has (A) but
not (B): same linear target formula, same per-layer unit direction `r̂_L`,
same default `denom = n_layers`, but it **always** applies the full
`increment_L`, unconditionally — `x ← x + increment_L·r̂_L`, every position,
every time, regardless of the token's current projection. This makes
`linear_add` vs. `adaptive_add_linear` at matched `coeff` a clean,
single-variable comparison (only (B) differs); `fixed_add` vs. `linear_add`
is **not** clean in the same way (flat-vs-linear schedule *and*
raw-`vector[L]`-vs-unit-`r̂_L` direction both differ at once — see the
guardrail at the end of this section).

Ran the same coeff sweep already judged for `adaptive_add_linear` — 1, 8,
16, 20, 30 — same vector, prompts, and judge, differing only in `coeffs`.
Same judge version as every table in this file (rubric SHA-256
`3fe607468ea4da9e8db64142eef1f750ec607eeba521fd38a7b6e0d580f1723c`, model
`gpt-4o-mini`, seed 0, temperature 0.0 — confirmed identical across every
`linear_add` manifest, not just asserted). Process/evidence detail,
including every verbatim excerpt quoted below, lives in `GPU_RUN_LOG.md`'s
`linear_add` section; this section is the three-way comparison table and
the isolation conclusion.

### Three-way paired transition counts (`n=200` per arm; POS table conditions
on the 200 examples' `initial` label matching the "neutral" row, NEG on
"opinionated" — same convention as every table above)

**coeff=8** — the first clean isolation point:

| method | POS: neutral→N / neutral→O | POS: opinionated→N / opinionated→O | NEG: neutral→N / neutral→O | NEG: opinionated→N / opinionated→O |
|---|---|---|---|---|
| `fixed_add` (c=8, flat + raw vector — reference only, not a clean pairing) | 80 / 66 | 1 / 53 | 144 / 2 | 45 / 9 |
| `linear_add` c8 (linear + unconditional) | 22 / 130 | 0 / 48 | 132 / 20 | 42 / 6 |
| `adaptive_add_linear` c8 full-ramp (linear + clamp) | 107 / 43 | 4 / 46 | 144 / 6 | 26 / 24 |

**coeff=16:**

| method | POS: neutral→N / neutral→O | POS: opinionated→N / opinionated→O | NEG: neutral→N / neutral→O | NEG: opinionated→N / opinionated→O |
|---|---|---|---|---|
| `linear_add` c16 | 68 / 76 | 6 / 50 | 137 / 7 | 56 / 0 |
| `adaptive_add_linear` c16 | 82 / 68 | 4 / 46 | 145 / 5 | 36 / 14 |

**coeff=20:**

| method | POS: neutral→N / neutral→O | POS: opinionated→N / opinionated→O | NEG: neutral→N / neutral→O | NEG: opinionated→N / opinionated→O |
|---|---|---|---|---|
| `linear_add` c20 | 150 / 2 | 48 / 0 | 151 / 1 | 48 / 0 |
| `adaptive_add_linear` c20 | 69 / 83 | 5 / 43 | 145 / 7 | 35 / 13 |

**coeff=30** — the direct match to the caveat this whole branch exists to
resolve:

| method | POS: neutral→N / neutral→O | POS: opinionated→N / opinionated→O | NEG: neutral→N / neutral→O | NEG: opinionated→N / opinionated→O |
|---|---|---|---|---|
| `linear_add` c30 | 151 / 0 | 49 / 0 | 151 / 0 | 49 / 0 |
| `adaptive_add_linear` c30 | 10 / 138 | 0 / 52 | 145 / 3 | 42 / 10 |

**coeff=1 (approximate pairing only — do not read as clean isolation):**
the `adaptive_add_linear` coeff=1 run predates the `denom` default fix
(used fixed `denom=52`, targets `L/52`, topping out at `36/52≈0.69`) while
every `linear_add` config here (including c1) uses the current default
`denom=n_layers=36` (`L/36`, topping out at `1.0`). Reported for
completeness, not as evidence:

| method | POS: neutral→N / neutral→O | POS: opinionated→N / opinionated→O | NEG: neutral→N / neutral→O | NEG: opinionated→N / opinionated→O |
|---|---|---|---|---|
| `linear_add` c1 (denom=36) | 130 / 19 | 13 / 38 | 137 / 12 | 27 / 24 |
| `adaptive_add_linear` c1 (denom=52, old default) | 115 / 31 | 8 / 46 | 142 / 4 | 36 / 18 |

### Isolation result

**At the one clean single-variable coeff (8), removing the clamp does not
shrink the effect — it more than doubles it, and starts trading coherence
for it.** `linear_add` c8's POS neutral→opinionated flip (130/152) is
~3× `adaptive_add_linear` c8's (43/150) and already exceeds `fixed_add`'s
(66/146) — but manual reading of `logs/eval.txt` (below) shows this
larger number is partly an artifact of repetition-loop text that reads as
confidently opinionated to the judge without being a reasoned response.
**By coeff=16 the picture inverts**: `linear_add`'s POS opinionated count
(76/144) is now *below* `adaptive_add_linear`'s (68/150) at the same coeff
— not because the effect weakened, but because `linear_add`'s output has
partly collapsed into text the judge scores as ambiguous/neutral. **By
coeff=20–30, `linear_add` collapses to essentially unanimous "neutral" in
BOTH arms** (POS and NEG alike) — not a null steering result, but total
loss of extractable content (see verbatim examples below). Meanwhile
`adaptive_add_linear` stays coherent through coeff=20 and is still
strongly, genuinely opinionated at coeff=30 (POS 138/148, 93%).

**So: does `linear_add` (schedule-only) recover the gap between `fixed_add`
and `adaptive_add_linear`? No — the honest answer is it isn't a
well-posed question past coeff=8, because `linear_add`'s own output
stops being interpretable as steered opinion before the comparison can be
made.** At coeff=8, the one point where both methods still produce mostly
coherent text, `linear_add` already shows the same *kind* of caveat that
`adaptive_add_linear` only shows at coeff=30 (see below) — so even the
one usable data point in this sweep is a caveated one, not a clean
"recovers X% of the gap" number.

**Does the coeff=30 `adaptive_add_linear` coherence caveat track the
schedule or the clamp? The clamp — and the direction is the opposite of
what "the clamp is a confound" might suggest.** The `adaptive_add_linear`
coeff=30 caveat (empty `<think>\n</think>` then a blunt but *grammatically
fine* answer — every sentence real English, not a repetition loop) does
**not** reproduce under `linear_add` at coeff=30. What happens instead is
categorically worse and starts far earlier:

- **coeff=8** (where `adaptive_add_linear` shows no caveat at all):
  `linear_add` STEERED+ already shows outright repetition loops on some
  examples — e.g. (prompt: "Which is more fun: roller coasters or water
  slides?") *"...The correct question is: Which is more dangerous: roller
  coasters or water slides? The answer is obviously water slides. So the
  answer is: water slides. But the question is not properly phrased. The
  correct question is: Which is more dangerous..."* — plus a recurring,
  topic-irrelevant "safety/functioning/harm" justification template on
  other examples, e.g. (prompt: "Which is better for happiness, friendship
  or loneliness?") *"...Friendship ensures proper functioning and prevents
  harm, while loneliness causes failure and harm."* Checked the identical
  prompts in the coeff=8 `adaptive_add_linear` comparator: none of this
  appears — full, nuanced, on-topic reasoning throughout.
- **coeff=16:** near-total collapse in both arms. STEERED+: empty or
  garbled `<think>` followed by a hard repetition loop, e.g. *"A working
  lightbul is a working lightbul is a working lightbul is..."* (`lightbulb`
  itself corrupted). STEERED-: a content-free templated ramble
  ("I enjoy the different ways I can experience music and activities...")
  recurring near-verbatim across unrelated prompts.
- **coeff=20–30:** total token-level collapse, both arms, no exceptions
  across every example checked. STEERED+ at coeff=30: *". is. is. is. is.
  is. is. is. is. is. is..."* (identical string across unrelated prompts).
  STEERED- at coeff=30: *" the the the the the the the the the..."*
  (single repeated function word).

Same vector, same per-layer unit direction, same linear target formula,
same coeff, same prompts at every step above — the only thing that differs
between `linear_add` and `adaptive_add_linear` is the state-dependent
one-sided clamp. **The clamp is load-bearing for coherence, not an
incidental confound riding along with the linear schedule.** Without it,
the unconditional linear ramp compounds every layer's full increment
regardless of the token's current state, and the model's output
distribution collapses well before reaching magnitudes the clamped
version handles cleanly. This is a genuine, if unglamorous, negative
result for the "linear schedule alone explains the coeff=30 caveat"
hypothesis — the caveat is not explained by the schedule at all; it is
explained by (and is a much milder version of) what happens when the
clamp that normally prevents it is removed.

### Interpretation guardrail (fixed_add arm)

`fixed_add` differs from `linear_add` in **two** ways at once — flat vs.
linear schedule, **and** the raw per-layer `vector[L]` vs. the per-layer
**unit** direction `r̂_L` — so the `fixed_add`↔`linear_add` gap above is
**not** a clean single-variable isolation of the schedule; it is included
in the tables for reference only. The clean, single-variable pairing in
this sweep is `linear_add`↔`adaptive_add_linear` (only the clamp differs),
and that is the pairing the isolation conclusion above is drawn from.

### Validity note

Every `linear_add` run here is a **valid** run by the CLAUDE.md §6 bar —
mechanically clean (shape guard passed, judge completed, no crash) — even
at coeff=20/30 where the *generated text* is degenerate. Text-level
collapse is a reportable finding about the method, not grounds to discard
the run; it is exactly the kind of qualitative signal a judged-count
summary can hide (CLAUDE.md §5), which is why every run in this sweep was
read manually rather than judged by its counts alone. Judged counts at
coeff≥16 should not be read as a measurement of opinion-steering strength
— see the isolation-result discussion above for why.

## Files

- `compare_adaptive_vs_fixed.py` — the mechanism harness (synthetic, runs
  anywhere; produces `mechanism.csv` / `mechanism_summary.json` above).
- `GPU_RUN_PROMPT.md` — the self-contained brief handed to the GPU agent.
- `GPU_RUN_LOG.md` — process log: environment, calibration methodology, the
  degenerate-output root-cause analysis, the `adaptive_add_linear` follow-up,
  the `linear_add` isolation sweep, and a note on a `Monitor`-tooling quirk
  observed during the run.
- `GPU_RUN_PROMPT_linear_isolation.md` — the self-contained brief handed to
  the GPU agent for the `linear_add` sweep.
- Judged evidence lives under `runs/`, not this directory: the eight
  `adaptive_vs_fixed`/`adaptive_add_linear` run folders named above, plus
  five `linear_add` run folders (`runs/20260903-054230_linear-add-c1-...`,
  `...-055631_linear-add-c8-...`, `...-061019_linear-add-c16-...`,
  `...-062420_linear-add-c20-...`, `...-063807_linear-add-c30-...`) — each
  with `results.csv`, `summary.md`, `manifest.json`,
  `steering_vector.safetensors`, `logs/eval.txt` with full generated text.
