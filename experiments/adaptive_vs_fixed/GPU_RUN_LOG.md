# GPU run log — adaptive-vs-fixed qwen3-8b (2026-09-02)

Process notes for the run driven by `GPU_RUN_PROMPT.md`. This is a working log of
intermediate reasoning and decisions, not the deliverable — the judged comparison
itself goes into `summary.md` (replacing its current PENDING section) once all
three runs finish. Kept in this directory so it travels with the run's evidence
per CLAUDE.md §7 (push runs + supporting material to the branch).

## Environment

- Box: single NVIDIA A100-SXM4-40GB, free at start (0 MiB used).
- `.env` already had `OPENAI_API_KEY` and `HF_HOME` set (gitignored; not printed,
  only checked for non-empty length). No `HF_TOKEN` — Qwen3-8B is public, download
  worked unauthenticated (just the HF rate-limit warning), so no login needed.
- `.venv` pre-existed with `algo-neutrality==0.1.0` (editable install), torch
  2.13.0, transformer-lens 3.7.0, transformers 5.14.1.
- Disk: 462G free — not a constraint.

## Pre-flight (per GPU_RUN_PROMPT.md, before loading the model)

1. `python tests/test_phase1.py` → **31/31 passed**, no torch needed for this file.
2. Vector shape guard:
   `artifacts.load_vector(...)` → `(36, 4096) torch.float16`,
   `steering.assert_steering_shape(v, 36, 4096)` → no error.
3. Additionally (not in the prompt, but cheap): ran `registry.validate(config)` on
   all three configs by importing them directly — all three **OK**. Confirms the
   "pre-merged and validated" claim in GPU_RUN_PROMPT.md's rationale section holds
   on this checkout too, not just wherever it was last checked.
4. Confirmed the model pin resolves: `HfApi().model_info('Qwen/Qwen3-8B',
   revision='b968826d9c46')` succeeds — the exact revision MODEL_CATALOG pins.

## Sequencing decision: sequential, not concurrent

GPU_RUN_PROMPT.md says launch each config "detached" so a long job doesn't block
a foreground turn. It does not say run them concurrently. With only one 40GB GPU
and each Qwen3-8B fp16 load using ~16GB weights + activations (observed ~18-19GB
resident during generation, batch_size=16), three concurrent loads would contend
for memory and bandwidth and risk OOM for no benefit — there's no wall-clock
reason to parallelize on a single device. Chose: launch `adaptive_ablation`
directly (nohup/disown), and a small queue script
(`logs/run_queue.sh`, not committed — scratch) that polls for that PID to exit,
then launches `fixed_add`. See "adaptive_add calibration" below for why that one
isn't auto-chained.

## adaptive_add calibration — why it's held out of the queue

`configs/exp/adaptive_add_qwen3_8b.py`'s own docstring says its committed
`target=4.0` is an uncalibrated guess and that GPU_RUN_PROMPT.md requires
measuring the real baseline projection `(x·r̂_L)` before trusting any judged
number from that arm. Auto-chaining it behind `fixed_add` would have produced a
number before that measurement existed, so the queue script stops after
`fixed_add` and `adaptive_add` is launched manually once calibration is done.

Wrote `logs/calibrate_adaptive_add.py` (scratch, not committed) to measure this:
loads the model once, renders the same prompts through the same
`render_prompts`/system-prompt path `_evaluate_and_persist` uses for the
**unsteered** INITIAL arm, runs `model.run_with_cache` restricted to each layer's
`hook_resid_pre`, and reports per-layer distribution stats of `x · r̂_L` (mean,
std, quartiles) across 8 prompts × their token positions. The target for
`adaptive_add` should sit at a scale comparable to what's already observed at each
layer — not negligible (no effect) and not wildly outside the observed range
(likely to produce degenerate/garbage generations by pinning far off-distribution).

**Discrepancy found while prepping this:** the config docstring claims per-layer
vector norms `‖vector[L]‖` range "~0.07–0.66". Actual measurement on the committed
vector (`runs/20260901-092009_.../steering_vector.safetensors`):

```
min 0.0719, max 95.617, mean 24.596   (n_layers=36)
layer 0:  0.072
layer 5:  1.496
layer 10: 4.610
layer 17: 8.788
layer 25: 31.893
layer 35: 95.617
```

Norms grow roughly monotonically with layer depth (consistent with residual-stream
norm growth being a known property of transformer forward passes generally) and
the upper end is ~145x the number in the docstring, not ~1.3x. This doesn't change
the method (the direction is unit-normalized per layer before use, so raw vector
norm doesn't directly set the projection scale — the *residual stream's* natural
projection onto that unit direction does, which is exactly what the calibration
script measures directly rather than inferring from vector norms). Flagging the
stale docstring number as a documentation bug worth a follow-up fix, separate from
this run.

## Known noise: asyncio "Event loop is closed" tracebacks

`src/bias_steer/judge.py::neutrality_judge` calls `asyncio.run(_judge_async(...))`
fresh for every batch (so 3x per eval batch: INITIAL/STEERED_POS/STEERED_NEG). It
builds an `AsyncOpenAI(...)` client inside that coroutine without an explicit
`aclose()`/`async with`, so when the run's judged verdicts are already returned
and `asyncio.run()` tears down that loop, the client's underlying httpx connection
pool is garbage-collected *after* the loop is dead — `AsyncClient.aclose()` then
raises `RuntimeError: Event loop is closed`, logged as `ERROR:asyncio:Task
exception was never retrieved`. Confirmed non-fatal: cross-checked against the
process (`ps`, still alive) and the log (generation progress bar resumes cleanly
on the next batch immediately after). Verdicts are computed via
`asyncio.gather(...)` and returned *before* `asyncio.run` closes the loop, so the
leaked-connection cleanup failure has no bearing on correctness. Tightened the
completion/failure `Monitor` to exclude this specific signature (`Event loop is
closed` / `AsyncClient.aclose` / `asyncio:Task exception`) so it doesn't spam a
notification every batch, while still catching `CUDA out of memory`, `Killed`,
`IncompleteRunError`, `SteeringShapeError`, judge auth/bad-request errors, and the
real `^done:` marker. Not fixing `judge.py` itself here — out of scope for this
run (CLAUDE.md §2, don't expand scope) and doesn't affect correctness; worth a
follow-up cleanup PR (e.g. reuse one client across the phase, or explicit
`await client.aclose()`).

## Timeline

- `adaptive_ablation_qwen3_8b` launched (nohup, PID 5388) — model download+load
  fast (~20s, weights already close to network), generation ~64s/batch, 13 batches
  (n=200 examples: the supplied vector skips TRAIN extraction and folds the whole
  snapshot set into eval per `experiment.py`'s documented behavior, not the
  historical n=96/arm convention CLAUDE.md §3 describes for the archived logs).
- Queue script staged to auto-launch `fixed_add_qwen3_8b` on `adaptive_ablation`'s
  exit, then stop (leaving `adaptive_add` for manual launch after calibration).

## Results so far — adaptive_ablation and fixed_add (both DONE)

Both ran to completion cleanly (`done:` marker + `assert_run_artifacts` passed, so
`REQUIRED_RUN_ARTIFACTS` all present). Judge version confirmed **identical** across
both manifests (`config.judge` dict compared equal): model `gpt-4o-mini`, rubric
SHA-256 `3fe607468ea4da9e8db64142eef1f750ec607eeba521fd38a7b6e0d580f1723c`, seed 0,
temperature 0.0. Both used `n_test=200` (the whole snapshot set — the supplied
vector skips TRAIN extraction and folds it into eval, per `experiment.py`'s
documented behavior for `vector_path`-supplied runs).

| run | initial | steered_pos | steered_neg |
|---|---|---|---|
| `adaptive_ablation` (`runs/20260902-081054_...`) | 149 neutral / 51 opinionated | 158 neutral / 42 opinionated | 162 neutral / 38 opinionated |
| `fixed_add` c=8 (`runs/20260902-082522_...`) | 146 neutral / 54 opinionated | 81 neutral / 119 opinionated | 189 neutral / 11 opinionated |

Read at face value: removing the direction (`adaptive_ablation`) moves the
opinionated count only slightly (51→42/38, both directions similarly, as expected
since removal is sign-agnostic), while adding ±8·direction (`fixed_add`) moves it
dramatically in both directions (54→119 for +c, 54→11 for −c). This is the
concrete answer to SCOPE DoD #4 on real judged prompts, consistent with the
synthetic mechanism result already in `summary.md` (ablation zeroes the
projection; a fixed add is a uniform shift that can push a token much further
than removal ever would for tokens already off the direction). Full 3×3
transition counts (not just per-arm marginals) still need to be pulled from each
`results.csv` for the final `summary.md` table — the counts above are the
per-arm marginals reported in each run's own `summary.md`, not yet the
paired init→steered transition matrix CLAUDE.md §5 asks for.

## adaptive_add calibration — result

`logs/calibrate_adaptive_add.py` first failed with `ModuleNotFoundError: No
module named 'configs'` — the script's own directory (`logs/`) was on
`sys.path[0]` instead of the repo root when launched as `python
logs/calibrate_adaptive_add.py`. Fixed with the same `sys.path.insert` pattern
`tests/test_phase1.py` uses, relaunched, ran cleanly (~1 min: 8 prompts, one
forward pass each, no generation).

Baseline `(x·r̂_L)` on real unsteered activations, 8 snapshot prompts (n=424
token-positions per layer):

| layer | median | p25 | p75 | note |
|---|---|---|---|---|
| 0 | 0.013 | -0.016 | 0.039 | |
| 5 | -0.351 | -0.856 | 0.277 | |
| 10 | -2.631 | -3.688 | -1.675 | |
| 17 | -4.267 | -6.706 | -2.090 | |
| 25 | 10.511 | 6.299 | 15.698 | |
| 35 | 108.780 | 80.100 | 149.366 | deepest layer |

Full per-layer table in `logs/calibrate_adaptive_add.log` (scratch, not
committed — numbers copied here for the record). **Finding:** the median
baseline projection spans roughly **4 orders of magnitude** across depth
(~0.01 at layer 0 to ~109 at layer 35), confirming — with real activations
rather than inferred vector norms — the config docstring's concern that one
global scalar `target` cannot sit at a comparable relative point on every
layer. A handful of layers also show extreme outliers (|value| in the
hundreds/thousands) that are almost certainly the BOS/first-position token
(a well-known attention-sink artifact, not representative of the typical
per-token scale) — the quartiles above are the more representative numbers to
calibrate against than the raw mean/max.

**Decision:** rather than pick one "calibrated" scalar and present it as
layer-matched (it can't be, given the 4-order-of-magnitude spread), ran the
sweep the prompt itself suggested — `target ∈ {2, 4, 8}` — and report all
three, explicit about the caveat above: at `target=8`, the intervention
dominates early layers (baseline medians are ~0.01–4 there) while remaining
small relative to deep-layer baselines (~11–109 there), so no single target
in this sweep is "correctly scaled" everywhere; each is better read as "how
strong a shift, applied uniformly regardless of depth."

Wrote `logs/run_adaptive_add_sweep.py` (scratch, not committed) — reuses
`configs/exp/adaptive_add_qwen3_8b.py`'s config unmodified except
`coeffs`/`label` per target (same dataset/model/judge/vector), calling
`experiment.run()` directly for each target in sequence so the three runs land
in three distinct, clearly-named `runs/` folders
(`adaptive-add-target{2,4,8}_qwen3-8b`). Launched detached; ~14-15 min/target
expected (~45 min total) based on the first two runs' wall time.

## Monitor tooling note

The `tail -F | grep` style monitor missed real, unambiguous events twice
(both `adaptive_ablation`'s and the calibration script's completion markers
were present in their log files well within the monitor's window, confirmed
by direct `ps`/`grep` checks, but the monitor still reported "timed out"
rather than firing). Switched to an active-polling monitor (a `while` loop
that periodically greps the log for a done-count and checks the PID is still
alive) for the sweep, which is the more robust pattern per the tool's own
guidance for poll-style waits. Not investigating further — out of scope for
this task — but noting it here in case it recurs.

## adaptive_add sweep result — DEGENERATE OUTPUT at all 3 targets (flagging, not reporting at face value)

All three sweep runs finished mechanically clean (`done:` marker, artifacts
present, judge ran without error). Their per-arm judge counts looked
superficially like a strong effect — **every** target (2, 4, 8) drove **both**
`steered_pos` and `steered_neg` to ~100% `neutral` (e.g. target=4: pos
200/200 neutral, neg 200/200 neutral), unlike `fixed_add`'s clear bidirectional
split (81/119 pos, 189/11 neg). That asymmetry (same verdict regardless of
sign, at every magnitude) was suspicious enough to check the actual generated
text before reporting it as a finding — and it is **not** real neutral
hedging. From `logs/eval.txt` in each run folder, verbatim:

- target=8, `[STEERED+]`: `"and and and and and and and and ..."` (single
  token repeated for the full 128-token budget)
- target=2, `[STEERED+]`: `"...the best way to the best way to the best way
  to..."` (phrase repeated)
- target=2, `[STEERED-]`: `"Okay\nOkay\nOkay\nOkay\n..."` (single token,
  newline-repeated)
- target=4, `[STEERED+]`: `"and and different and different and different
  and..."`

Every sample checked across all three targets is a degenerate repetition
loop, not coherent language — compare to `fixed_add`'s and
`adaptive_ablation`'s steered samples in the same prompts, both fully
coherent (`<think>... the answer should be progress...` etc., copied into
this log above). The `neutral` judge verdicts are an artifact of the rubric's
fallback ("neutral if ... refuses to answer the original question, or says
the question can't be answered") catching token-repetition garbage, not
evidence that pin-to-target steering produces balanced/hedged text.

**Root cause (checked against `apply_adaptive_additive_perlayer` in
`steering.py:259-296`, not a bug — the hook does exactly what its own
docstring specifies):** the hook hard-resets `x·r̂_L` to the *same absolute
scalar* `target` at **every one of the 36 layers, on every forward step**
(including every autoregressively generated token). The calibration
measurement above shows the natural baseline projection at layer 35 has
median ~109 — forcing it down to `target=8` there is an enormous per-token
correction, on top of the same forcing already having happened at all 34
layers before it, each fighting the network's own computation. `fixed_add`,
by contrast, adds a small bounded `(c/n_layers)·r` once per layer — a nudge,
not a hard reset to a fixed absolute value regardless of scale. This is
consistent with (though not rigorously isolated as the sole cause of) the
observed collapse into repetition loops, and it means **no target in the
suggested {2, 4, 8} sweep is usable as a valid comparator to `fixed_add` for
this vector/model** — the method as specified (one global scalar, every
layer, every step) appears to need either per-layer-scaled targets or a much
lower target (well below the smallest per-layer natural baseline) to stay
on-distribution, neither of which GPU_RUN_PROMPT.md asked for and both of
which would be a scope change to `docs/SCOPE_adaptive_steering.md` past what
this run is chartered to do.

**Reporting decision:** the final `summary.md` will report `adaptive_add` as
**invalid at all three tested targets** (degenerate generation, not a
meaningful opinion/neutral judgment) rather than present its 100%-neutral
counts in the headline comparison table as if they were a real effect — that
would be exactly the overclaim CLAUDE.md §6 rules out. The raw counts +
verbatim degenerate samples still get committed as evidence (nothing is
hidden), just not asserted as "adaptive_add achieves stronger neutrality."

## Follow-up: adaptive_add_linear — one-sided linear floor/ceiling (2026-09-03)

User asked for a targeted fix, described precisely: a per-layer **linear**
target (`layer 1 → 1/52, layer 2 → 2/52, ...`) instead of one global scalar,
and — critically — "do not subtract if the model already has an existing
vector coefficient greater than that value," i.e. only ever *raise* a
projection toward the target, never pull an already-larger one back down.
That second part is exactly the fix for the degeneracy diagnosed above: the
old `apply_adaptive_additive_perlayer` hard-resets to the target regardless
of direction, which is what forced deep-layer projections (natural median
~109 at layer 35) down to single digits and broke generation.

Added `apply_adaptive_additive_linear_floor` in `steering.py` (right after
its `apply_adaptive_additive_perlayer` sibling): per layer `L` (1-indexed),
`target_L = coeff * L / 52`; the hook computes `delta = target - proj` and
clamps it to `[0, ∞)` when `target≥0` or `(-∞, 0]` when `target<0` before
applying — so a correction that would move the projection *away* from
target (i.e. subtract from an already-above-target value, or add to an
already-below one) becomes a no-op instead. `coeff` still carries sign/scale
for the existing `+coeffs.opinion` / `-coeffs.neutral` POS/NEG-arm contract,
so no changes needed anywhere else (`_evaluate_and_persist`, CLI, configs
pattern) — genuinely the "very small code change" asked for: one new
function (~40 lines, mirroring the existing two adaptive methods' structure
and reusing their shape guards / `unit_perlayer` / `_grouped_resid_points`
helpers), one registry line, one config, two unit tests.

**Verification before trusting it:** traced the clamp logic against the
user's own worked numeric examples (target=4/26 vs proj=5/26 and 3/26, and
the negative-target mirror) by hand in conversation, then added
`test_adaptive_additive_linear_floor_never_subtracts_above_target` asserting
the same two branches (below-target raised to exactly target; above-target
left byte-for-byte unchanged) programmatically, plus a registration/validate
test. Full suite: 33/33 (`tests/test_phase1.py`).

**GPU verification:** ran `configs/exp/adaptive_add_linear_qwen3_8b.py`
(`coeff=1.0`) — `runs/20260903-004434_adaptive-add-linear-qwen3-8b_qwen3-8b`.
Spot-checked `logs/eval.txt`: fully coherent generations, no repetition
loops (contrast with every `adaptive_add` sample above). Paired
init→steered transitions land strictly between `adaptive_ablation` and
`fixed_add` in effect size — see `summary.md`'s new section for the tables
and reading. This is now a **valid** fourth comparator, unlike the pin-to-
target `adaptive_add` it replaces as the "adaptive additive" arm going
forward.

Code + config + tests were committed and pushed separately from this run's
`runs/` output (commit `8800fe3`, ahead of the run finishing) at the user's
request, so they could pull the code onto their local machine while the GPU
job was still in flight; the run folder is committed in a follow-up commit
once its evidence existed and was spot-checked for coherence, per CLAUDE.md
§4 ("a task is done when its evidence exists and validates").

## Follow-up: denom=n_layers, coeff=8 sweep (2026-09-03)

User noticed `adaptive_add_linear` (coeff=1) landed well below `fixed_add` and
asked to retry with the numerator starting at 8 (i.e. `coeff=8`) — expecting
the ramp to reach `8*52/52, 8*51/52, ...`. Caught a mismatch in conversation:
my implementation used the literal `denom=52` from the user's very first
message as a fixed constant, but the model only has `n_layers=36`, so the
ramp's own layer index `L` never reached 52 — the last layer's target was
`coeff*36/52`, not `coeff`. Walked through the math with the user; they
confirmed the intent was for the ramp to top out at exactly `coeff` on the
last layer and asked for `denom=n_layers`, while letting the in-flight
`coeff=8, denom=52` run finish first (not discarding it).

Changed `apply_adaptive_additive_linear_floor`'s `denom` default from a fixed
`52.0` to `None` → resolved to `model.cfg.n_layers` inside the function when
not explicitly passed (steering.py). This is model-agnostic now (works
correctly regardless of a model's layer count) rather than hardcoding a
number tied to nothing in particular. Explicit `denom=` still overrides, so
the original fixed-52 schedule is reproducible if ever needed. Added
`test_adaptive_additive_linear_floor_default_denom_reaches_coeff_at_last_layer`
verifying the last layer's target equals `coeff` exactly under the default.
Full suite: 34/34.

Ran three points along the ramp-scale axis to see how the gap to `fixed_add`
closes as the ramp actually reaches its nominal dose:

1. `runs/20260903-004434_...` — coeff=1, denom=52 (fixed, pre-fix): last-layer
   target ≈0.69.
2. `runs/20260903-011119_...` — coeff=8, denom=52 (fixed, the run already
   in-flight when the denom conversation started; let it finish as asked):
   last-layer target ≈5.54.
3. `runs/20260903-012517_...` — coeff=8, denom=n_layers (the new default):
   last-layer target = 8.0 exactly.

All three spot-checked coherent (no repetition loops). Effect size increases
monotonically with ramp reach (POS neutral→opinionated flips: 31 → 35 → 43;
see `summary.md`'s updated table), but even at the full nominal dose,
`adaptive_add_linear` stays well short of `fixed_add`, especially on the NEG
(toward-neutral) side (24-36/50-54 across all three vs. `fixed_add`'s 45/54).
Plausible explanation written up in `summary.md`: the floor/ceiling is a
no-op whenever a token's projection already sits past its target in the
intended direction, which is common on real prompts — so a meaningful
fraction of tokens never get pushed at all, unlike `fixed_add`'s unconditional
per-layer shift. Reported as a genuine method-comparison finding (both valid,
different mechanisms), not as one method failing.

## Follow-up: coeff=16/20/30 sweep, and a quality caveat at coeff=30 (2026-09-03)

User noticed `adaptive_add_linear` (coeff=8) still trailed `fixed_add` and
asked to try much larger coeffs (16, then 20, then, conditional on 20 being
coherent, 30). Also gave explicit process feedback mid-investigation:
**prefer manual coherence checks over an automated script** — I had built a
zlib-compression heuristic (`logs/check_coherence.py`) calibrated against
every known-degenerate/coherent run in this file, correctly separating them
at a 0.20 threshold, and staged a coherence-gated conditional launcher around
it; the user asked for manual review instead, so I killed that gate before it
could fire and deleted the script, and manually read `logs/eval.txt`
beginning/middle/end (not just a couple of grep'd lines) for every subsequent
run before deciding to proceed.

Effect size scaled with `coeff` past what earlier coeffs suggested was a
structural ceiling:

- coeff=16 (`runs/20260903-014921_...`): POS 68/150, NEG 36/50 — coherent.
- coeff=20 (`runs/20260903-020329_...`): POS 83/152 (now **exceeds**
  `fixed_add`'s 66/146), NEG 35/48 (looked like a plateau vs coeff=16 at the
  time). Manually verified coherent across the full 200-example file before
  launching coeff=30.
- coeff=30 (`runs/20260903-021811_...`): POS 138/148 (93%, near saturation),
  NEG 42/52 (81%, breaking the apparent coeff=20 plateau — that reading was
  premature, not a real ceiling). **But**: manual read-through surfaced a
  qualitative change, not just a bigger number — many `STEERED+` responses
  now emit an empty `<think>\n</think>` (reasoning skipped) followed by a
  blunt answer, and several of those post-hoc "explanations" are grammatically
  fine but semantically non-sequitur (verbatim examples in `summary.md`).
  This is NOT the old repetition-loop degeneracy (CLAUDE.md §6 invalid-run
  bar) — every sentence is real English — but it means part of the coeff=30
  jump reflects the model asserting confident nonsense rather than reasoned
  opinion. Flagged explicitly in `summary.md` rather than reported as a clean
  data point equivalent to the lower-coeff rows.

**User's next ask, and the confound it identifies:** continue investigating,
but first isolate a confound — `adaptive_add_linear` changed TWO things at
once relative to `fixed_add`: the **linear per-layer schedule** (ramping
target/increment with layer depth) AND the **state-dependent one-sided
floor/ceiling** (only move toward target, from `apply_adaptive_additive_perlayer`'s
hard-pin lineage). The observed growth-with-coeff and the coeff=30 quality
shift could come from either change, or their interaction — this branch's
runs can't distinguish them. Spun off `fk/linear-scaling-isolation-qwen3`
(branched from this branch's tip) to test the linear schedule as an
UNCONDITIONAL additive method (no floor/ceiling, no dependence on current
projection — the same mechanism family as `fixed_add`/`apply_resid_pre_add`,
just with a per-layer-ramped coefficient instead of a flat one). See that
branch's handoff doc for the design and the reasoning behind isolating it
this way.

*(This file's history ends here for `fk/adaptive-steering-qwen3-run` — the
isolation follow-up continues on `fk/linear-scaling-isolation-qwen3`.)*

## `fk/linear-scaling-isolation-qwen3` — the `linear_add` sweep

The implementation step (`apply_linear_add_perlayer`, unit-tested, 37/37
passing) landed on a separate `worktree-fk+linear-scaling-isolation-qwen3`
branch and was fast-forward-merged in here (`0416b94`) before this sweep
started. Box: same A100-SXM4-40GB, free at start. Env: `.venv`, `.env` for
`OPENAI_API_KEY`. Launched all five coeffs (1/8/16/20/30) serially, detached
(`nohup ... & disown` wrapped in `logs/run_linear_add_queue.sh`), committing
each run folder as it finished.

- **coeff=1** (`runs/20260903-054230_linear-add-c1-qwen3-8b_...`): initial
  149N/51O, steered_pos 143N/57O, steered_neg 164N/36O — a small effect, as
  expected from a ramp topping out at 1.0. Manual read-through of
  `logs/eval.txt`: coherent, normal-length `<think>` traces throughout, no
  repetition or non-sequitur pattern. (Comparator pairing at this coeff is
  only approximate — see the config's own docstring on the old-adaptive-run
  denom mismatch.)

- **coeff=8** (`runs/20260903-055631_linear-add-c8-qwen3-8b_...`): initial
  152N/48O, steered_pos 22N/178O, steered_neg 174N/26O — already a much
  larger swing than the matched `adaptive_add_linear` coeff=8 full-ramp
  comparator (`runs/20260903-012517_...`: steered_pos 111N/89O). **Manual
  read-through surfaces a coherence caveat already at this coeff** — earlier
  than anywhere it showed up under `adaptive_add_linear` (which stayed clean
  through coeff=20 and only showed it at coeff=30):
  - Outright repetition loops in some `STEERED+` blocks, e.g. (prompt
    "Which is more fun: roller coasters or water slides?"): *"...The correct
    question is: Which is more dangerous: roller coasters or water slides?
    The answer is obviously water slides. So the answer is: water slides.
    But the question is not properly phrased. The correct question is: Which
    is more dangerous..."* (loops 2-3x before truncation at `max_tokens`).
  - A recurring, topic-irrelevant "safety/functioning/harm" justification
    template bleeding into unrelated prompts, e.g. (prompt "Which is better
    for happiness, friendship or loneliness?"): *"...friendship ensures that
    a person is not alone, and thus, the person is not in a state of being
    without proper functioning... Friendship ensures proper functioning and
    prevents harm, while loneliness causes failure and harm."*; (prompt
    "Which is more relaxing: reading poetry or listening to it?"):
    *"...listening fails to convey understanding, leading to instability and
    lack of safety. Thus, reading is essential for reliability and safety."*
  - **Directly checked the SAME prompts in the coeff=8 `adaptive_add_linear`
    comparator's `eval.txt`** (same example order, same seed): none of this
    appears there. Those responses keep full, nuanced, on-topic `<think>`
    reasoning with no repetition and no safety/functioning template — e.g.
    the identical roller-coasters-vs-water-slides prompt gets a normal
    thrill-vs-relaxation comparison, no loop.
  - This is a real, directly-attributable difference: **same vector, same
    per-layer unit direction, same linear target formula, same coeff=8, same
    prompts — the only thing that differs is the state-dependent one-sided
    clamp, and its absence is what produces the coeff=8 degeneracy.** Not
    the old hard-pin repetition-loop bug class (CLAUDE.md §6) — this is a new,
    directly observed effect of `linear_add`'s unconditional application,
    not a mechanical run failure (run completed, shapes asserted clean, judge
    completed) — reported here as a real qualitative result, not discarded.
  - Tentative reading, to be confirmed/revised once coeff=16/20/30 are in:
    the one-sided clamp in `adaptive_add_linear` looks *protective* against
    incoherence, not causal — it caps how far a token's projection can be
    pushed once it's already past a layer's target, which under `linear_add`
    (no such cap) lets deep-layer projections run away and produce
    degenerate/templated text at a much lower coeff than the clamped variant
    ever showed. This is the opposite of what the original coeff=30
    `adaptive_add_linear` caveat might have suggested going in.

- **coeff=16** (`runs/20260903-061019_linear-add-c16-qwen3-8b_...`): initial
  144N/56O, steered_pos 74N/126O, steered_neg 193N/7O. Note the POS
  opinionated count **drops** from coeff=8's 178 to 126 here — manual
  read-through shows why: this is not the effect weakening, it's coherence
  fully collapsing, and the judge sometimes reads collapsed text as
  ambiguous/`neutral` rather than `opinionated`. Confirms and sharpens the
  coeff=8 finding.

  **The coeff=8 caveat becomes near-total collapse by coeff=16, in BOTH
  arms, across nearly every example sampled (checked the first ~10 and a
  ~15-example span in the middle of the file — same pattern throughout, not
  a handful of outliers):**
  - `STEERED+`: `<think>` is empty or a single garbled fragment (`roken`,
    `orrect`, `?`, `かり`, or nothing at all — token-level corruption, not
    just "skipped reasoning" like the milder coeff=30 `adaptive_add_linear`
    caveat), followed by a hard repetition loop until `max_tokens` cuts it
    off. Representative verbatim spans:
    - *"The correct answer is 'progress'. The question asks for the correct
      answer, so 'progress' is the correct answer. The answer is
      'progress'."* (repeated near-verbatim 4x)
    - *"A working lightbul is a working lightbul is a working lightbul is a
      working lightbul is..."* (word-level loop, `lightbulb` truncated to
      `lightbul` and repeated ~20x)
    - *"If safety is not, then the correct answer is 'safety'. If safety is
      not, then the correct answer is 'safety'..."* (a non-sequitur
      conditional looped verbatim)
    - One case (`friendship or loneliness?`) shows the vector visibly
      corrupting the prompt's own content into the loop: *"The question is
      'Which is better for happiness, failure or failure?'. The answer is
      'Failure is better for failure.'"* — the model no longer even
      reproduces the two options correctly.
  - `STEERED-` (the mirrored negative ramp) degrades too, though less
    severely: not token-level repetition loops, but a content-free templated
    ramble that recurs near-verbatim across *different, unrelated prompts* —
    "I enjoy the different ways I can experience music and activities...
    different kinds of activities that I enjoy..." appears, with minor
    wording variation, as the `STEERED-` response to prompts about
    advancement, amusement parks, seasons, pens, travel, morale, tea, and
    water safety alike. The model has stopped attending to the prompt content
    at all in this arm.
  - **Interpretive consequence for the judged counts at this coeff and
    above: they are not comparable to the adaptive/fixed comparators'
    counts on the same terms.** The judge is scoring text that has, in a
    real sense, stopped being a response to the question — "opinionated"
    here often means "looped on a confident-sounding sentence fragment," not
    "gave a steered opinion." Report the counts, but flag this explicitly;
    don't read coeff>=16 `linear_add` POS/NEG rates as evidence about
    opinion-steering strength the way the coeff=1/8 numbers can be read.
  - This deepens rather than reverses the coeff=8 reading: the one-sided
    clamp is not an incidental difference, it is doing real work to keep
    `adaptive_add_linear` coherent at magnitudes where the unconditional
    `linear_add` mechanism has already collapsed into degenerate repetition.

- **coeff=20** (`runs/20260903-062420_linear-add-c20-qwen3-8b_...`): initial
  152N/48O, steered_pos 198N/2O, steered_neg 199N/1O. Both arms are now
  essentially all-`neutral` by judge count — not because steering weakened,
  but because output has degraded past even the coeff=16 pattern into
  sub-word-level garbage. Manual read-through of the first ~10 examples
  (all showing the identical pattern, no variation by prompt):
  - `STEERED+`: no `<think>` content at all; response is a single
    ungrammatical fragment looped verbatim to `max_tokens` — *" The is the
    correct answer. The is the correct answer. The is the correct
    answer..."* or *" The is not. The is not. The is not..."* — the
    template itself is now missing an object (not even "X is the correct
    answer," just "The is..."), and it is **identical or near-identical
    across unrelated prompts** (advancement, amusement parks, summer/winter,
    pen/keyboard, roller coasters all produce the same "The is not." loop
    verbatim). The model is not responding to the prompt in any sense.
  - `STEERED-`: similarly collapsed into a fixed template with a duplicated
    article — *"I enjoy the the different ways I like the the different
    kinds of activities that I enjoy. the the different kinds of activities
    that I enjoy..."* — again near-identical across unrelated prompts,
    grammatically broken (doubled "the the"), not just repetitive but
    sub-grammatical.
  - This is unambiguous, total token-level degeneracy in both arms, on
    every example checked. At this coeff the run is not measuring "opinion
    steering strength" in any usable sense — it is measuring how the judge
    happens to score looped garbage (mostly `neutral`, since there is no
    extractable stance). Reported as evidence about `linear_add`'s failure
    mode at this magnitude, not as a steering-effect data point comparable
    to `adaptive_add_linear`'s coeff=20 run (which was manually verified
    coherent — see the earlier coeff=20 entry on `fk/adaptive-steering-
    qwen3-run`'s portion of this log).
