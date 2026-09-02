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

*(Updated as the run progresses — see the bottom of this file for the latest status.)*
