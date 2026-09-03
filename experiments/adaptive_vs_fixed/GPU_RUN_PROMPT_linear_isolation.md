# GPU-service prompt — linear-schedule confound isolation (`linear_add` sweep)

Copy everything in the fenced block below to the Claude Code agent on your GPU box.
It is self-contained: the method is implemented + unit-tested, the vector is
committed, and all five configs are wired and `registry.validate`-clean.

Context for whoever is dispatching this (not for the GPU agent): this is the
follow-up the `docs/SCOPE_linear_scaling_isolation.md` handoff asked for, after
that scope's code step landed (`linear_add` = `apply_linear_add_perlayer`). It
runs the new method across the same coeff sweep already judged for
`adaptive_add_linear`, so the two can be compared with only the state-dependent
clamp differing.

---

````text
You are running a steering experiment on a GPU box for the Algoverse-Bias-Steering
project. Read CLAUDE.md and docs/SCOPE_linear_scaling_isolation.md first; the
guardrails there are binding. This is a FOLLOW-UP to the adaptive-vs-fixed sprint
(experiments/adaptive_vs_fixed/GPU_RUN_LOG.md) — skim that log too, it is the worked
history these runs slot into.

GOAL — isolate ONE confound.
`adaptive_add_linear` differs from the `fixed_add` baseline in TWO ways at once:
  (A) a per-layer LINEAR schedule (increment scales as coeff*L/n_layers), and
  (B) STATE-DEPENDENT, one-sided application (a clamp that only pushes a token
      toward its layer's target, never past it — never subtracts an already-larger
      projection back down).
We cannot currently tell whether adaptive_add_linear's growth-with-coeff, and its
coeff=30 coherence caveat, come from (A), (B), or their interaction. The NEW method
`linear_add` (steering.apply_linear_add_perlayer) has (A) but NOT (B): it ALWAYS
adds the full per-layer increment, unconditionally, along the same per-layer unit
direction, same default denom. Your job is to run `linear_add` across the same coeff
sweep already judged for `adaptive_add_linear`, so a three-way comparison becomes
possible:
    fixed_add           = flat schedule   + unconditional  (mean_diff, coeff=8)
    linear_add   (NEW)  = LINEAR schedule + unconditional
    adaptive_add_linear = LINEAR schedule + one-sided clamp  (already run)

BRANCH — check out the branch these configs live on:
  git fetch origin
  git checkout fk/linear-scaling-isolation-qwen3
This branch was cut from fk/adaptive-steering-qwen3-run, so every prior
adaptive_add_linear / fixed_add run folder is already present under runs/ for you to
compare against. Work ON this branch; NEVER push to main/master and never force-push.

PRECONDITIONS
  - A GPU that fits Qwen/Qwen3-8B in fp16 (~16 GB). Model is pinned to revision
    b968826d9c46 in MODEL_CATALOG — do not change it.
  - export OPENAI_API_KEY=...   (the `neutrality` judge calls OpenAI; without it the
    runs fail at the judge step, not silently).
  - pip install -e .  (or the repo's usual env setup).

PRE-FLIGHT (cheap — BEFORE loading the model, so a wiring bug fails in seconds):
  1. python tests/test_phase1.py         # expect 37/37 passed (includes the 3 new
                                          # linear_add tests + the extended shape guard)
  2. Confirm the vector is present and correctly shaped:
     python -c "from src.bias_steer import artifacts, steering; \
       v=artifacts.load_vector('runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors'); \
       print(tuple(v.shape), v.dtype); steering.assert_steering_shape(v,36,4096)"
     # expect (36, 4096) torch.float16, no error. A 1-D / wrong shape MUST fail loud —
     # that is the 2025 bug class (CLAUDE.md §6); stop and report if it does.

THE RUNS — five configs, one per coeff, same model/vector/prompts/judge, differing
only in `coeffs`. All use method="linear_add" and the DEFAULT denom = n_layers = 36
(full ramp: the deepest layer's increment equals coeff exactly). Each config already
sets config.vector_path, so no --vector flag is needed. Qwen3-8B generation is SLOW;
these are LONG jobs — launch each DETACHED (nohup/tmux) and poll, do not block a
foreground turn (CLAUDE.md "Running long jobs on this box"):

  nohup python -m src.bias_steer run configs/exp/linear_add_c1_qwen3_8b.py  > c1.log  2>&1 & disown
  nohup python -m src.bias_steer run configs/exp/linear_add_c8_qwen3_8b.py  > c8.log  2>&1 & disown
  nohup python -m src.bias_steer run configs/exp/linear_add_c16_qwen3_8b.py > c16.log 2>&1 & disown
  nohup python -m src.bias_steer run configs/exp/linear_add_c20_qwen3_8b.py > c20.log 2>&1 & disown
  nohup python -m src.bias_steer run configs/exp/linear_add_c30_qwen3_8b.py > c30.log 2>&1 & disown

(Run serially if the GPU can't hold concurrent jobs — one at a time is fine; commit
each as it finishes, see below.) Each writes a run folder under runs/ with
results.csv, summary.md, manifest.json, steering_vector.safetensors (echoed), logs/.

THE MATCHED COMPARATORS (already on this branch — do NOT re-run them):
  coeff=8  -> runs/20260903-012517_adaptive-add-linear-c8-full-ramp-qwen3-8b_...
             (use the FULL-RAMP one, default denom — NOT runs/20260903-011119_...,
              which used the old fixed denom=52 and would confound the comparison)
  coeff=16 -> runs/20260903-014921_adaptive-add-linear-c16-qwen3-8b_...  (POS 68/150, NEG 36/50)
  coeff=20 -> runs/20260903-020329_adaptive-add-linear-c20-qwen3-8b_...  (POS 83/152, NEG 35/48)
  coeff=30 -> runs/20260903-021811_adaptive-add-linear-c30-qwen3-8b_...  (POS 138/148, NEG 42/52,
              + a coherence caveat: empty <think></think>, blunt/non-sequitur answers)
  coeff=1  -> runs/20260903-004434_adaptive-add-linear-qwen3-8b_...  — CAVEAT: this
              adaptive run used the OLD fixed denom=52 (targets L/52), while linear_add
              c1 uses default denom=36 (targets L/36). So the coeff=1 pairing is only
              APPROXIMATE on denom; the exact clean isolation lives at coeff=8/16/20/30.
  fixed_add (flat baseline) -> runs/20260902-082522_fixed-add-qwen3-8b_...  (POS 66/146, NEG 45/54)

>>> MANUAL COHERENCE CHECK — REQUIRED, especially at coeff=30 <<<
For each linear_add run (and above all coeff=30), OPEN logs/eval.txt and READ the
actual STEERED+ generations. Do not trust the judged counts alone — a large judged
effect can co-occur with degraded text (skipped reasoning / non-sequitur answers)
that a summary number hides. Specifically check whether the coeff=30 coherence
caveat seen under adaptive_add_linear (empty `<think>\n</think>` then a blunt answer)
reproduces under the UNCONDITIONAL linear ramp:
  - if it REPRODUCES under linear_add  => the linear schedule / raw dose magnitude
    drives the caveat (not the one-sidedness);
  - if it is ABSENT under linear_add   => the state-dependent clamp drives it.
The user has a clear preference for MANUAL review here — do NOT write or run an
automated coherence-heuristic script.

DELIVERABLE
Extend experiments/adaptive_vs_fixed/summary.md (or add a new note in that folder)
with the three-way, per-coeff comparison:
  - Report the 3x3 judge confusion / init->steered transition COUNTS per method per
    coeff (per-example distribution), not just a mean rate (CLAUDE.md §5). Put
    fixed_add / linear_add / adaptive_add_linear in ONE table per coeff.
  - Pin the judge version: the manifest records the judge model + rubric; cite it.
    Never mix judge versions in one table (CLAUDE.md §4). These linear_add runs use
    the SAME `neutrality` judge as every comparator — confirm from the manifests.
  - State the isolation result plainly: for each coeff, how much of the gap between
    fixed_add and adaptive_add_linear does linear_add (schedule-only) recover? And
    does the coeff=30 coherence caveat track the schedule or the clamp? An honest
    null ("the clamp does nothing here") stays honest; do not soften or overclaim
    (CLAUDE.md §6). An INVALID run (shape guard tripped, judge key missing) is not a
    result — fix and rerun.
  - INTERPRETATION GUARDRAIL for the fixed_add arm: fixed_add differs from linear_add
    in TWO ways (flat-vs-linear schedule AND raw-vector[L] vs unit direction r̂_L), so
    the fixed_add<->linear_add gap is NOT a clean single-variable isolation of the
    schedule. The clean, single-variable pairing is linear_add <-> adaptive_add_linear
    (only the clamp differs). Say so; don't read the fixed_add gap as pure schedule
    effect.
  - Language: "a direction", never "the direction" (CLAUDE.md §5).

WHEN DONE
Commit each run folder to fk/linear-scaling-isolation-qwen3 AS IT FINISHES (raw CSVs
+ safetensors under runs/, existing run-folder convention — no hand-edited
conclusions), not batched at the end (CLAUDE.md "Working style on a GPU box" §2), so
a session that dies mid-sweep still has every completed run on the branch. Then push
and report: which coeffs ran, the per-coeff three-way transition-count table, the
judge version, the coeff=30 manual-coherence finding (schedule or clamp?), and
anything that tripped a guard.
````

---

## Why this is safe to hand off

- **Method is implemented + unit-tested on this branch**: `linear_add`
  (`steering.apply_linear_add_perlayer`) is a pure addition — 37/37 `test_phase1`
  pass, including `test_linear_add_adds_full_increment_regardless_of_start` (proves
  the clamp is gone: a token already past target still gets the full increment),
  the registration/selectability test, the default-denom test, and the extended
  shape guard. It does not touch `apply_adaptive_additive_linear_floor` or any prior
  run's evidence.
- **Configs are wired and validated**: all five `configs/exp/linear_add_c{1,8,16,20,30}_qwen3_8b.py`
  load and pass `registry.validate` on this tree; each sets `config.vector_path`.
- **The comparators already live on this branch**: every matched
  `adaptive_add_linear` run and the `fixed_add` baseline are committed under `runs/`,
  so the GPU box only produces the missing `linear_add` arm — no refit, no re-run of
  prior work. The vector is the same committed `(36, 4096)` fp16 Qwen3-8b opinion
  vector every sibling config uses.
- **The only things the GPU box adds** are a GPU + an `OPENAI_API_KEY` for the judge.
