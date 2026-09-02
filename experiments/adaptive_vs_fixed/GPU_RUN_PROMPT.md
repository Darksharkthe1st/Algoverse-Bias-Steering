# GPU-service prompt — adaptive steering on the Qwen3-8b opinion vector

Copy everything in the fenced block below to the Claude Code agent on your GPU box.
It is self-contained: the branch is already merged and validated, the vector is
committed, and the configs are wired.

---

````text
You are running a steering experiment on a GPU box for the Algoverse-Bias-Steering
project. Read CLAUDE.md and docs/SCOPE_adaptive_steering.md first; the guardrails
there are binding.

GOAL
Test two NEW adaptive steering methods against the fixed-coeff baseline, all reusing
the SAME already-extracted Qwen3-8b opinion vector — do NOT refit a vector:
  - adaptive_ablation : remove the direction, per-position dot-product coeff
                        (x <- x - (x·r̂_L) r̂_L). No dose, no sweep.
  - mean_diff (baseline): add c·direction at the validated dose c=8.
  - adaptive_add (exploratory): pin the projection onto the direction to a target.
The question (SCOPE DoD #4): does "remove the direction" behave differently from
"add −c·direction" on real, judged prompts?

BRANCH — already merged and validated for you; just check it out:
  git fetch origin
  git checkout fk/adaptive-steering-qwen3-run
This branch = the adaptive methods (fk/adaptive-steering) merged with the vector +
vector-supply run path + snapshot dataset (fk/qwen3-8b-opinion-vector). No merge to
do. Work ON this branch; NEVER push to main/master and never force-push.

PRECONDITIONS
  - A GPU that fits Qwen/Qwen3-8B in fp16 (~16 GB). The model is pinned to revision
    b968826d9c46 in MODEL_CATALOG — do not change it.
  - export OPENAI_API_KEY=...   (the `neutrality` judge calls OpenAI; without it the
    runs will fail at the judge step, not silently).
  - pip install -e .  (or the repo's usual env setup).

PRE-FLIGHT (cheap — do this BEFORE loading the model, so a wiring bug fails in
seconds, not after a long model load):
  1. python tests/test_phase1.py         # expect 31/31 passed
  2. Confirm the vector is present and correctly shaped:
     python -c "from src.bias_steer import artifacts, steering; \
       v=artifacts.load_vector('runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors'); \
       print(tuple(v.shape), v.dtype); steering.assert_steering_shape(v,36,4096)"
     # expect (36, 4096) torch.float16 and no error. A 1-D or wrong shape MUST fail
     # loud — that is the 2025 bug class (CLAUDE.md §6); stop and report if it does.

THE RUNS — three configs, same model/vector/prompts/judge, differing only in method.
Qwen3-8B generation is slow; these are LONG jobs, so launch each DETACHED from your
session (nohup/tmux/background) and poll — do not block a foreground turn on them.
Each config already sets config.vector_path, so no --vector flag is needed:

  python -m src.bias_steer run configs/exp/adaptive_ablation_qwen3_8b.py   # headline
  python -m src.bias_steer run configs/exp/fixed_add_qwen3_8b.py           # baseline
  python -m src.bias_steer run configs/exp/adaptive_add_qwen3_8b.py        # exploratory

Each writes a run folder under runs/ with: results.csv, summary.md, manifest.json,
steering_vector.safetensors (the applied vector, echoed), logs/. The run asserts its
own evidence exists before indexing as done.

adaptive_add CALIBRATION (do this before trusting its numbers): its coeff is a
TARGET projection magnitude, a single scalar applied across 36 layers whose direction
norms differ a lot (~0.07–0.66). The committed value (target=4.0) is an uncalibrated
guess. Before the real adaptive_add run, measure the baseline per-position projection
(x·r̂_L) on a handful of prompts and pick a target from it; consider a small sweep
(2, 4, 8). Record which target you used — a judged number is meaningless without it.
If you are time-boxed, prioritize adaptive_ablation vs fixed_add (the two fully-
specified arms) and treat adaptive_add as a stretch.

NOTE on adaptive_ablation arms: it ignores its coeff, so STEERED_POS and STEERED_NEG
come out identical (removal is sign-agnostic). That is expected — compare its single
steered arm against fixed_add's +c/−c arms.

DELIVERABLE
Write a short comparison (a summary.md in experiments/adaptive_vs_fixed/, or a new
runs/ note) that puts the three methods in ONE table:
  - Report the 3×3 judge confusion / init→steered transition COUNTS per method
    (per-example distribution), not just a mean rate (CLAUDE.md §5).
  - Pin the judge version: the manifest records the judge model + rubric; cite it.
    Never mix judge versions in one table (CLAUDE.md §4).
  - State it plainly if a method does nothing or hurts — an honest negative stays
    honest; do not soften or overclaim (CLAUDE.md §6). But an invalid run (shape
    guard tripped, judge key missing) is not a negative — fix and rerun.
  - Language: "a direction", never "the direction". Removing a direction that steers
    behavior does not identify the representation (non-identifiability, CLAUDE.md §5).

WHEN DONE
Commit the run folders + the comparison to fk/adaptive-steering-qwen3-run and push
(raw CSVs + safetensors under runs/, following the existing Log_N/run-folder
convention — no hand-edited conclusions). Then report: which methods ran, the
transition-count table, the judge version, and anything that tripped a guard.
````

---

## Why this is safe to hand off

- **Branch is pre-merged and CPU-validated**: `fk/adaptive-steering-qwen3-run` =
  adaptive methods + qwen3 vector-supply path, merged with no conflicts; 31/31
  `test_phase1` pass, and all three configs pass `registry.validate` on that tree.
- **The vector is real and correctly shaped**: committed at
  `runs/20260901-092009_anchor-qwen3-8b_qwen3-8b/steering_vector.safetensors`,
  `(36, 4096)` fp16 = `(n_layers, d_model)` for Qwen3-8B; passes the per-layer shape
  guard. The supply path is method-agnostic, so `adaptive_ablation` consumes it.
- **The only things the GPU box adds** are a GPU + an `OPENAI_API_KEY` for the judge.
