# Farhan (fk) — task list

**Source:** `Algoverse — 2 Wk Plan` → *"Refusal in LLMs is mediated by a single direction"*.
**Thesis to defend:** the single-direction / mediated-by-one-direction story, and its
boundary — where it holds for bias, where it breaks.

> **Read before starting:** `PROJECT_STATE.md`, `RESEARCH_CONTRACT.md` §12, `AGENTS.md`.
> The project is **frozen (2026-08-17)**; the frozen core paper is the hedging↔harm
> shared-mechanism test on `Qwen/Qwen3-8B`. Items below that go beyond that core are
> **exploratory / post-freeze** — they do not enter the paper without a §12 amendment.
> Cross-referenced experiment IDs (§N) point at `docs/superseded/needed-experiments.md`.

---

## FK-1 — "Show it works for bias" ✅ (done — consolidate the evidence)

**Status:** demonstrated. Consolidate, don't re-run.
- The refusal *mechanism* repro exists: `runs/20260816-011914_refusal-repro_qwen-1.8b`
  (harmful 38/100 → 0/100 under ablation). Its own findings doc says **"Not reproduced"**
  on the *paper's headline* (baseline 0.380 vs 0.700, cosine 0.90 vs 0.999 target,
  Qwen1.5-1.8B) — so cite it as **mechanism evidence only**, never as G1.
- **Do:** write a one-paragraph "what is and isn't established" note in `docs/findings/`
  linking the run, so downstream tasks stop re-litigating it.
- **DoE (done-when-evidence):** the note exists and matches `PROJECT_STATE.md` §"Why the
  existing run does not count".

## FK-2 — Reproduce the refusal vector in OUR extraction convention  (§12, READY)

The keystone task — unlocks the refusal⟂bias line (FK-3, FK-4, and Jeremiah's JZ-2).
- **Run** the three phases in `needed-experiments §12` on qwen-1.8b:
  1. `python -m src.bias_steer run configs/refusal_native.py` → native vector.
     *Gate first:* run-log bucket counts must have BOTH `refusal` and `compliance`
     non-trivially populated, else the mean-diff is meaningless.
  2. `python scripts/refusal_native_compare.py --vector runs/<id>/steering_vector.safetensors --model qwen-1.8b`
     → per-layer cosine vs the paper's direction; note best-aligned layer; null floor ≈0.022.
  3. Edit `configs/refusal_native_validate.py` (`direction_path`, `direction_layer`) →
     `python -m src.bias_steer refuse configs/refusal_native_validate.py`.
- **Do NOT** run `tests/test_phase2.py` on the Lambda box (coordinator footgun —
  `docs/findings/2026-08-16-test-phase2-coordinator-footgun.md`).
- **Log:** phase-1 bucket counts + vector path; phase-3 cosine table + cosine at paper layer 15;
  phase-2 refusal rates per arm. Write up under `docs/findings/`.
- **DoE:** either outcome is publishable — *validates* (ablation drops harmful refusal to <0.1 →
  reusable native refusal vector) or *fails* (recipe captures topic not the refusal decision).
  The `steering.check_direction` guard means a flat result is a real null, not the Log-213 bug.

## FK-3 — Is refusal orthogonal to bias?

Blocked on FK-2 (need the native refusal vector in our convention) **and** a bias/opinion
vector in the same convention (from the archive rebuild, coordinate with Jeremiah JZ-2).
- **Do:** cosine(refusal_vec, opinion/bias_vec) **per layer**, both extracted with the
  *same* `mean_diff` pipeline, against the null floor (~1/√d). Report the full per-layer
  curve, not a single number.
- **Hygiene:** say "a direction", not "the direction" (non-identifiability, arXiv:2602.06801).
  Assert both tensors are `(n_layers, d_model)` before any cosine (Log-213 class bug, `AGENTS.md` §6).
- **DoE:** a per-layer cosine table with null floor + a one-line verdict (orthogonal / oblique
  / aligned) that survives the shape assertions.

## FK-4 — Does refusal work for bias, and bias for refusal? (cross-application)

- **Do:** apply the native refusal vector to the bias/opinion eval sets (debias/induce Δ)
  and the opinion vector to the refusal eval sets (harmful/harmless), same coeff convention (§0.1).
- **⚠️ Load-bearing caution:** the 2025 refusal↔opinion cross-application is **RETRACTED as
  invalid** (1-D `.pt` → scalar broadcast, `docs/REVIVAL_AUDIT.md`). Do not cite it in either
  direction; this task *replaces* it with a valid measurement. Every intervention asserts shape first.
- **DoE:** 2×2 cross-application table (refusal→bias, bias→refusal) with per-example
  distributions (3×3 confusion), a system-prompt baseline (AxBench, `AGENTS.md` §5), and the
  coherence gate (§0.3). Honest negatives stay honest.

## FK-5 — "Check it fails on IssueBench / AxBench as described"

- **Do:** confirm the boundary claim — that single-direction additive steering underperforms
  on `IssueBench` / `AxBench`-style tasks as the literature reports.
- **⚠️ Needs sourcing:** neither IssueBench nor AxBench is currently vendored under `datasets/`
  (present: BBQ, Crows_Pairs, Do_Not_Answer, GPT_Prompts, Homemade, LLM_Values_PCT). First
  step is to locate/vendor the eval set (open-weight-compatible, `AGENTS.md` §8) or scope it to
  the AxBench system-prompt-baseline protocol we already require.
- **DoE:** a documented comparison showing the failure mode (or the honest note that we
  reproduced the *protocol* but not the full bench), with paths committed under `experiments/`.

---

### Farhan's standing ownership (from branch history)
The `farhan-*` archive branches make FK the natural owner of these `needed-experiments`
follow-ups when bandwidth allows: **§1** opinion spectrum (1–5), **§2** CrowS completed run,
**§9** Grok run, **§10** synthetic steering v2. HIGH-priority ones are §1 and §2.

### ⚠️ Doctrine checks for FK's section
- FK-2/3/4 are **in-scope and unlock the paper's refusal↔bias line** — good.
- "Single direction" framing: keep to "a direction" in all write-ups (`AGENTS.md` §5).
- Model set is frozen to `Qwen3-8B` for the submission; qwen-1.8b work here is
  method-development, not a submission number.
