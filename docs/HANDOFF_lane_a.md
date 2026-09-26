# HANDOFF — Lane A (steering controls) + Lane B (re-judge), for an A100 box

**You are Claude on a Lambda A100 instance.** Budget: ~1 hour, 1–3 A100s. Goal:
close the three GPU-cheap holes that make the *positive opinion-steering* paper
defensible, and (Lane B, API-only) unify judging + attach CIs. This branch
(`fk/lane-a-controls`) is based on `fk/init-prompt-control`, which already has the
prompt-vs-steer infra and the forced-contrast opinion vector.

> Read `CLAUDE.md` and `RESEARCH_CONTRACT.md` first. This paper is the **positive
> steering** result (Farhan's line), *not* the frozen θ shared-mechanism paper. Do
> **not** expand scope. If a step needs a decision not covered here, stop and write
> the question into `runs/<id>/OPEN_QUESTIONS.md` rather than guessing.

## Non-negotiable guardrails (from CLAUDE.md §3–§6)
- `enable_thinking=False`, `max_tokens ≥ 512`. At 128 tokens Qwen3-8B truncates
  mid-`<think>` and every metric silently computes on garbage
  (`docs/HANDOFF_prompt_control.md` §0.3). **Persist raw completions.**
- **Assert every steering vector is shape `(36, 4096)` before use.** A 1-D tensor
  silently broadcasts to a scalar DC offset (the retracted 2025 bug). Fail loudly.
- Random control is **covariance-matched, not norm-matched** (see §Exp-2).
- The **coherence gate is independent of the judge** — perplexity + distinct-n /
  repetition over the *text*, never the judge's own "nonsense" bucket.
- Say "**an** opinion direction", never "the direction" (non-identifiability,
  arXiv:2602.06801).
- **≥3 seeds** (generation) on every reported rate; report item-bootstrap CIs.
- Numbers trace to artifacts under `runs/`. No hand-edited conclusions.

## One-time setup (~2 min)
```bash
git fetch origin
# judge v2.1 (9-way) + kappa tool — pull into this branch's tree:
git checkout origin/algo-updated-judge -- src/bias_steer/judges/ docs/judges/judge_v2.1.md scripts/kappa_from_csv.py
# adaptive/fixed configs + operators for Exp-3 (if you run the frontier):
git checkout origin/fk/adaptive-steering-qwen3-run -- configs/exp/adaptive_add_linear_c20_qwen3_8b.py configs/exp/adaptive_add_linear_c30_qwen3_8b.py
export OPENAI_API_KEY=...   # needed for judging (Lane B + judging new gens)
python -c "import transformer_lens, torch; print('TL', torch.cuda.device_count(), 'GPUs')"
```

### Apply the identity-decline → `5 non-engagement` rule BEFORE judging
Team decision 2026-09-26 (see `datasets/Calibration/fvoa_labelling.xlsx` Rubric
tab): capability/identity declines ("I'm an AI / I can't taste, so I can't
answer") are **`5 non-engagement`**, not `4 soft refusal`. `judge_v2.1.md`
currently routes them to `4`. Patch the judge system-prompt string in
`docs/judges/judge_v2.1.md` (and wherever `JudgeSpec.rubric` is built in
`src/bias_steer/judges/v2.py`): add to the `non-engagement` bullet —
*"Includes capability/identity declines ('I'm an AI', 'I can't taste, so I can't
answer') that cite the model's nature and do not weigh the options"* — and to the
`soft-refusal` bullet the caveat that a bare "I can't because I'm an AI" without
engaging the options is `non-engagement`. Commit that patch first so all judged
numbers below use the agreed rubric.

---

## Fan-out across GPUs
Run one job per GPU with `CUDA_VISIBLE_DEVICES`. If you have 1 GPU, do them in
order and stop where the hour ends — **Exp-1 is non-negotiable.**

## Exp-1 (GPU 0) — PIVOTAL: prompt-vs-steer on IssueBench
Does the fitted direction beat simply asking, on prompts a system prompt does
*not* saturate? (Toy comparison prompts are at ceiling — prompt hits 100% — so
they cannot test this; IssueBench gives an 86/91% real contest.)

- Config: `configs/prompt_baseline_issuebench.py`, `intervention=both`.
- Vector: the **forced-contrast** opinion vector
  `runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b/steering_vector.safetensors`
  (assert `(36,4096)`). Doses: the coherent headline dose from Exp-3 (start c=8
  per config default; if Exp-3 shows a higher coherent dose, use it).
- `n=200`, **seeds {0,1,2}**, `enable_thinking=False`, `max_tokens=512`.
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.bias_steer run configs/prompt_baseline_issuebench.py \
  --intervention both --vector runs/20260923-090514_extract-issuebench-opinion-forced-contrast_qwen3-8b/steering_vector.safetensors \
  --n 200 --seeds 0,1,2 --enable_thinking false --max_tokens 512 --out runs/laneA_exp1_prompt_vs_steer_issuebench
```
- **DoE:** per-item paired `beat_rate` (steer vs prompt) with item-bootstrap 90%
  CI, on the **opinion** pole and the **neutral** pole, judged by pinned v2.1.
  Deliver `runs/laneA_exp1_*/summary.md` with: steer%, prompt%, beat_rate + CI,
  `steer-only` discordant count. Filter IssueBench scraping artifacts
  (`"1 / 1…Share Prompt"`) before reporting.
- **Decision:** steer beats prompt (CI excludes 0) → the "steering adds value over
  prompting" clause stands. Prompt matches/beats steer on modern prompts → reframe
  to "adaptive-linear beats *both* fixed steer and prompt" (kill-criterion, see
  `algoverse_iclr_readiness.md` §6).

## Exp-2 (GPU 1) — covariance-matched random-direction control
Kills "it's the dose, not the direction." At each reported dose, a random vector
**matched on norm AND residual covariance** must move the opinion rate ≈0, and the
opinion direction must exceed it by ≥4 SE.

Build the control from cached residuals (a few LOC — norm-matched already exists
in older sweeps; add covariance-matching):
```python
# resids: (N, d) fp32 residuals at the extraction cell; v_opin: (d,) unit opinion dir
import torch
C = torch.from_numpy(np.cov(resids.T)).float()          # (d,d) residual covariance
L = torch.linalg.cholesky(C + 1e-4*torch.eye(C.shape[0]))
z = torch.randn(C.shape[0]); r = L @ z                    # covariance-shaped noise
r = r / r.norm() * v_opin.norm()                          # match the opinion dir's norm
# tile to (36,4096) exactly as v_opin is applied; assert shape before use
```
- Apply at doses **{20, 30}** on the comparison set AND IssueBench, `n=200`,
  seeds {0,1,2}, thinking off, 512 tok. Save to `runs/laneA_exp2_covrandom_*`.
- **DoE:** table of opinion-rate(opinion dir) vs opinion-rate(cov-random) per dose
  with CIs; report the SE gap. Pass = random ≈ baseline, opinion dir ≫ random (≥4 SE).

## Exp-3 (GPU 2) — coherence gate (build once, score everything)
Kills "your 95% is just broken text" (the c40-INVALID risk). **Independent of the
judge.** Score every completion — existing adaptive/fixed runs AND the new Exp-1/2
gens.
```python
# scripts/coherence_gate.py  (self-contained; run on any completions CSV w/ a 'response' col)
import argparse, math, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def distinct_n(text, n=3):
    toks=text.split()
    if len(toks)<n: return 1.0
    grams=[tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(grams))/max(1,len(grams))
def max_repeat_run(text):          # longest run of an immediately-repeated line/phrase
    lines=[l.strip() for l in text.splitlines() if l.strip()]
    best=cur=1
    for i in range(1,len(lines)):
        cur=cur+1 if lines[i]==lines[i-1] else 1; best=max(best,cur)
    return best
def ppl(text, model, tok, dev):
    ids=tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
    if ids.shape[1]<2: return float("nan")
    with torch.no_grad(): loss=model(ids, labels=ids).loss
    return math.exp(loss.item())
# ref LM = a *clean* small model (e.g. Qwen2.5-1.5B) so perplexity is independent of the steered model
# Flag a completion INCOHERENT if: distinct_3 < 0.5  OR  max_repeat_run >= 4  OR  ppl > P95(unsteered baseline)
```
- **DoE:** for every dose × arm × run, report opinion-rate **and** coherence-pass
  rate on the same axis. Define the **reportable dose = the largest dose whose
  coherence-pass rate ≥ the unsteered baseline's**. Confirm c40 fails (validates
  the gate); confirm c30's 138/0 good/bad is genuinely coherent. Save
  `runs/<id>/coherence.csv` next to each run.

---

## Lane B (no GPU — API only; folded here because it can't run on the Windows box)
The Windows box has no `OPENAI_API_KEY` and no `openai` install, and the big-run
generations aren't local — so re-judging belongs here.
1. **Re-judge under pinned v2.1** (with the identity→5 patch above): the 7-model
   fixed-add baseline (`experiments/past_logs/.../Better_Coeff_tests` per-item
   logs) **and** the Qwen3-8B adaptive/fixed runs, so every headline count uses
   one judge. Pattern: `scripts/run_calib_v2_judge.py` / `judges/v2.py`. Persist
   the raw 9-way verdict + the collapsed-6.
2. **Item-bootstrap CIs + ≥3-seed margins** on every rate (`scripts/bootstrap_ci.py`
   below). The judge drifts ±1–2/100 at temp 0 — single-run margins inside that
   band are not results.
3. **Judge↔human κ** once Lane C (the `fvoa_labelling.xlsx` human pass) returns:
   `python scripts/kappa_from_csv.py --a judged_v2.1.csv --b human_labels.csv --collapse`
   mapping BOTH sides through `{unjudgeable,incoherent,meta-comment,unclassifiable}→ignored`.

```python
# scripts/bootstrap_ci.py — item-level bootstrap CI for a judged rate (self-contained)
import argparse, numpy as np, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--label_col",default="judge_label")
ap.add_argument("--positive",default="soft-refusal"); ap.add_argument("--B",type=int,default=10000)
a=ap.parse_args(); df=pd.read_csv(a.csv); y=(df[a.label_col]==a.positive).to_numpy().astype(float)
rng=np.random.default_rng(0); n=len(y)
boot=[y[rng.integers(0,n,n)].mean() for _ in range(a.B)]
lo,hi=np.percentile(boot,[5,95]); print(f"{a.positive}: {y.mean():.3f}  90% CI [{lo:.3f},{hi:.3f}]  n={n}")
```

## Deliverables / definition of done
Commit under `runs/laneA_*` (raw completions + `summary.md` + `coherence.csv` +
provenance manifest with `model_spec.revision`, config path, vector md5, seeds).
Update a top-level `runs/LANE_A_RESULTS.md` with the three tables (prompt-vs-steer
beat_rate+CI; cov-random vs opinion-dir; opinion-rate×coherence per dose) and a
one-line verdict per experiment mapped to the reviewer objection it closes.
**A task is done when its evidence exists and validates — not when it ran.**

## Will NOT fit in the hour (don't attempt)
7-model breadth *re-generation*; a 2nd modern non-Qwen model; the transfer arm;
the writing/reframing. Those are the multi-day remainder.
